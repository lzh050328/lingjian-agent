"""
模型管理器 (model_manager.py)

职责：
    - 根据配置动态加载和切换 XGBoost 预测模型
    - 提供线程安全的模型状态管理
    - 推理前自动应用与训练配套的 StandardScaler，保证数据分布一致

pickle artifact 格式（由 train_xgboost.py 写入）：
    {
        "model":           XGBClassifier,    # 训练好的分类模型
        "scaler":          StandardScaler,   # 与模型配套的特征标准化器
        "feature_columns": List[str],        # 训练时的特征顺序，用于推理校验
        "saved_at":        str,              # 保存时间戳，便于追溯版本
    }

⚠️  推理时必须先用 scaler 对原始特征做 transform，再送入模型，
    否则输入分布与训练时不一致，预测结果将不可信。
"""

import os
import pickle
import threading
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from ykzh_config import config_manager, logger
from ykzh_util import get_project_root


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 合法的模型模式
VALID_MODES = frozenset({"mock", "real"})

# 支持的文件格式（按加载方式分类）
_PICKLE_EXTENSIONS = frozenset({".pkl", ".pickle"})
_NATIVE_EXTENSIONS = frozenset({".json", ".ubj"})

# 默认特征维度（从配置读取，兜底值与训练脚本保持一致）
EXPECTED_FEATURE_COUNT: int = config_manager.get(
    "ai.dispatch.model.feature_count", 10
)


# ---------------------------------------------------------------------------
# 内部数据类：加载产物
# ---------------------------------------------------------------------------

@dataclass
class _ModelArtifact:
    """
    pickle 文件中的完整加载产物。

    model 和 scaler 必须成对使用：
        scaler 在训练时 fit，推理时必须用同一个 scaler 做 transform，
        才能保证输入分布与训练阶段一致。
    """
    model: xgb.XGBClassifier
    scaler: Optional[StandardScaler] = None
    feature_columns: List[str] = field(default_factory=list)
    saved_at: str = ""


# ---------------------------------------------------------------------------
# ModelManager
# ---------------------------------------------------------------------------

class ModelManager:
    """
    XGBoost 模型管理器（线程安全单例）

    加载流程：
        1. 从磁盘读取 pickle artifact（model + scaler + feature_columns）
        2. 将 model / scaler / feature_columns 分别存入实例属性

    推理流程（predict）：
        1. 校验特征格式与维度
        2. 若存在 scaler，先做 transform（与训练阶段对齐）
        3. 调用 model.predict / predict_proba，返回结构化结果

    典型用法：
        from model_manager import model_manager

        if model_manager.load_model():
            result = model_manager.predict(features)
    """

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        # 核心推理对象
        self._model: Optional[xgb.XGBClassifier] = None
        self._scaler: Optional[StandardScaler] = None     # 必须与 model 配套使用
        self._feature_columns: List[str] = []             # 训练时的特征顺序

        # 状态标志
        self._current_mode: Optional[str] = None
        self._model_loaded: bool = False
        self._lock = threading.Lock()                     # 保护多线程下的状态一致性

        # 从配置读取
        self._model_enabled: bool = config_manager.get("ai.dispatch.model.enabled", False)
        self._default_mode: str = config_manager.get("ai.dispatch.model.mode", "mock")
        self._model_paths: dict = {
            "mock": config_manager.get("ai.dispatch.model.path.mock"),
            "real": config_manager.get("ai.dispatch.model.path.real"),
        }

        logger.info(
            "ModelManager 初始化完成 | enabled=%s | default_mode=%s",
            self._model_enabled, self._default_mode,
        )

    # ------------------------------------------------------------------
    # 私有方法：路径解析
    # ------------------------------------------------------------------

    def _get_model_path(self, mode: str) -> Optional[str]:
        """
        根据模式返回对应的模型文件路径。

        将相对路径转换为基于项目根目录的绝对路径，确保无论从哪个目录
        执行代码都能正确找到模型文件。

        Args:
            mode: 模型模式，合法值见 VALID_MODES。

        Returns:
            模型文件的绝对路径；模式非法时返回 None。
        """
        if mode not in VALID_MODES:
            logger.error("非法模型模式: %r，合法值: %s", mode, VALID_MODES)
            return None

        relative_path = self._model_paths.get(mode)
        if not relative_path:
            return None

        return get_project_root() / relative_path

    # ------------------------------------------------------------------
    # 私有方法：文件加载
    # ------------------------------------------------------------------

    def _load_artifact_from_file(self, model_path: str) -> Optional[_ModelArtifact]:
        """
        从磁盘加载完整的模型 artifact（model + scaler + 元信息）。

        根据文件扩展名自动选择加载方式：
            - .pkl / .pickle : pickle 格式，解析 artifact dict
            - .json / .ubj   : XGBoost 原生格式，仅包含 model（无 scaler）

        Args:
            model_path: 模型文件路径。

        Returns:
            _ModelArtifact 实例；加载失败时返回 None。
        """
        if not os.path.exists(model_path):
            logger.error("模型文件不存在: %s", model_path)
            return None

        ext = os.path.splitext(model_path)[-1].lower()

        try:
            # ── pickle 格式 ──────────────────────────────────────────────
            if ext in _PICKLE_EXTENSIONS:
                with open(model_path, "rb") as fp:
                    raw = pickle.load(fp)

                return self._parse_pickle_artifact(raw, model_path)

            # ── XGBoost 原生格式 ──────────────────────────────────────────
            elif ext in _NATIVE_EXTENSIONS:
                model = xgb.XGBClassifier()
                model.load_model(model_path)
                # 原生格式不携带 scaler，推理时输入需已经过标准化
                logger.info("已加载 XGBoost 原生格式模型（不含 scaler）: %s", model_path)
                return _ModelArtifact(model=model)

            # ── 不支持的格式 ──────────────────────────────────────────────
            else:
                logger.error(
                    "不支持的模型文件格式: %r，支持: %s",
                    ext, _PICKLE_EXTENSIONS | _NATIVE_EXTENSIONS,
                )
                return None

        except Exception:
            logger.exception("加载模型文件时发生异常: %s", model_path)
            return None

    def _parse_pickle_artifact(
        self, raw: object, model_path: str
    ) -> Optional[_ModelArtifact]:
        """
        解析 pickle.load() 的返回值，提取 model / scaler / feature_columns。

        兼容两种保存格式：
            格式 A（当前标准，由 train_xgboost.py 写入）：
                dict，含 "model"、"scaler"、"feature_columns"、"saved_at" 键

            格式 B（旧版兼容）：
                直接保存 XGBClassifier 对象（无 scaler）

        Args:
            raw:        pickle.load() 的原始返回值。
            model_path: 模型文件路径（仅用于日志）。

        Returns:
            _ModelArtifact；无法解析时返回 None。
        """
        # ── 格式 B：直接是模型对象（旧版） ──────────────────────────────
        if isinstance(raw, xgb.XGBClassifier):
            logger.warning(
                "pickle 文件为旧版格式（仅含 model，无 scaler），"
                "推理时将跳过特征标准化，可能影响预测精度: %s",
                model_path,
            )
            return _ModelArtifact(model=raw)

        # ── 格式 A：标准 dict artifact ────────────────────────────────────
        if isinstance(raw, dict):
            logger.debug("pickle 文件为 dict，键列表: %s", list(raw.keys()))

            model = raw.get("model")
            if not isinstance(model, xgb.XGBClassifier):
                logger.error(
                    "pickle dict 中 'model' 字段类型错误: 期望 XGBClassifier，"
                    "实际 %s。文件: %s\n键及值类型: %s",
                    type(model).__name__,
                    model_path,
                    {k: type(v).__name__ for k, v in raw.items()},
                )
                return None

            scaler = raw.get("scaler")
            if not isinstance(scaler, StandardScaler):
                # scaler 缺失不阻断加载，但必须告警，因为这会导致预测偏差
                logger.warning(
                    "pickle dict 中未找到有效的 StandardScaler（实际: %s），"
                    "推理时将跳过特征标准化，可能导致预测偏差: %s",
                    type(scaler).__name__, model_path,
                )
                scaler = None

            feature_columns: List[str] = raw.get("feature_columns", [])
            saved_at: str = raw.get("saved_at", "")

            logger.info(
                "模型 artifact 解析完成 | scaler=%s | feature_columns=%d 个 | saved_at=%s",
                "✓" if scaler else "✗（缺失）",
                len(feature_columns),
                saved_at or "未记录",
            )
            return _ModelArtifact(
                model=model,
                scaler=scaler,
                feature_columns=feature_columns,
                saved_at=saved_at,
            )

        # ── 无法识别的类型 ────────────────────────────────────────────────
        logger.error(
            "pickle 文件内容类型无法识别: %s，期望 XGBClassifier 或 dict。文件: %s",
            type(raw).__name__, model_path,
        )
        return None

    # ------------------------------------------------------------------
    # 公开方法：模型生命周期
    # ------------------------------------------------------------------

    def load_model(self, mode: Optional[str] = None) -> bool:
        """
        加载指定模式的模型 artifact（线程安全）。

        若目标模式与当前已加载模式相同则直接返回 True（幂等）。
        加载成功后会同时更新 _model / _scaler / _feature_columns。

        Args:
            mode: 目标模式（"mock" / "real"）；None 时使用配置默认值。

        Returns:
            True 表示加载成功，False 表示失败。
        """
        if not self._model_enabled:
            logger.warning("模型功能未启用，跳过加载")
            return False

        target_mode = mode or self._default_mode

        with self._lock:
            # 幂等：已是目标模式且已加载则直接返回
            if self._current_mode == target_mode and self._model_loaded:
                logger.debug("模型已处于 %r 模式，无需重新加载", target_mode)
                return True

            model_path = self._get_model_path(target_mode)
            if not model_path:
                return False

            artifact = self._load_artifact_from_file(model_path)
            if artifact is None:
                return False

            # 原子更新状态（全部在锁内完成，保证可见性）
            self._model = artifact.model
            self._scaler = artifact.scaler
            self._feature_columns = artifact.feature_columns
            self._current_mode = target_mode
            self._model_loaded = True

        logger.info("模型切换完成 → %r 模式", target_mode)
        return True

    def switch_model(self, mode: str) -> bool:
        """
        切换模型模式（load_model 的语义化封装）。

        Args:
            mode: 目标模式（"mock" / "real"）。

        Returns:
            True 表示切换成功，False 表示失败。
        """
        logger.info("切换模型 → %r 模式", mode)
        return self.load_model(mode)

    def unload_model(self) -> None:
        """
        卸载当前模型并释放内存（线程安全）。

        可在需要强制重新加载或节省内存时调用。
        """
        with self._lock:
            self._model = None
            self._scaler = None
            self._feature_columns = []
            self._current_mode = None
            self._model_loaded = False
        logger.info("模型已卸载")

    # ------------------------------------------------------------------
    # 公开方法：状态查询
    # ------------------------------------------------------------------

    def is_model_ready(self) -> bool:
        """
        检查模型是否已就绪（启用 + 已加载 + 实例非空）。

        Returns:
            就绪返回 True，否则返回 False。
        """
        return (
            self._model_enabled
            and self._model_loaded
            and self._model is not None
        )

    def get_current_mode(self) -> Optional[str]:
        """
        返回当前已加载模型的模式名称。

        Returns:
            当前模式字符串；模型未加载时返回 None。
        """
        return self._current_mode if self._model_loaded else None

    def get_model_info(self) -> dict:
        """
        返回模型管理器的完整状态快照，用于监控或诊断。

        Returns:
            dict，包含：
                enabled        : 模型功能是否启用
                loaded         : 模型是否已加载到内存
                current_mode   : 当前模式（未加载时为 None）
                scaler_loaded  : scaler 是否已加载（True 才会做标准化）
                feature_columns: 当前模型的特征列（空列表表示未加载或旧版格式）
                paths          : 各模式对应的文件路径
                files_exist    : 各模式对应文件是否实际存在于磁盘
        """
        return {
            "enabled":         self._model_enabled,
            "loaded":          self._model_loaded,
            "current_mode":    self._current_mode,
            "scaler_loaded":   self._scaler is not None,
            "feature_columns": self._feature_columns,
            "paths":           dict(self._model_paths),
            "files_exist": {
                mode: bool(path and os.path.exists(path))
                for mode, path in self._model_paths.items()
            },
        }

    # ------------------------------------------------------------------
    # 公开方法：特征校验与预测
    # ------------------------------------------------------------------

    def validate_features(self, features: list) -> bool:
        """
        校验输入特征的格式和维度是否符合模型要求。

        校验规则（按顺序）：
            1. 必须是 list 类型
            2. 长度必须等于 feature_columns 长度（或 EXPECTED_FEATURE_COUNT）
            3. 每个元素必须是 int 或 float，且不含 NaN / Inf

        Args:
            features: 待校验的特征列表。

        Returns:
            校验通过返回 True，否则返回 False。
        """
        if not isinstance(features, list):
            logger.error("特征数据类型错误: 期望 list，实际 %s", type(features).__name__)
            return False

        expected = len(self._feature_columns) if self._feature_columns else EXPECTED_FEATURE_COUNT
        if len(features) != expected:
            logger.error("特征维度不匹配: 期望 %d，实际 %d", expected, len(features))
            return False

        for idx, val in enumerate(features):
            if not isinstance(val, (int, float)):
                logger.error(
                    "特征[%d] 类型错误: 期望 int/float，实际 %s",
                    idx, type(val).__name__,
                )
                return False
            if np.isnan(val) or np.isinf(val):
                logger.error("特征[%d] 含非法值（NaN 或 Inf）: %s", idx, val)
                return False

        return True

    def predict(self, features: list) -> Optional[dict]:
        """
        使用当前已加载模型对输入特征进行分类预测。

        推理流程：
            1. 检查模型就绪状态
            2. 校验特征格式与维度
            3. 若存在 scaler，先做 transform（与训练时对齐）  ← 关键步骤
            4. 调用 model.predict / predict_proba
            5. 返回结构化预测结果

        Args:
            features: 原始（未标准化）特征列表，顺序需与训练时 FEATURE_COLUMNS 一致。

        Returns:
            预测结果字典，包含：
                prediction       (bool) : True 表示"预测为延误"
                prediction_label (str)  : "delayed" / "not_delayed"
                probability      (dict) : 各类别置信度
                    - not_delayed (float)
                    - delayed     (float)
                confidence       (float): 最高类别的置信度
                scaler_applied   (bool) : 本次推理是否应用了 scaler
                model_mode       (str)  : 当前使用的模型模式
            失败时返回 None。
        """
        if not self.is_model_ready():
            logger.error("模型未就绪，无法执行预测")
            return None

        if not self.validate_features(features):
            return None

        try:
            # 用 DataFrame 包裹特征，列名与训练时 scaler.fit 的 DataFrame 保持一致，
            # 避免 sklearn 发出 "X does not have valid feature names" UserWarning。
            # 若 feature_columns 未记录（旧版模型），则退化为 numpy array。
            if self._feature_columns:
                X = pd.DataFrame([features], columns=self._feature_columns)
            else:
                X = np.array([features], dtype=float)  # shape: (1, n_features)

            # ── 关键：推理前应用 scaler，与训练阶段对齐 ──────────────────
            # scaler 在训练时用全量训练集 fit，推理时必须用同一个实例 transform，
            # 否则特征数值范围与训练时不一致，模型预测会产生系统性偏差。
            scaler_applied = False
            if self._scaler is not None:
                X = self._scaler.transform(X)
                scaler_applied = True
            else:
                logger.warning("scaler 未加载，本次推理跳过特征标准化（可能影响预测精度）")

            prediction: int = int(self._model.predict(X)[0])
            proba: np.ndarray = self._model.predict_proba(X)[0]

            result = {
                "prediction":       bool(prediction),
                "prediction_label": "delayed" if prediction else "not_delayed",
                "probability": {
                    "not_delayed": float(proba[0]),
                    "delayed":     float(proba[1]),
                },
                "confidence":     float(proba.max()),
                "scaler_applied": scaler_applied,
                "model_mode":     self._current_mode,
            }

            logger.info(
                "预测完成 | label=%s | confidence=%.4f | scaler=%s | mode=%s",
                result["prediction_label"],
                result["confidence"],
                "✓" if scaler_applied else "✗",
                self._current_mode,
            )
            return result

        except Exception:
            logger.exception("模型推理时发生异常")
            return None


# ---------------------------------------------------------------------------
# 全局单例
# ---------------------------------------------------------------------------

# 模块级单例，供其他模块直接 import 使用：
#   from model_manager import model_manager
model_manager = ModelManager()
model_manager.load_model()

# ---------------------------------------------------------------------------
# 本地测试入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 打印初始状态
    logger.info("模型信息: %s", model_manager.get_model_info())

    # 尝试加载默认模型
    if model_manager.load_model():
        logger.info(
            "加载成功 | mode=%s | scaler=%s",
            model_manager.get_current_mode(),
            "✓" if model_manager._scaler else "✗（缺失）",
        )

        # 构造测试特征（原始值，predict 内部会自动 scaler.transform）
        test_features = [0.1 * i for i in range(EXPECTED_FEATURE_COUNT)]
        result = model_manager.predict(test_features)
        logger.info("预测结果: %s", result)
    else:
        logger.error("模型加载失败，请检查配置和文件路径")