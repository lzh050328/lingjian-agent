"""
预测服务 (predict_service.py)

职责：
    - 对外提供调度延误预测 API，屏蔽底层 ModelManager 的实现细节
    - 负责业务层特征校验（类型、值域）
    - 提供单条预测、批量预测、模型状态查询、模式切换等功能

与 ModelManager 的分工：
    - PredictService : 业务语义校验（字段存在性、值域合法性）、响应格式封装
    - ModelManager   : 特征维度校验、scaler 标准化、模型推理
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ykzh_config import logger
from ykzh_agent.dispatch.transport_task.service import VALID_MODES, model_manager


# ---------------------------------------------------------------------------
# 特征元信息
# ---------------------------------------------------------------------------

# 特征名称列表（顺序与训练时 FEATURE_COLUMNS 保持严格一致）
# ⚠️ 此处为唯一数据源，ModelManager 的 feature_columns 在加载 artifact 后
#    也来自同一训练脚本，两者应保持同步。
FEATURE_NAMES: List[str] = [
    "usedWeightRatio",
    "usedVolumeRatio",
    "orderCount",
    "avgWeight",
    "stdWeight",
    "maxOrderWeightRatio",
    "skippedOrderCount",
    "remainingOrderCount",
    "remainingHighPriorityRatio",
    "rounds",
]

# 特征值域约束：key 为特征名，value 为 (min, max) 闭区间
# 仅对有明确业务约束的特征配置，其余特征只做类型校验
_FEATURE_RANGE: Dict[str, tuple] = {
    "usedWeightRatio":            (0.0, 1.0),
    "usedVolumeRatio":            (0.0, 1.0),
    "remainingHighPriorityRatio": (0.0, 1.0),
    "maxOrderWeightRatio":        (0.0, 1.0),
    # 计数类特征：非负整数（上界不限）
    "orderCount":            (0, float("inf")),
    "skippedOrderCount":     (0, float("inf")),
    "remainingOrderCount":   (0, float("inf")),
    "rounds":                (0, float("inf")),
}


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    """返回当前 UTC 时间的 ISO 8601 字符串（含时区信息）。"""
    return datetime.now(timezone.utc).isoformat()


def _error_response(error: str, **extra) -> Dict[str, Any]:
    """
    构造统一的失败响应。

    Args:
        error: 错误描述信息。
        **extra: 需要附加到响应中的其他字段。

    Returns:
        包含 success=False 和 error 字段的字典。
    """
    return {"success": False, "error": error, **extra}


# ---------------------------------------------------------------------------
# PredictService
# ---------------------------------------------------------------------------

class PredictService:
    """
    调度延误预测服务。

    提供从"原始特征字典"到"结构化预测结果"的完整业务流程封装：
        输入 Dict[str, float]  →  业务校验  →  特征向量  →  模型推理  →  结构化响应

    典型用法：
        from predict_service import predict_service

        result = predict_service.predict_delay({
            "usedWeightRatio": 0.85,
            "orderCount": 15,
            ...
        })
    """

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        self.feature_names: List[str] = FEATURE_NAMES
        logger.info("PredictService 初始化完成，特征列数: %d", len(self.feature_names))

    # ------------------------------------------------------------------
    # 私有方法：业务层特征校验
    # ------------------------------------------------------------------

    def _validate_features(self, features: Dict[str, Any]) -> Optional[str]:
        """
        业务层特征校验（字段存在性 + 类型 + 值域）。

        与 ModelManager.validate_features 的分工：
            - 本方法：接收 dict，校验业务语义（哪些字段必须有、值域是否合理）
            - ModelManager：接收 list，校验维度与数值合法性（NaN/Inf）

        Args:
            features: 原始特征字典。

        Returns:
            校验通过返回 None；失败返回描述错误原因的字符串。
        """
        # 1. 必需字段存在性检查
        missing = [f for f in self.feature_names if f not in features]
        if missing:
            return f"缺少必需特征: {missing}"

        # 2. 类型 + 值域检查
        for name, value in features.items():
            if name not in self.feature_names:
                continue  # 忽略多余字段，不作强制拦截

            if not isinstance(value, (int, float)):
                return f"特征 {name!r} 类型错误: 期望 int/float，实际 {type(value).__name__}"

            if name in _FEATURE_RANGE:
                lo, hi = _FEATURE_RANGE[name]
                if not (lo <= value <= hi):
                    return f"特征 {name!r} 值 {value} 超出合法范围 [{lo}, {hi}]"

        return None

    def _build_feature_vector(self, features: Dict[str, Any]) -> List[float]:
        """
        将特征字典按 FEATURE_NAMES 顺序转换为特征向量。

        严格按照 feature_names 的顺序提取，确保与训练时列顺序一致。
        调用前应已通过 _validate_features 校验。

        Args:
            features: 已通过业务校验的特征字典。

        Returns:
            有序特征值列表。
        """
        return [float(features[name]) for name in self.feature_names]

    # ------------------------------------------------------------------
    # 公开方法：单条预测
    # ------------------------------------------------------------------

    def predict_delay(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        对单条调度特征进行延误预测。

        Args:
            features: 调度特征字典，必须包含 FEATURE_NAMES 中的所有键。

        Returns:
            响应字典，包含：
                success    (bool): 是否成功
                error      (str | None): 错误描述，成功时为 None
                prediction (dict | None): 模型预测结果（见 ModelManager.predict）
                timestamp  (str): UTC 时间戳（ISO 8601）
        """
        # 1. 业务层特征校验
        error_msg = self._validate_features(features)
        if error_msg:
            logger.warning("特征校验未通过: %s", error_msg)
            return _error_response(error_msg, prediction=None)

        # 2. 构造特征向量（顺序严格对齐训练时的列顺序）
        feature_vector = self._build_feature_vector(features)

        # 3. 调用模型推理（ModelManager 内部负责 scaler transform）
        prediction_result = model_manager.predict(feature_vector)
        if prediction_result is None:
            return _error_response("模型预测失败，请检查模型是否已加载", prediction=None)

        logger.info(
            "单条预测完成 | label=%s | confidence=%.4f",
            prediction_result.get("prediction_label"),
            prediction_result.get("confidence", 0),
        )
        return {
            "success":    True,
            "error":      None,
            "prediction": prediction_result,
            "timestamp":  _utc_now_iso(),
        }

    # ------------------------------------------------------------------
    # 公开方法：批量预测
    # ------------------------------------------------------------------

    def batch_predict_delay(
        self, features_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        对多条调度特征进行批量延误预测。

        单条失败不影响其他条目，失败条目的 prediction 字段为 None。

        Args:
            features_list: 特征字典列表，每个元素格式同 predict_delay 的入参。

        Returns:
            响应字典，包含：
                success       (bool): 整体流程是否成功（列表非空即 True）
                error         (str | None): 整体错误描述
                predictions   (list): 每条记录的预测结果列表
                summary       (dict): 统计摘要（total / success_count / error_count）
                timestamp     (str): UTC 时间戳（ISO 8601）
        """
        if not features_list:
            return _error_response("特征列表为空", predictions=[])

        results: List[Dict[str, Any]] = []
        success_count = 0

        for idx, features in enumerate(features_list):
            try:
                result = self.predict_delay(features)
                results.append(result)
                if result["success"]:
                    success_count += 1
            except Exception:
                # 单条异常不中断整批，记录日志后继续
                logger.exception("批量预测第 %d 条记录时发生异常", idx)
                results.append(_error_response("处理时发生未预期异常", prediction=None))

        error_count = len(features_list) - success_count
        logger.info(
            "批量预测完成 | 总计 %d 条 | 成功 %d | 失败 %d",
            len(features_list), success_count, error_count,
        )
        return {
            "success":     True,
            "error":       None,
            "predictions": results,
            "summary": {
                "total":         len(features_list),
                "success_count": success_count,
                "error_count":   error_count,
            },
            "timestamp": _utc_now_iso(),
        }

    # ------------------------------------------------------------------
    # 公开方法：模型管理
    # ------------------------------------------------------------------

    def get_model_status(self) -> Dict[str, Any]:
        """
        获取当前模型的状态快照。

        Returns:
            响应字典，包含：
                success       (bool)
                model_status  (dict): 来自 ModelManager.get_model_info()
                feature_names (list): 本服务使用的特征列表
                timestamp     (str)
        """
        try:
            return {
                "success":       True,
                "error":         None,
                "model_status":  model_manager.get_model_info(),
                "feature_names": self.feature_names,
                "timestamp":     _utc_now_iso(),
            }
        except Exception:
            logger.exception("获取模型状态时发生异常")
            return _error_response("获取模型状态异常", model_status=None)

    def switch_model_mode(self, mode: str) -> Dict[str, Any]:
        """
        切换模型模式（mock ↔ real）。

        Args:
            mode: 目标模式，合法值见 ModelManager.VALID_MODES（"mock" / "real"）。

        Returns:
            响应字典，包含：
                success      (bool)
                error        (str | None)
                message      (str): 成功描述
                current_mode (str): 切换后的实际模式
                timestamp    (str)
        """
        # 复用 ModelManager 的 VALID_MODES，避免在两处维护合法值列表
        if mode not in VALID_MODES:
            return _error_response(
                f"无效的模型模式: {mode!r}，合法值: {VALID_MODES}",
                current_mode=model_manager.get_current_mode(),
            )

        try:
            success = model_manager.switch_model(mode)
            if success:
                logger.info("模型模式切换成功 → %r", mode)
                return {
                    "success":      True,
                    "error":        None,
                    "message":      f"已切换至 {mode!r} 模式",
                    "current_mode": mode,
                    "timestamp":    _utc_now_iso(),
                }
            return _error_response(
                f"切换至 {mode!r} 模式失败，请检查模型文件是否存在",
                current_mode=model_manager.get_current_mode(),
            )
        except Exception:
            logger.exception("切换模型模式时发生异常")
            return _error_response(
                "切换模型模式时发生异常",
                current_mode=model_manager.get_current_mode(),
            )


# ---------------------------------------------------------------------------
# 全局单例
# ---------------------------------------------------------------------------

# 模块级单例，供其他模块直接 import 使用：
#   from predict_service import predict_service
predict_service = PredictService()


# ---------------------------------------------------------------------------
# 本地测试入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 打印模型状态
    logger.info("模型状态: %s", predict_service.get_model_status())

    # 单条预测测试
    test_features = {
        "usedWeightRatio":            0.85,
        "usedVolumeRatio":            0.75,
        "orderCount":                 15,
        "avgWeight":                  128.5,
        "stdWeight":                  36.2,
        "maxOrderWeightRatio":        0.15,
        "skippedOrderCount":          2,
        "remainingOrderCount":        5,
        "remainingHighPriorityRatio": 0.25,
        "rounds":                     3,
    }
    result = predict_service.predict_delay(test_features)
    logger.info("单条预测结果: %s", result)

    # 批量预测测试
    batch_result = predict_service.batch_predict_delay([test_features, test_features])
    logger.info(
        "批量预测摘要: %s", batch_result.get("summary")
    )