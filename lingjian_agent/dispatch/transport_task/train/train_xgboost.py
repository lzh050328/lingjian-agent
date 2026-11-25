"""
XGBoost 模型训练器

职责：
    - 从 MongoDB 读取调度决策记录（真实数据 or 模拟数据）
    - 对特征进行预处理与标准化
    - 训练 XGBoost 二分类模型（预测是否延误）
    - 将模型与 Scaler 一起持久化，供推理阶段使用
"""
import os
import pickle
from datetime import datetime
from typing import Dict, Optional, Tuple

import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from lingjian_config import config_manager, logger
from lingjian_agent.dispatch.transport_task.repository import dispatch_decision_repository
from lingjian_util import get_project_root

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 参与训练的特征列（与 MongoDB 文档中的字段名保持一致）
FEATURE_COLUMNS: list[str] = [
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

# 预测目标列
TARGET_COLUMN: str = "isDelayed"

# XGBoost 默认超参数
# 注意：early_stopping_rounds 在 XGBoost >= 1.6 后必须在构造函数中指定，
#       不再作为 fit() 的参数传入。
DEFAULT_MODEL_PARAMS: Dict = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 8,              # 增加深度以提高模型表达能力
    "learning_rate": 0.05,       # 降低学习率，增加稳定性
    "n_estimators": 300,         # 进一步增加树的数量
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "early_stopping_rounds": 15, # 增加早停轮次
    "min_child_weight": 1,       # 降低最小叶子权重
    "gamma": 0.1,               # 添加gamma参数增加正则化
}


# ---------------------------------------------------------------------------
# 主类
# ---------------------------------------------------------------------------


class XGBoostTrainer:
    """
    XGBoost 模型训练器

    完整训练流程：
        1. 从 MongoDB 拉取已完成的调度记录
        2. 特征预处理（缺失值填充 + StandardScaler 标准化）
        3. 划分训练集 / 测试集并训练模型
        4. 在测试集上打印评估指标
        5. 将 (model, scaler) 一同序列化保存，确保推理时输入分布一致
    """

    def __init__(self, model_params: Optional[Dict] = None) -> None:
        """
        初始化训练器。

        Args:
            model_params: 自定义 XGBoost 超参数字典；若为 None 则使用默认值。
        """
        self.feature_columns = FEATURE_COLUMNS
        self.target_column = TARGET_COLUMN
        # 允许外部传入超参数，便于调参实验
        self.model_params = model_params if model_params is not None else DEFAULT_MODEL_PARAMS.copy()

        logger.info("XGBoostTrainer 初始化完成，特征列数：%d", len(self.feature_columns))

    # ------------------------------------------------------------------
    # 私有方法
    # ------------------------------------------------------------------

    def _get_data_from_mongodb(
        self,
        use_mock_data: bool = False,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 10_000,
    ) -> pd.DataFrame:
        """
        从 MongoDB 拉取调度决策记录并转换为 DataFrame。

        使用 mock 数据时，会临时将 repository 的 collection 切换到模拟集合，
        并在 finally 块中无论是否异常都恢复原始集合，避免全局状态污染。

        Args:
            use_mock_data: True 时读取模拟集合，False 时读取真实集合。
            start_date:    数据筛选起始时间（None 表示不限）。
            end_date:      数据筛选结束时间（None 表示不限）。
            limit:         最多返回的记录条数。

        Returns:
            包含特征列与目标列的 DataFrame；若无数据则返回空 DataFrame。
        """
        # 保存原始集合引用，用于 finally 恢复
        original_collection = dispatch_decision_repository.collection

        try:
            if use_mock_data:
                # 临时切换到模拟集合
                mock_db = dispatch_decision_repository.client[
                    config_manager.get("mongo.database.ccAi")
                ]
                dispatch_decision_repository.collection = mock_db[
                    config_manager.get("mongo.collection.dispatchDecisionRecordMock")
                ]
                logger.info("已切换至模拟数据集合")
            else:
                logger.info("使用真实数据集合")

            records = dispatch_decision_repository.find_completed_records(
                start_date, end_date, limit
            )

            if not records:
                logger.warning("MongoDB 中未找到符合条件的记录")
                return pd.DataFrame()

            # 将每条记录展平为字典，再批量构建 DataFrame（比逐行 append 高效）
            data_list = [
                {
                    # ---- 特征 ----
                    "usedWeightRatio":           record.features.usedWeightRatio,
                    "usedVolumeRatio":           record.features.usedVolumeRatio,
                    "orderCount":                record.features.orderCount,
                    "avgWeight":                 record.features.avgWeight,
                    "stdWeight":                 record.features.stdWeight,
                    "maxOrderWeightRatio":       record.features.maxOrderWeightRatio,
                    "skippedOrderCount":         record.features.skippedOrderCount,
                    "remainingOrderCount":       record.features.remainingOrderCount,
                    "remainingHighPriorityRatio": record.features.remainingHighPriorityRatio,
                    "rounds":                    record.features.rounds,
                    # ---- 标签及辅助字段 ----
                    "isDelayed":      record.result.isDelayed,
                    "manualAdjusted": record.result.manualAdjusted,
                    "isFullLoaded":   record.result.isFullLoaded,
                }
                for record in records
            ]

            df = pd.DataFrame(data_list)
            logger.info("成功获取 %d 条训练记录", len(df))
            return df

        except Exception as exc:
            logger.error("从 MongoDB 获取数据时发生异常：%s", exc, exc_info=True)
            return pd.DataFrame()

        finally:
            # 无论是否异常，都恢复原始集合，防止全局状态污染
            dispatch_decision_repository.collection = original_collection

    def _prepare_features(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, StandardScaler]:
        """
        特征预处理：缺失值填充 → StandardScaler 标准化。

        注意：返回已 fit 的 scaler，调用方需将其与模型一起保存，
        以保证推理阶段与训练阶段的数据分布完全一致。

        Args:
            df: 原始 DataFrame，需包含 feature_columns 和 target_column。

        Returns:
            X_scaled: 标准化后的特征矩阵。
            y:        目标向量（0/1）。
            scaler:   已 fit 的 StandardScaler 实例。

        Raises:
            ValueError: 数据中缺少必要的列时抛出。
            Exception:  其他预处理异常。
        """
        # 校验必要列是否存在
        missing_cols = set(self.feature_columns + [self.target_column]) - set(df.columns)
        if missing_cols:
            raise ValueError(f"DataFrame 缺少以下必要列：{missing_cols}")

        delay_count = int(df[self.target_column].sum())
        normal_count = len(df) - delay_count
        logger.info(
            "数据概览 — 总量：%d，延误：%d，正常：%d，延误率：%.2f%%",
            len(df), delay_count, normal_count,
            delay_count / len(df) * 100,
        )

        try:
            X = df[self.feature_columns].copy()
            y = df[self.target_column].copy()

            # 用各列均值填充缺失值（简单可靠，适合数值型特征）
            X.fillna(X.mean(numeric_only=True), inplace=True)

            # fit_transform 并保留列名与索引，便于后续特征重要性分析
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X),
                columns=X.columns,
                index=X.index,
            )

            logger.info("特征预处理完成")
            return X_scaled, y, scaler

        except Exception as exc:
            logger.error("特征预处理时发生异常：%s", exc, exc_info=True)
            raise

    def _train_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> xgb.XGBClassifier:
        """
        划分数据集并训练 XGBoost 模型，训练完成后在测试集上打印评估指标。

        Args:
            X: 标准化后的特征矩阵。
            y: 目标向量。

        Returns:
            训练完成的 XGBClassifier 实例。

        Raises:
            Exception: 训练过程中的任何异常。
        """
        try:
            # stratify=y 保证训练集与测试集中正负样本比例一致
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            logger.info("训练集：%d 条，测试集：%d 条", len(X_train), len(X_test))

            model = xgb.XGBClassifier(**self.model_params)

            # eval_set 用于 early stopping 监控（early_stopping_rounds 已在构造函数中指定）
            # verbose=False 避免每轮日志刷屏
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False,
            )

            # ---------- 评估 ----------
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            metrics: Dict[str, float] = {
                "accuracy":  accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall":    recall_score(y_test, y_pred, zero_division=0),
                "f1":        f1_score(y_test, y_pred, zero_division=0),
                "auc":       roc_auc_score(y_test, y_prob),
            }
            logger.info(
                "测试集评估 — 准确率：%.4f，精确率：%.4f，召回率：%.4f，F1：%.4f，AUC：%.4f",
                metrics["accuracy"], metrics["precision"],
                metrics["recall"],   metrics["f1"], metrics["auc"],
            )

            return model

        except Exception as exc:
            logger.error("模型训练时发生异常：%s", exc, exc_info=True)
            raise

    def _save_artifact(
        self,
        model: xgb.XGBClassifier,
        scaler: StandardScaler,
        model_path: str,
    ) -> bool:
        """
        将 model 与 scaler 打包序列化保存到同一文件。

        两者必须同时保存：推理时需要用同一个 scaler 对输入做归一化，
        才能保证输入分布与训练时一致，避免预测偏差。

        保存格式（pickle dict）：
            {
                "model":  XGBClassifier,
                "scaler": StandardScaler,
                "feature_columns": List[str],   # 用于推理时校验输入列
                "saved_at": str,                # 保存时间戳，便于追溯
            }

        Args:
            model:      训练完成的 XGBClassifier。
            scaler:     与模型配套的 StandardScaler。
            model_path: 目标文件路径（含文件名）。

        Returns:
            True 表示保存成功，False 表示失败。
        """
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            artifact = {
                "model":           model,
                "scaler":          scaler,
                "feature_columns": self.feature_columns,
                "saved_at":        datetime.now().isoformat(),
            }

            with open(model_path, "wb") as fp:
                pickle.dump(artifact, fp)

            logger.info("模型 artifact（model + scaler）已保存至：%s", model_path)
            return True

        except Exception as exc:
            logger.error("保存模型时发生异常：%s", exc, exc_info=True)
            return False

    # ------------------------------------------------------------------
    # 公开方法
    # ------------------------------------------------------------------

    def train_and_save(
        self,
        use_mock_data: bool = False,
        model_path: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 10_000,
    ) -> bool:
        """
        完整训练流程入口：数据获取 → 特征预处理 → 模型训练 → 持久化保存。

        Args:
            use_mock_data: True 时使用模拟集合，False 时使用真实集合。
            model_path:    模型文件保存路径；若为 None 则从配置读取。
            start_date:    数据筛选起始时间。
            end_date:      数据筛选结束时间。
            limit:         最多拉取的记录条数。

        Returns:
            True 表示训练与保存均成功，False 表示任意步骤失败。
        """
        logger.info("========== 开始训练 XGBoost 模型（%s）==========",
                    "mock 数据" if use_mock_data else "真实数据")
        try:
            # 1. 获取数据
            df = self._get_data_from_mongodb(use_mock_data, start_date, end_date, limit)
            if df.empty:
                logger.error("训练数据为空，终止训练")
                return False

            # 2. 特征预处理（同时返回 scaler，避免推理时数据分布不一致）
            X, y, scaler = self._prepare_features(df)

            # 3. 训练模型
            model = self._train_model(X, y)

            # 4. 确定保存路径
            if model_path is None:
                config_key = (
                    "ai.dispatch.model.path.mock" if use_mock_data
                    else "ai.dispatch.model.path.real"
                )
                model_path = get_project_root() / config_manager.get(config_key)

            if not model_path:
                logger.error("模型保存路径未配置（config key: %s）", config_key)
                return False

            # 5. 保存 model + scaler
            success = self._save_artifact(model, scaler, model_path)
            if success:
                logger.info("========== 模型训练完成 ==========")
            return success

        except Exception as exc:
            logger.error("训练流程中发生未预期异常：%s", exc, exc_info=True)
            return False


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------


def train_mock_model() -> bool:
    """使用模拟数据训练并保存模型。"""
    return XGBoostTrainer().train_and_save(use_mock_data=True)


def train_real_model() -> bool:
    """使用真实数据训练并保存模型。"""
    return XGBoostTrainer().train_and_save(use_mock_data=False)


# ---------------------------------------------------------------------------
# 脚本入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    trainer = XGBoostTrainer()

    logger.info("开始训练模拟数据模型...")
    ok = trainer.train_and_save(use_mock_data=True, limit=20000)
    logger.info("模拟模型训练%s", "成功" if ok else "失败")