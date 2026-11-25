"""
快递员调度预测器

该类负责加载、管理和使用XGBoost模型进行快递员配送预测。
主要功能包括：
1. 自动加载已训练的模型或触发训练流程
2. 提供模型预测接口
3. 支持批量和单个样本预测
"""

import os
import joblib
import pandas as pd
from typing import List, Union, Dict, Any
from xgboost import XGBClassifier
from ykzh_config import logger


class CourierDispatchPredictor:
    """快递员调度预测器

    该类负责加载、管理和使用XGBoost模型进行快递员配送预测。
    主要功能包括：
    1. 自动加载已训练的模型或触发训练流程
    2. 提供模型预测接口
    3. 支持批量和单个样本预测
    """

    def __init__(self, model_path: str = "xgb_model.pkl", data_path: str = "../data/courier_data.csv"):
        """
        初始化快递员调度预测器

        Args:
            model_path: 模型文件路径，相对于当前文件位置
            data_path: 训练数据路径，相对于当前文件位置
        """
        # 获取当前文件所在目录的绝对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # 构建模型和数据的绝对路径
        self.model_path = os.path.join(current_dir, model_path)
        self.data_path = os.path.join(current_dir, data_path)

        # 模型实例
        self.model: XGBClassifier = None

        # 自动加载模型
        self._load_or_train_model()

    def _load_or_train_model(self) -> None:
        """
        内部方法：加载模型或触发训练

        如果模型文件存在，则直接加载
        如果模型文件不存在，则调用训练脚本生成模型
        """
        if os.path.exists(self.model_path):
            logger.info(f"发现已训练的模型文件: {self.model_path}")
            try:
                self.model = joblib.load(self.model_path)
                logger.info("模型加载成功")
            except Exception as e:
                logger.error(f"模型加载失败: {e}")
                logger.info("将重新训练模型...")
                self._train_model()
        else:
            logger.warning(f"未找到模型文件: {self.model_path}")
            logger.info("将开始训练新模型...")
            self._train_model()

    def _train_model(self) -> None:
        """
        内部方法：训练新模型

        调用train.py中的训练逻辑来生成新的模型
        """
        try:
            # 直接执行训练脚本
            # 注意：这里会执行train.py中的所有代码，包括模型训练和保存
            exec(open("train.py").read())

            # 训练完成后重新加载模型
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                logger.info("新模型训练并加载成功")
            else:
                raise FileNotFoundError("训练完成后未找到模型文件")

        except Exception as e:
            logger.error(f"模型训练失败: {e}")
            raise RuntimeError(f"无法训练模型: {e}")

    def predict(self, features: Union[List[float], Dict[str, float], pd.DataFrame]) -> Union[int, List[int]]:
        """
        预测快递员是否能够按时完成配送

        Args:
            features: 输入特征，支持以下格式：
                - List[float]: [task_count, distance, idle_time, hour]
                - Dict[str, float]: {'task_count': x, 'distance': x, 'idle_time': x, 'hour': x}
                - pd.DataFrame: 包含特征列的数据框

        Returns:
            预测结果：
                - 单个样本返回 int (0=未按时, 1=按时)
                - 多个样本返回 List[int]

        Raises:
            ValueError: 输入特征格式错误
            RuntimeError: 模型未加载
        """
        if self.model is None:
            raise RuntimeError("模型未加载，请先调用_load_or_train_model()")

        # 处理不同的输入格式
        if isinstance(features, list) and len(features) == 4:
            # 单个样本的列表格式
            X = pd.DataFrame([features], columns=["task_count", "distance", "idle_time", "hour"])
            prediction = self.model.predict(X)[0]
            return int(prediction)

        elif isinstance(features, dict):
            # 单个样本的字典格式
            required_keys = ["task_count", "distance", "idle_time", "hour"]
            if not all(key in features for key in required_keys):
                raise ValueError(f"字典必须包含所有必需的特征: {required_keys}")

            X = pd.DataFrame([list(features.values())], columns=required_keys)
            prediction = self.model.predict(X)[0]
            return int(prediction)

        elif isinstance(features, pd.DataFrame):
            # DataFrame格式（支持批量预测）
            required_columns = ["task_count", "distance", "idle_time", "hour"]
            if not all(col in features.columns for col in required_columns):
                raise ValueError(f"DataFrame必须包含所有必需的特征列: {required_columns}")

            X = features[required_columns]
            predictions = self.model.predict(X)
            return [int(pred) for pred in predictions]

        else:
            raise ValueError(
                "不支持的特征格式。请使用: "
                "1) List[float]: [task_count, distance, idle_time, hour] "
                "2) Dict[str, float]: {'task_count': x, 'distance': x, 'idle_time': x, 'hour': x} "
                "3) pd.DataFrame: 包含必需特征列的数据框"
            )

    def predict_proba(self, features: Union[List[float], Dict[str, float], pd.DataFrame]) -> Union[List[float], List[List[float]]]:
        """
        预测快递员按时完成配送的概率

        Args:
            features: 输入特征（格式同predict方法）

        Returns:
            概率预测结果：
                - 单个样本返回 [prob_class_0, prob_class_1]
                - 多个样本返回 [[prob_class_0, prob_class_1], ...]
        """
        if self.model is None:
            raise RuntimeError("模型未加载，请先调用_load_or_train_model()")

        # 处理不同的输入格式（与predict方法类似）
        if isinstance(features, list) and len(features) == 4:
            X = pd.DataFrame([features], columns=["task_count", "distance", "idle_time", "hour"])
            proba = self.model.predict_proba(X)[0]
            return proba.tolist()

        elif isinstance(features, dict):
            required_keys = ["task_count", "distance", "idle_time", "hour"]
            if not all(key in features for key in required_keys):
                raise ValueError(f"字典必须包含所有必需的特征: {required_keys}")

            X = pd.DataFrame([list(features.values())], columns=required_keys)
            proba = self.model.predict_proba(X)[0]
            return proba.tolist()

        elif isinstance(features, pd.DataFrame):
            required_columns = ["task_count", "distance", "idle_time", "hour"]
            if not all(col in features.columns for col in required_columns):
                raise ValueError(f"DataFrame必须包含所有必需的特征列: {required_columns}")

            X = features[required_columns]
            probabilities = self.model.predict_proba(X)
            return probabilities.tolist()

        else:
            raise ValueError(
                "不支持的特征格式。请使用: "
                "1) List[float]: [task_count, distance, idle_time, hour] "
                "2) Dict[str, float]: {'task_count': x, 'distance': x, 'idle_time': x, 'hour': x} "
                "3) pd.DataFrame: 包含必需特征列的数据框"
            )

    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息和状态

        Returns:
            包含模型信息的字典
        """
        if self.model is None:
            return {
                "status": "未加载",
                "model_path": self.model_path,
                "data_path": self.data_path
            }

        return {
            "status": "已加载",
            "model_path": self.model_path,
            "data_path": self.data_path,
            "model_type": type(self.model).__name__,
            "n_estimators": getattr(self.model, "n_estimators", None),
            "max_depth": getattr(self.model, "max_depth", None),
            "learning_rate": getattr(self.model, "learning_rate", None)
        }


def test():
    """
    演示CourierDispatchPredictor的使用

    该函数展示了如何：
    1. 初始化快递员调度预测器
    2. 进行单个样本预测
    3. 进行批量样本预测
    4. 获取模型信息
    """
    logger.info("=== 快递员调度预测模型测试 ===")

    try:
        # 1. 初始化快递员调度预测器
        logger.info("\n1. 初始化快递员调度预测器...")
        predictor = CourierDispatchPredictor()

        # 2. 获取模型信息
        logger.info("\n2. 模型信息:")
        model_info = predictor.get_model_info()
        for key, value in model_info.items():
            logger.info(f"   {key}: {value}")

        # 3. 单个样本预测（列表格式）
        logger.info("\n3. 单个样本预测（列表格式）:")
        sample_list = [3, 15.5, 30, 14]  # 3个任务，15.5km距离，30分钟空闲，下午2点
        prediction = predictor.predict(sample_list)
        probability = predictor.predict_proba(sample_list)
        logger.info(f"   输入特征: {sample_list}")
        logger.info(f"   预测结果: {prediction} ({'按时完成' if prediction == 1 else '未按时完成'})")
        logger.info(f"   概率分布: 未按时={probability[0]:.3f}, 按时={probability[1]:.3f}")

        # 4. 单个样本预测（字典格式）
        logger.info("\n4. 单个样本预测（字典格式）:")
        sample_dict = {
            "task_count": 2,
            "distance": 8.3,
            "idle_time": 45,
            "hour": 10
        }
        prediction = predictor.predict(sample_dict)
        probability = predictor.predict_proba(sample_dict)
        logger.info(f"   输入特征: {sample_dict}")
        logger.info(f"   预测结果: {prediction} ({'按时完成' if prediction == 1 else '未按时完成'})")
        logger.info(f"   概率分布: 未按时={probability[0]:.3f}, 按时={probability[1]:.3f}")

        # 5. 批量样本预测（DataFrame格式）
        logger.info("\n5. 批量样本预测（DataFrame格式）:")
        samples_df = pd.DataFrame({
            "task_count": [1, 4, 2, 5],
            "distance": [5.2, 20.1, 12.3, 25.8],
            "idle_time": [60, 15, 30, 10],
            "hour": [9, 16, 11, 18]
        })
        predictions = predictor.predict(samples_df)
        probabilities = predictor.predict_proba(samples_df)

        logger.info("   批量预测结果:")
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            status = "按时完成" if pred == 1 else "未按时完成"
            logger.info(f"     样本{i+1}: {status} (概率: 按时={prob[1]:.3f})")

        logger.info("\n=== 测试完成 ===")

    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")


# 设置全局的预测对象
courier_dispatch_predictor = CourierDispatchPredictor()

if __name__ == "__main__":
    test()