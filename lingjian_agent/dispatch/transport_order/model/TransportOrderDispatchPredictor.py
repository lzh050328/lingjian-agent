"""
运输订单调度预测器

该类负责加载、管理和使用XGBoost模型进行运输订单优先级预测。
主要功能包括：
1. 自动加载已训练的模型或触发训练流程
2. 提供模型预测接口
3. 支持批量和单个样本预测
"""

import os
import time
import joblib
import pandas as pd
from typing import List, Union, Dict, Any
from xgboost import XGBClassifier
from lingjian_config import logger

class TransportOrderDispatchPredictor:
    """运输订单调度预测器

    该类负责加载、管理和使用XGBoost模型进行运输订单优先级预测。
    主要功能包括：
    1. 自动加载已训练的模型或触发训练流程
    2. 提供模型预测接口
    3. 支持批量和单个样本预测
    """

    def __init__(self, model_path: str = "transport_order_model.pkl", data_path: str = "../data/train_data.csv"):
        """
        初始化运输订单调度预测器

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

        # 特征列定义
        self.feature_columns = [
            "total_weight",        # 订单总重量(kg)
            "total_volume",        # 订单总体积(m³)
            "wait_time",          # 等待时间(秒)
            "route_order_count"   # 当前线路订单数量
        ]

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
            # 切换到当前目录以正确执行训练脚本
            original_cwd = os.getcwd()
            os.chdir(os.path.dirname(os.path.abspath(__file__)))

            # 直接执行训练脚本
            # 注意：这里会执行train.py中的所有代码，包括模型训练和保存
            exec(open("train.py").read())

            # 训练完成后重新加载模型
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                logger.info("新模型训练并加载成功")
            else:
                raise FileNotFoundError("训练完成后未找到模型文件")

            # 恢复原始工作目录
            os.chdir(original_cwd)

        except Exception as e:
            logger.error(f"模型训练失败: {e}")
            raise RuntimeError(f"无法训练模型: {e}")

    def predict(self, features: Union[List[float], Dict[str, float], pd.DataFrame]) -> Union[int, List[int]]:
        """
        预测运输订单的优先级

        Args:
            features: 输入特征，支持以下格式：
                - List[float]: [total_weight, total_volume, wait_time, route_order_count]
                - Dict[str, float]: {'total_weight': x, 'total_volume': x, 'created': timestamp, 'route_order_count': x}
                - pd.DataFrame: 包含特征列的数据框

        Returns:
            预测结果：
                - 单个样本返回 int (0=普通优先级, 1=高优先级)
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
            X = pd.DataFrame([features], columns=self.feature_columns)
            prediction = self.model.predict(X)[0]
            return int(prediction)

        elif isinstance(features, dict):
            # 单个样本的字典格式
            required_keys = ["total_weight", "total_volume", "route_order_count"]

            # 检查必需的特征键
            if not all(key in features for key in required_keys):
                raise ValueError(f"字典必须包含所有必需的特征: {required_keys}")

            # 处理时间戳，计算等待时间
            if "created" in features:
                wait_time = int(time.time()) - features["created"]
            else:
                # 如果没有提供created，假设wait_time为0
                wait_time = 0

            feature_values = [
                features["total_weight"],
                features["total_volume"],
                wait_time,
                features["route_order_count"]
            ]

            X = pd.DataFrame([feature_values], columns=self.feature_columns)
            prediction = self.model.predict(X)[0]
            return int(prediction)

        elif isinstance(features, pd.DataFrame):
            # DataFrame格式（支持批量预测）
            required_columns = self.feature_columns.copy()

            # 如果DataFrame中包含created列，需要计算wait_time
            if "created" in features.columns:
                now = int(time.time())
                features = features.copy()  # 避免修改原始DataFrame
                features["wait_time"] = now - features["created"]

            if not all(col in features.columns for col in required_columns):
                raise ValueError(f"DataFrame必须包含所有必需的特征列: {required_columns}")

            X = features[required_columns]
            predictions = self.model.predict(X)
            return [int(pred) for pred in predictions]

        else:
            raise ValueError(
                "不支持的特征格式。请使用: "
                "1) List[float]: [total_weight, total_volume, wait_time, route_order_count] "
                "2) Dict[str, float]: {'total_weight': x, 'total_volume': x, 'created': timestamp, 'route_order_count': x} "
                "3) pd.DataFrame: 包含必需特征列的数据框"
            )

    def predict_proba(self, features: Union[List[float], Dict[str, float], pd.DataFrame]) -> Union[List[float], List[List[float]]]:
        """
        预测运输订单的优先级概率

        Args:
            features: 输入特征（格式同predict方法）

        Returns:
            概率预测结果：
                - 单个样本返回 [prob_normal, prob_high]
                - 多个样本返回 [[prob_normal, prob_high], ...]
        """
        if self.model is None:
            raise RuntimeError("模型未加载，请先调用_load_or_train_model()")

        # 处理不同的输入格式（与predict方法类似）
        if isinstance(features, list) and len(features) == 4:
            X = pd.DataFrame([features], columns=self.feature_columns)
            proba = self.model.predict_proba(X)[0]
            return proba.tolist()

        elif isinstance(features, dict):
            required_keys = ["total_weight", "total_volume", "route_order_count"]

            # 检查必需的特征键
            if not all(key in features for key in required_keys):
                raise ValueError(f"字典必须包含所有必需的特征: {required_keys}")

            # 处理时间戳，计算等待时间
            if "created" in features:
                wait_time = int(time.time()) - features["created"]
            else:
                # 如果没有提供created，假设wait_time为0
                wait_time = 0

            feature_values = [
                features["total_weight"],
                features["total_volume"],
                wait_time,
                features["route_order_count"]
            ]

            X = pd.DataFrame([feature_values], columns=self.feature_columns)
            proba = self.model.predict_proba(X)[0]
            return proba.tolist()

        elif isinstance(features, pd.DataFrame):
            required_columns = self.feature_columns.copy()

            # 如果DataFrame中包含created列，需要计算wait_time
            if "created" in features.columns:
                now = int(time.time())
                features = features.copy()  # 避免修改原始DataFrame
                features["wait_time"] = now - features["created"]

            if not all(col in features.columns for col in required_columns):
                raise ValueError(f"DataFrame必须包含所有必需的特征列: {required_columns}")

            X = features[required_columns]
            probabilities = self.model.predict_proba(X)
            return probabilities.tolist()

        else:
            raise ValueError(
                "不支持的特征格式。请使用: "
                "1) List[float]: [total_weight, total_volume, wait_time, route_order_count] "
                "2) Dict[str, float]: {'total_weight': x, 'total_volume': x, 'created': timestamp, 'route_order_count': x} "
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
                "data_path": self.data_path,
                "feature_columns": self.feature_columns
            }

        return {
            "status": "已加载",
            "model_path": self.model_path,
            "data_path": self.data_path,
            "model_type": type(self.model).__name__,
            "feature_columns": self.feature_columns,
            "n_estimators": getattr(self.model, "n_estimators", None),
            "max_depth": getattr(self.model, "max_depth", None),
            "learning_rate": getattr(self.model, "learning_rate", None)
        }

    def score(self, data: Dict[str, Any]) -> Dict[str, float]:
        """
        FastAPI接口兼容的评分方法

        Args:
            data: 输入数据字典，格式如下：
                {
                    "total_weight": 10,
                    "total_volume": 2,
                    "created": 1710000000,
                    "route_order_count": 20
                }

        Returns:
            包含分数的字典：{"score": 0.85}
        """
        try:
            # 使用predict_proba获取高优先级的概率
            probability = self.predict_proba(data)
            # 返回高优先级（类别1）的概率作为分数
            score = probability[1]  # probability[0]是普通优先级，probability[1]是高优先级

            return {"score": float(score)}

        except Exception as e:
            logger.error(f"评分失败: {e}")
            raise


def test():
    """
    演示TransportOrderDispatchPredictor的使用

    该函数展示了如何：
    1. 初始化运输订单调度预测器
    2. 进行单个样本预测
    3. 进行批量样本预测
    4. 获取模型信息
    """
    logger.info("=== 运输订单调度预测模型测试 ===")

    try:
        # 1. 初始化运输订单调度预测器
        logger.info("\n1. 初始化运输订单调度预测器...")
        predictor = TransportOrderDispatchPredictor()

        # 2. 获取模型信息
        logger.info("\n2. 模型信息:")
        model_info = predictor.get_model_info()
        for key, value in model_info.items():
            logger.info(f"   {key}: {value}")

        # 3. 单个样本预测（列表格式）
        logger.info("\n3. 单个样本预测（列表格式）:")
        sample_list = [10.5, 2.3, 3600, 15]  # 10.5kg重量，2.3m³体积，1小时等待，15个线路订单
        prediction = predictor.predict(sample_list)
        probability = predictor.predict_proba(sample_list)
        logger.info(f"   输入特征: {sample_list}")
        logger.info(f"   预测结果: {prediction} ({'高优先级' if prediction == 1 else '普通优先级'})")
        logger.info(f"   概率分布: 普通={probability[0]:.3f}, 高优先级={probability[1]:.3f}")

        # 4. 单个样本预测（字典格式，包含时间戳）
        logger.info("\n4. 单个样本预测（字典格式，包含时间戳）:")
        current_time = int(time.time())
        sample_dict = {
            "total_weight": 5.2,
            "total_volume": 1.1,
            "created": current_time - 7200,  # 2小时前创建
            "route_order_count": 8
        }
        prediction = predictor.predict(sample_dict)
        probability = predictor.predict_proba(sample_dict)
        logger.info(f"   输入特征: {sample_dict}")
        logger.info(f"   预测结果: {prediction} ({'高优先级' if prediction == 1 else '普通优先级'})")
        logger.info(f"   概率分布: 普通={probability[0]:.3f}, 高优先级={probability[1]:.3f}")

        # 5. 使用score方法（FastAPI接口格式）
        logger.info("\n5. FastAPI接口格式评分:")
        score_result = predictor.score(sample_dict)
        logger.info(f"   评分结果: {score_result}")

        # 6. 批量样本预测（DataFrame格式）
        logger.info("\n6. 批量样本预测（DataFrame格式）:")
        samples_df = pd.DataFrame({
            "total_weight": [8.5, 15.2, 3.1, 20.8],
            "total_volume": [1.8, 3.5, 0.9, 4.2],
            "created": [current_time - 1800, current_time - 3600, current_time - 900, current_time - 5400],
            "route_order_count": [12, 25, 6, 30]
        })
        predictions = predictor.predict(samples_df)
        probabilities = predictor.predict_proba(samples_df)

        logger.info("   批量预测结果:")
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            priority = "高优先级" if pred == 1 else "普通优先级"
            logger.info(f"     样本{i+1}: {priority} (概率: 高优先级={prob[1]:.3f})")

        logger.info("\n=== 测试完成 ===")

    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")


# 设置全局的预测对象
transport_order_dispatch_predictor = TransportOrderDispatchPredictor()

if __name__ == "__main__":
    test()