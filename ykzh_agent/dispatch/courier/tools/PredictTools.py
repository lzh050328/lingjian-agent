import json
from dataclasses import dataclass
from typing import List, Dict, Any

from langchain.tools import tool

from ykzh_agent.dispatch.courier.model import courier_dispatch_predictor
from ykzh_config import logger


@dataclass
class CourierPredictionResult:
    """快递员预测结果数据类"""
    courier_id: str
    prediction: int  # 0=未按时完成, 1=按时完成
    probability: float  # 按时完成的概率
    status: str  # 状态描述


class CourierDispatchPredictTool:
    """快递员调度预测工具类

    该类封装了快递员调度预测功能，提供批量预测接口，
    用于预测快递员是否能按时完成配送任务。
    """

    @staticmethod
    def _prepare_features_for_prediction(courier_data: Dict[str, Any]) -> Dict[str, float]:
        """
        准备预测所需的特征数据

        Args:
            courier_data: 快递员数据字典，包含task_count, distance, idle_time, hour字段

        Returns:
            标准化后的特征字典
        """
        required_fields = ["task_count", "distance", "idle_time", "hour"]

        # 验证必需字段
        for field in required_fields:
            if field not in courier_data:
                raise ValueError(f"缺少必需字段: {field}")

        return {
            "task_count": float(courier_data["task_count"]),
            "distance": float(courier_data["distance"]),
            "idle_time": float(courier_data["idle_time"]),
            "hour": float(courier_data["hour"])
        }

    @staticmethod
    def predict_single_courier(courier_id: str, courier_data: Dict[str, Any]) -> CourierPredictionResult:
        """
        预测单个快递员是否能按时完成任务

        Args:
            courier_id: 快递员ID
            courier_data: 快递员特征数据，包含task_count, distance, idle_time, hour字段

        Returns:
            CourierPredictionResult对象，包含预测结果
        """
        try:
            # 准备特征数据
            features = CourierDispatchPredictTool._prepare_features_for_prediction(courier_data)

            # 进行预测
            prediction = courier_dispatch_predictor.predict(features)
            probabilities = courier_dispatch_predictor.predict_proba(features)

            # 按时完成的概率是第二个元素（索引1）
            on_time_probability = probabilities[1]

            # 转换为状态描述
            status = "按时完成" if prediction == 1 else "未按时完成"

            return CourierPredictionResult(
                courier_id=courier_id,
                prediction=prediction,
                probability=on_time_probability,
                status=status
            )

        except Exception as e:
            logger.error(f"预测快递员 {courier_id} 时发生错误: {e}")
            raise

    @staticmethod
    def predict_batch_couriers(couriers_data: Dict[str, Dict[str, Any]]) -> List[CourierPredictionResult]:
        """
        批量预测多个快递员是否能按时完成任务

        Args:
            couriers_data: 字典格式，key为快递员ID，value为快递员特征数据

        Returns:
            CourierPredictionResult对象列表
        """
        results = []

        for courier_id, courier_data in couriers_data.items():
            try:
                result = CourierDispatchPredictTool.predict_single_courier(courier_id, courier_data)
                results.append(result)
                logger.debug(f"快递员 {courier_id} 预测完成: {result.status}")
            except Exception as e:
                logger.error(f"预测快递员 {courier_id} 失败: {e}")
                # 继续处理其他快递员
                continue

        return results


def format_prediction_results(results: List[CourierPredictionResult]) -> str:
    """
    格式化预测结果为JSON字符串

    Args:
        results: CourierPredictionResult对象列表

    Returns:
        JSON格式的预测结果字符串
    """
    formatted_results = []

    for result in results:
        formatted_results.append({
            "courier_id": result.courier_id,
            "prediction": result.prediction,
            "probability": round(result.probability, 4),  # 保留4位小数
            "status": result.status
        })

    return json.dumps(formatted_results, ensure_ascii=False, indent=2)


@tool
def predict_courier_dispatch_single(courier_id: str, task_count: float, distance: float, idle_time: float,
                                    hour: float) -> str:
    """
    预测单个快递员是否能按时完成配送任务

    Args:
        courier_id: 快递员ID (必填)
        task_count: 任务数量 (必填)
        distance: 配送距离（公里）(必填)
        idle_time: 空闲时间（分钟）(必填)
        hour: 当前小时（24小时制，0-23）(必填)

    Returns:
        JSON字符串，包含预测结果：
        - courier_id: 快递员ID
        - prediction: 预测结果 (0=未按时完成, 1=按时完成)
        - probability: 按时完成的概率 (0-1之间的小数)
        - status: 状态描述 ("按时完成" 或 "未按时完成")

    Example:
        predict_courier_dispatch_single("C001", 3, 15.5, 30, 14)
    """
    logger.info(f"开始预测快递员 {courier_id} 的配送任务完成情况")

    courier_data = {
        "task_count": task_count,
        "distance": distance,
        "idle_time": idle_time,
        "hour": hour
    }

    try:
        result = CourierDispatchPredictTool.predict_single_courier(courier_id, courier_data)
        formatted_result = format_prediction_results([result])

        logger.info(f"快递员 {courier_id} 预测完成: {result.status}")
        return formatted_result

    except Exception as e:
        error_msg = f"预测快递员 {courier_id} 时发生错误: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg}, ensure_ascii=False)


@tool
def predict_courier_dispatch_batch(couriers_json: str) -> str:
    """
    批量预测多个快递员是否能按时完成配送任务

    Args:
        couriers_json: JSON字符串，包含多个快递员的特征数据
                       格式: {"courier_id1": {"task_count": x, "distance": x, "idle_time": x, "hour": x}, ...}

    Returns:
        JSON字符串数组，包含所有快递员的预测结果：
        [
          {
            "courier_id": "快递员ID",
            "prediction": 预测结果 (0=未按时完成, 1=按时完成),
            "probability": 按时完成的概率,
            "status": "状态描述"
          },
          ...
        ]

    Example:
        predict_courier_dispatch_batch('{"C001": {"task_count": 3, "distance": 15.5, "idle_time": 30, "hour": 14}, "C002": {"task_count": 2, "distance": 8.3, "idle_time": 45, "hour": 10}}')
    """
    logger.info("开始批量预测快递员配送任务完成情况")

    try:
        # 解析JSON输入
        couriers_data = json.loads(couriers_json)

        if not isinstance(couriers_data, dict):
            raise ValueError("输入必须是JSON对象格式")

        # 进行批量预测
        results = CourierDispatchPredictTool.predict_batch_couriers(couriers_data)

        if not results:
            return json.dumps({"error": "没有有效的预测结果"}, ensure_ascii=False)

        # 格式化结果
        formatted_results = format_prediction_results(results)

        logger.info(f"批量预测完成，共处理 {len(results)} 个快递员")
        return formatted_results

    except json.JSONDecodeError as e:
        error_msg = f"JSON解析错误: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg}, ensure_ascii=False)

    except Exception as e:
        error_msg = f"批量预测时发生错误: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg}, ensure_ascii=False)
