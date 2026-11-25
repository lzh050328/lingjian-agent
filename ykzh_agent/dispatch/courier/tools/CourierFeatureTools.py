import json
import random
from typing import Dict, Any, List
from datetime import datetime

from langchain.tools import tool
from ykzh_config import logger, config_manager
from ykzh_util import HttpClientUtil
from ykzh_util.NacosServiceSelector import NacosServiceSelector


class CourierFeatureTools:
    """快递员特征信息查询工具类

    该类提供快递员信息查询功能，包括任务数、距离、空闲时间和当前小时等信息。
    目前为模拟实现，后续可对接真实的服务端API。
    """

    @staticmethod
    def _generate_mock_courier_data(courier_id: str, user_longitude: float, user_latitude: float) -> Dict[str, Any]:
        """
        通过微服务获取快递员数据

        Args:
            courier_id: 快递员ID
            user_longitude: 用户经度
            user_latitude: 用户纬度

        Returns:
            包含快递员信息的字典
        """
        try:
            # 通过微服务获取快递员状态信息
            url = f"/couriers/status/{courier_id}"
            full_url = NacosServiceSelector.get_url(config_manager.get("cc.services.courier"), url)

            if not full_url:
                logger.warning(f"无法获取微服务地址，使用模拟数据: courier_id={courier_id}")
                return CourierFeatureTools._generate_fallback_data(courier_id, user_longitude, user_latitude)

            # 发起HTTP GET请求获取快递员数据
            courier_data = HttpClientUtil.get(full_url, "")

            if not courier_data:
                logger.warning(f"微服务返回空数据，使用模拟数据: courier_id={courier_id}")
                return CourierFeatureTools._generate_fallback_data(courier_id, user_longitude, user_latitude)

            logger.debug(f"【CourierFeatureTools】微服务返回数据: courier_id={courier_id}, data={courier_data}")

            # 从微服务响应中提取所需字段
            task_count = courier_data.get("tasks", {}).get("totalToday", random.randint(1, 8))
            idle_time = courier_data.get("availability", {}).get("idleDurationMin", random.randint(0, 120))

            # 如果微服务提供了位置信息，则使用真实位置计算距离
            location = courier_data.get("location", {})
            courier_lng = location.get("lng")
            courier_lat = location.get("lat")

            if courier_lng is not None and courier_lat is not None:
                # 使用真实位置计算距离
                distance = ((courier_lng - user_longitude) ** 2 + (courier_lat - user_latitude) ** 2) ** 0.5 * 111
            else:
                # 模拟快递员位置（在用户位置附近随机生成）
                courier_longitude = user_longitude + random.uniform(-0.1, 0.1)
                courier_latitude = user_latitude + random.uniform(-0.1, 0.1)
                distance = ((courier_longitude - user_longitude) ** 2 + (courier_latitude - user_latitude) ** 2) ** 0.5 * 111

            current_hour = datetime.now().hour

            return {
                "courier_id": courier_id,
                "task_count": task_count,
                "distance": round(distance, 2),
                "idle_time": idle_time,
                "hour": current_hour
            }

        except Exception as e:
            logger.error(f"通过微服务获取快递员数据失败，使用模拟数据: courier_id={courier_id}, error={e}")
            return CourierFeatureTools._generate_fallback_data(courier_id, user_longitude, user_latitude)

    @staticmethod
    def _generate_fallback_data(courier_id: str, user_longitude: float, user_latitude: float) -> Dict[str, Any]:
        """
        生成模拟的快递员数据（降级方案）

        Args:
            courier_id: 快递员ID
            user_longitude: 用户经度
            user_latitude: 用户纬度

        Returns:
            包含快递员信息的字典
        """
        # 模拟快递员位置（在用户位置附近随机生成）
        courier_longitude = user_longitude + random.uniform(-0.1, 0.1)
        courier_latitude = user_latitude + random.uniform(-0.1, 0.1)

        # 计算距离（简化计算，实际应用中应使用更精确的算法）
        distance = ((courier_longitude - user_longitude) ** 2 + (courier_latitude - user_latitude) ** 2) ** 0.5 * 111  # 粗略转换为公里

        # 生成模拟数据
        current_hour = datetime.now().hour
        task_count = random.randint(1, 8)  # 任务数1-8个
        idle_time = random.randint(0, 120)  # 空闲时间0-120分钟

        return {
            "courier_id": courier_id,
            "task_count": task_count,
            "distance": round(distance, 2),
            "idle_time": idle_time,
            "hour": current_hour
        }

    @staticmethod
    def query_single_courier(courier_id: str, user_longitude: float, user_latitude: float) -> Dict[str, Any]:
        """
        查询单个快递员的信息

        Args:
            courier_id: 快递员ID
            user_longitude: 用户经度
            user_latitude: 用户纬度

        Returns:
            包含快递员信息的字典
        """
        try:
            logger.info(f"查询快递员 {courier_id} 的信息，用户位置: ({user_longitude}, {user_latitude})")

            # 生成模拟数据
            courier_data = CourierFeatureTools._generate_mock_courier_data(courier_id, user_longitude, user_latitude)

            logger.info(f"快递员 {courier_id} 信息查询完成")
            return courier_data

        except Exception as e:
            logger.error(f"查询快递员 {courier_id} 信息时发生错误: {e}")
            raise

    @staticmethod
    def query_batch_couriers(courier_ids: List[str], user_longitude: float, user_latitude: float) -> Dict[str, Dict[str, Any]]:
        """
        批量查询多个快递员的信息

        Args:
            courier_ids: 快递员ID列表
            user_longitude: 用户经度
            user_latitude: 用户纬度

        Returns:
            字典，key为快递员ID，value为快递员信息
        """
        results = {}

        for courier_id in courier_ids:
            try:
                courier_data = CourierFeatureTools.query_single_courier(courier_id, user_longitude, user_latitude)
                results[courier_id] = courier_data
            except Exception as e:
                logger.error(f"查询快递员 {courier_id} 信息失败: {e}")
                # 继续处理其他快递员
                continue

        return results


@tool
def query_courier_info_single(courier_id: str, user_longitude: float, user_latitude: float) -> str:
    """
    查询单个快递员的信息

    Args:
        courier_id: 快递员ID (必填)
        user_longitude: 用户经度 (必填)
        user_latitude: 用户纬度 (必填)

    Returns:
        JSON字符串，包含快递员信息：
        {
            "courier_id": "快递员ID",
            "task_count": 任务数量,
            "distance": 与用户位置的距离（公里）,
            "idle_time": 空闲时间（分钟）,
            "hour": 当前小时（24小时制）
        }

    Example:
        query_courier_info_single("C001", 121.612519, 31.034925)
    """
    logger.info(f"开始查询快递员 {courier_id} 的信息")

    try:
        result = CourierFeatureTools.query_single_courier(courier_id, user_longitude, user_latitude)
        formatted_result = json.dumps(result, ensure_ascii=False, indent=2)

        logger.info(f"快递员 {courier_id} 信息查询完成")
        return formatted_result

    except Exception as e:
        error_msg = f"查询快递员 {courier_id} 信息时发生错误: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg}, ensure_ascii=False)


@tool
def query_courier_info_batch(courier_ids: str, user_longitude: float, user_latitude: float) -> str:
    """
    批量查询多个快递员的信息

    Args:
        courier_ids: 快递员ID列表，用逗号分隔的字符串 (必填)
                     例如: "C001,C002,C003"
        user_longitude: 用户经度 (必填)
        user_latitude: 用户纬度 (必填)

    Returns:
        JSON字符串，包含所有快递员的查询结果：
        {
            "courier_id1": {
                "task_count": 任务数量,
                "distance": 与用户位置的距离（公里）,
                "idle_time": 空闲时间（分钟）,
                "hour": 当前小时（24小时制）
            },
            "courier_id2": {
                ...
            },
            ...
        }

    Example:
        query_courier_info_batch("C001,C002,C003", 121.612519, 31.034925)
    """
    logger.info(f"开始批量查询快递员信息，快递员IDs: {courier_ids}")

    try:
        # 解析快递员ID列表
        courier_id_list = [cid.strip() for cid in courier_ids.split(",") if cid.strip()]

        if not courier_id_list:
            raise ValueError("快递员ID列表不能为空")

        # 进行批量查询
        results = CourierFeatureTools.query_batch_couriers(courier_id_list, user_longitude, user_latitude)

        if not results:
            return json.dumps({"error": "没有有效的查询结果"}, ensure_ascii=False)

        # 格式化结果
        formatted_results = json.dumps(results, ensure_ascii=False, indent=2)

        logger.info(f"批量查询完成，共处理 {len(results)} 个快递员")
        return formatted_results

    except Exception as e:
        error_msg = f"批量查询快递员信息时发生错误: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg}, ensure_ascii=False)