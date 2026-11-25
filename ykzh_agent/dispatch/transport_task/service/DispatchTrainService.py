"""
调度训练业务服务层

负责处理CREATE和COMPLETE消息的业务逻辑
"""

import json
from typing import Dict, Any, Optional

from ykzh_config import config_manager, logger
from ykzh_agent.dispatch.transport_task.repository import dispatch_decision_repository
from ykzh_agent.dispatch.transport_task.model import DispatchDecisionRecord


class DispatchTrainService:
    """
    调度训练业务服务

    负责处理来自RabbitMQ的消息，包括：
    - CREATE消息：创建新的调度决策记录
    - COMPLETE消息：更新已完成的调度决策记录
    """

    def __init__(self):
        """
        初始化服务

        Args:
            repository (Optional[DispatchDecisionRepository]): 数据访问层实例
        """
        self.repository = dispatch_decision_repository

    def handle_create_message(self, message_data: Dict[str, Any]) -> bool:
        """
        处理CREATE消息

        创建新的调度决策记录到MongoDB

        Args:
            message_data (Dict[str, Any]): CREATE消息数据

        Returns:
            bool: 处理成功返回True，失败返回False
        """
        try:
            # 验证消息格式
            if not self._validate_create_message(message_data):
                logger.error(f"CREATE消息格式验证失败: {message_data}")
                return False

            # 从消息创建记录
            record = DispatchDecisionRecord.from_create_message(message_data)

            # 插入到数据库（幂等操作）
            success = self.repository.insert_record(record)

            if success:
                logger.info(f"CREATE消息处理成功: {record.dispatchDecisionId}")
            else:
                logger.info(f"CREATE消息已存在，跳过处理: {record.dispatchDecisionId}")

            return True

        except Exception as e:
            logger.error(f"处理CREATE消息时发生错误: {e}")
            # 记录错误但不抛出异常，确保不影响核心调度流程
            return False

    def handle_complete_message(self, message_data: Dict[str, Any]) -> bool:
        """
        处理COMPLETE消息

        更新调度决策记录的结果信息

        Args:
            message_data (Dict[str, Any]): COMPLETE消息数据

        Returns:
            bool: 处理成功返回True，失败返回False
        """
        try:
            # 验证消息格式
            if not self._validate_complete_message(message_data):
                logger.error(f"COMPLETE消息格式验证失败: {message_data}")
                return False

            dispatch_decision_id = message_data["dispatchDecisionId"]
            is_delayed = message_data["isDelayed"]
            manual_adjusted = message_data["manualAdjusted"]

            # 更新记录
            success = self.repository.update_record_by_complete(
                dispatch_decision_id, is_delayed, manual_adjusted
            )

            if success:
                logger.info(f"COMPLETE消息处理成功: {dispatch_decision_id}")
            else:
                logger.warning(f"COMPLETE消息处理失败，记录不存在: {dispatch_decision_id}")

            return success

        except Exception as e:
            logger.error(f"处理COMPLETE消息时发生错误: {e}")
            # 记录错误但不抛出异常，确保不影响核心调度流程
            return False

    def _validate_create_message(self, message_data: Dict[str, Any]) -> bool:
        """
        验证CREATE消息格式

        Args:
            message_data (Dict[str, Any]): 消息数据

        Returns:
            bool: 验证通过返回True，否则返回False
        """
        try:
            # 检查必需字段
            required_fields = [
                "dispatchDecisionId", "truck", "features", "result",
                "status", "createdAt", "updatedAt"
            ]

            for field in required_fields:
                if field not in message_data:
                    logger.error(f"CREATE消息缺少必需字段: {field}")
                    return False

            # 检查truck字段
            truck = message_data["truck"]
            if "maxWeight" not in truck or "maxVolume" not in truck:
                logger.error("CREATE消息truck字段格式错误")
                return False

            # 检查features字段
            features = message_data["features"]
            required_feature_fields = [
                "usedWeightRatio", "usedVolumeRatio", "orderCount",
                "avgWeight", "stdWeight", "maxOrderWeightRatio",
                "skippedOrderCount", "remainingOrderCount",
                "remainingHighPriorityRatio", "rounds"
            ]

            for field in required_feature_fields:
                if field not in features:
                    logger.error(f"CREATE消息features缺少字段: {field}")
                    return False

            # 检查result字段
            result = message_data["result"]
            if "isFullLoaded" not in result:
                logger.error("CREATE消息result字段格式错误")
                return False

            return True

        except Exception as e:
            logger.error(f"验证CREATE消息时发生错误: {e}")
            return False

    def _validate_complete_message(self, message_data: Dict[str, Any]) -> bool:
        """
        验证COMPLETE消息格式

        Args:
            message_data (Dict[str, Any]): 消息数据

        Returns:
            bool: 验证通过返回True，否则返回False
        """
        try:
            # 检查必需字段
            required_fields = ["dispatchDecisionId", "isDelayed", "manualAdjusted"]

            for field in required_fields:
                if field not in message_data:
                    logger.error(f"COMPLETE消息缺少必需字段: {field}")
                    return False

            # 检查字段类型
            if not isinstance(message_data["isDelayed"], bool):
                logger.error("COMPLETE消息isDelayed字段类型错误")
                return False

            if not isinstance(message_data["manualAdjusted"], bool):
                logger.error("COMPLETE消息manualAdjusted字段类型错误")
                return False

            return True

        except Exception as e:
            logger.error(f"验证COMPLETE消息时发生错误: {e}")
            return False

    def get_statistics(self) -> Dict[str, int]:
        """
        获取调度决策统计信息

        Returns:
            Dict[str, int]: 统计信息
        """
        try:
            return self.repository.get_statistics()
        except Exception as e:
            logger.error(f"获取统计信息时发生错误: {e}")
            return {"total": 0, "pending": 0, "completed": 0}

    def get_pending_records(self, limit: int = 100) -> list:
        """
        获取待处理的记录

        Args:
            limit (int): 返回记录数量限制

        Returns:
            list: 待处理记录列表
        """
        try:
            records = self.repository.find_by_status("PENDING", limit)
            return [record.to_dict() for record in records]
        except Exception as e:
            logger.error(f"获取待处理记录时发生错误: {e}")
            return []

    def get_completed_records(self, start_date: Optional[str] = None,
                            end_date: Optional[str] = None,
                            limit: int = 1000) -> list:
        """
        获取已完成的记录，用于模型训练

        Args:
            start_date (Optional[str]): 开始时间 (ISO格式字符串)
            end_date (Optional[str]): 结束时间 (ISO格式字符串)
            limit (int): 返回记录数量限制

        Returns:
            list: 已完成记录列表
        """
        try:
            # 转换时间格式
            from datetime import datetime
            start_datetime = datetime.fromisoformat(start_date) if start_date else None
            end_datetime = datetime.fromisoformat(end_date) if end_date else None

            records = self.repository.find_completed_records(start_datetime, end_datetime, limit)
            return [record.to_dict() for record in records]
        except Exception as e:
            logger.error(f"获取已完成记录时发生错误: {e}")
            return []

    def process_message(self, routing_key: str, message_body: str) -> bool:
        """
        处理RabbitMQ消息的统一入口

        Args:
            routing_key (str): 路由键 (CREATE 或 COMPLETE)
            message_body (str): 消息体 (JSON字符串)

        Returns:
            bool: 处理成功返回True，失败返回False
        """
        try:
            # 解析JSON消息
            message_data = json.loads(message_body)

            # 根据路由键处理不同类型的消息
            if routing_key == config_manager.get("rabbitmq.aigc.routingKey.create"):
                return self.handle_create_message(message_data)
            elif routing_key == config_manager.get("rabbitmq.aigc.routingKey.complete"):
                return self.handle_complete_message(message_data)
            else:
                logger.warning(f"未知的路由键: {routing_key}")
                return False

        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}")
            return False
        except Exception as e:
            logger.error(f"处理消息时发生错误: {e}")
            return False


# 创建全局服务实例
dispatch_train_service = DispatchTrainService()