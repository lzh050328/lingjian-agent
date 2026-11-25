"""
调度决策记录数据访问层

负责与MongoDB的dispatch_decision_record集合进行交互
"""

from typing import Optional, List
from datetime import datetime

from pymongo import MongoClient, ASCENDING
from pymongo.errors import DuplicateKeyError
from bson import ObjectId

from ykzh_config import config_manager, logger
from ykzh_agent.dispatch.transport_task.model import DispatchDecisionRecord


class DispatchDecisionRepository:
    """
    调度决策记录数据访问对象

    提供MongoDB的增删改查操作，确保数据访问的幂等性
    """

    def __init__(self):
        """
        初始化MongoDB连接
        """
        # 从配置管理器获取MongoDB配置
        mongo_uri = config_manager.get("mongo.uri")
        database_name = config_manager.get("mongo.database.ccAi")
        collection_name = config_manager.get("mongo.collection.dispatchDecisionRecord")

        if not mongo_uri:
            raise ValueError("MongoDB URI 配置缺失")
        if not database_name:
            raise ValueError("MongoDB 数据库名配置缺失")
        if not collection_name:
            raise ValueError("MongoDB 集合名配置缺失")

        try:
            # 创建MongoDB客户端连接
            self.client = MongoClient(mongo_uri)
            self.database = self.client[database_name]
            self.collection = self.database[collection_name]

            # 创建唯一索引，确保dispatchDecisionId的唯一性
            self._create_indexes()

            logger.info(f"成功连接到MongoDB: {database_name}.{collection_name}")

        except Exception as e:
            logger.error(f"连接MongoDB失败: {e}")
            raise

    def _create_indexes(self):
        """
        创建必要的索引
        """
        try:
            # 创建dispatchDecisionId的唯一索引
            self.collection.create_index(
                [("dispatchDecisionId", ASCENDING)],
                unique=True,
                name="dispatch_decision_id_unique"
            )

            # 创建createdAt索引，用于查询优化
            self.collection.create_index(
                [("createdAt", ASCENDING)],
                name="created_at_index"
            )

            # 创建status索引，用于状态查询
            self.collection.create_index(
                [("status", ASCENDING)],
                name="status_index"
            )

            logger.info("MongoDB索引创建成功")

        except Exception as e:
            logger.warning(f"创建MongoDB索引时出现警告: {e}")

    def insert_record(self, record: DispatchDecisionRecord) -> bool:
        """
        插入新的调度决策记录（幂等操作）

        Args:
            record (DispatchDecisionRecord): 要插入的记录

        Returns:
            bool: 插入成功返回True，记录已存在返回False

        Raises:
            Exception: 其他数据库错误
        """
        try:
            # 转换为字典格式
            data = record.to_dict()

            # 插入记录
            result = self.collection.insert_one(data)

            if result.inserted_id:
                record._id = result.inserted_id
                logger.info(f"成功插入记录: {record.dispatchDecisionId}")
                return True
            else:
                logger.warning(f"插入记录失败: {record.dispatchDecisionId}")
                return False

        except DuplicateKeyError:
            # 记录已存在，幂等处理
            logger.info(f"记录已存在，跳过插入: {record.dispatchDecisionId}")
            return False

        except Exception as e:
            logger.error(f"插入记录时发生错误: {record.dispatchDecisionId}, 错误: {e}")
            raise

    def update_record_by_complete(self, dispatch_decision_id: str, is_delayed: bool, manual_adjusted: bool) -> bool:
        """
        根据COMPLETE消息更新记录

        Args:
            dispatch_decision_id (str): 调度决策ID
            is_delayed (bool): 是否延迟
            manual_adjusted (bool): 是否手动调整

        Returns:
            bool: 更新成功返回True，记录不存在返回False
        """
        try:
            # 构建更新数据
            update_data = {
                "$set": {
                    "result.isDelayed": is_delayed,
                    "result.manualAdjusted": manual_adjusted,
                    "status": "COMPLETED",
                    "updatedAt": datetime.utcnow()
                }
            }

            # 执行更新操作
            result = self.collection.update_one(
                {"dispatchDecisionId": dispatch_decision_id},
                update_data
            )

            if result.matched_count > 0:
                logger.info(f"成功更新记录: {dispatch_decision_id}")
                return True
            else:
                logger.warning(f"记录不存在，无法更新: {dispatch_decision_id}")
                return False

        except Exception as e:
            logger.error(f"更新记录时发生错误: {dispatch_decision_id}, 错误: {e}")
            raise

    def find_by_dispatch_decision_id(self, dispatch_decision_id: str) -> Optional[DispatchDecisionRecord]:
        """
        根据调度决策ID查找记录

        Args:
            dispatch_decision_id (str): 调度决策ID

        Returns:
            Optional[DispatchDecisionRecord]: 找到的记录，不存在返回None
        """
        try:
            data = self.collection.find_one({"dispatchDecisionId": dispatch_decision_id})

            if data:
                return DispatchDecisionRecord.from_dict(data)
            else:
                return None

        except Exception as e:
            logger.error(f"查找记录时发生错误: {dispatch_decision_id}, 错误: {e}")
            raise

    def find_by_status(self, status: str, limit: int = 1000) -> List[DispatchDecisionRecord]:
        """
        根据状态查找记录

        Args:
            status (str): 状态值
            limit (int): 返回记录数量限制

        Returns:
            List[DispatchDecisionRecord]: 记录列表
        """
        try:
            cursor = self.collection.find({"status": status}).limit(limit)
            records = []

            for data in cursor:
                records.append(DispatchDecisionRecord.from_dict(data))

            return records

        except Exception as e:
            logger.error(f"根据状态查找记录时发生错误: {status}, 错误: {e}")
            raise

    def find_completed_records(self, start_date: Optional[datetime] = None,
                             end_date: Optional[datetime] = None,
                             limit: int = 1000) -> List[DispatchDecisionRecord]:
        """
        查找已完成的记录，用于模型训练

        Args:
            start_date (Optional[datetime]): 开始时间
            end_date (Optional[datetime]): 结束时间
            limit (int): 返回记录数量限制

        Returns:
            List[DispatchDecisionRecord]: 已完成的记录列表
        """
        try:
            # 构建查询条件
            query = {"status": "COMPLETED"}

            # 添加时间范围过滤
            if start_date or end_date:
                time_query = {}
                if start_date:
                    time_query["$gte"] = start_date
                if end_date:
                    time_query["$lte"] = end_date

                if time_query:
                    query["createdAt"] = time_query

            cursor = self.collection.find(query).limit(limit)
            records = []

            for data in cursor:
                records.append(DispatchDecisionRecord.from_dict(data))

            logger.info(f"找到 {len(records)} 条已完成记录")
            return records

        except Exception as e:
            logger.error(f"查找已完成记录时发生错误: {e}")
            raise

    def get_statistics(self) -> dict:
        """
        获取集合统计信息

        Returns:
            dict: 统计信息
        """
        try:
            total_count = self.collection.count_documents({})
            pending_count = self.collection.count_documents({"status": "PENDING"})
            completed_count = self.collection.count_documents({"status": "COMPLETED"})

            return {
                "total": total_count,
                "pending": pending_count,
                "completed": completed_count
            }

        except Exception as e:
            logger.error(f"获取统计信息时发生错误: {e}")
            raise

    def close(self):
        """
        关闭MongoDB连接
        """
        try:
            if hasattr(self, 'client') and self.client:
                self.client.close()
                logger.info("MongoDB连接已关闭")
        except Exception as e:
            logger.error(f"关闭MongoDB连接时发生错误: {e}")


# 创建全局仓库实例
dispatch_decision_repository = DispatchDecisionRepository()