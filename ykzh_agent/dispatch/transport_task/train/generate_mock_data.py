"""
模拟数据生成器

根据业务规则自动生成调度训练数据，并写入MongoDB的dispatch_decision_record_mock集合
"""

import random
from datetime import datetime, timedelta

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))

from ykzh_config import config_manager, logger
from ykzh_agent.dispatch.transport_task.model import DispatchDecisionRecord, TruckInfo, Features, Result
from ykzh_agent.dispatch.transport_task.repository import dispatch_decision_repository


class MockDataGenerator:
    """
    模拟数据生成器

    根据业务规则生成符合真实分布的调度训练数据
    """

    def __init__(self):
        """
        初始化生成器
        """
        self.mock_collection_name = config_manager.get("mongo.collection.dispatchDecisionRecordMock")
        if not self.mock_collection_name:
            raise ValueError("MongoDB Mock集合名配置缺失")

        logger.info(f"模拟数据将写入集合: {self.mock_collection_name}")

    def _calculate_delay_probability(self, features: Features) -> float:
        """
        根据业务规则计算延误概率（优化版本）

        业务规则：
        - 高装载率 + 高优先级剩余订单 → 高延误概率
        - 低装载率 + 无跳单 + 低优先级 → 低延误概率
        - 增加极端情况的概率差异，提高模型置信度

        Args:
            features (Features): 调度特征

        Returns:
            float: 延误概率 (0-1)
        """
        # 判断极端情况，直接返回高/低概率

        # 高延误风险情况：高装载率 + 高优先级剩余 + 多跳单
        if (features.usedWeightRatio > 0.85 and
            features.remainingHighPriorityRatio > 0.4 and
            features.skippedOrderCount > 3):
            return 0.9

        # 高延误风险情况：多轮次 + 多剩余订单
        if features.rounds > 4 and features.remainingOrderCount > 8:
            return 0.85

        # 低延误风险情况：低装载率 + 无跳单 + 少剩余
        if (features.usedWeightRatio < 0.4 and
            features.skippedOrderCount == 0 and
            features.remainingOrderCount <= 2):
            return 0.1

        # 低延误风险情况：首次配送 + 低优先级
        if features.rounds == 1 and features.remainingHighPriorityRatio < 0.1:
            return 0.15

        # 中等风险情况：使用加权计算
        base_prob = 0.4  # 降低基础延误概率，平衡数据分布

        # 装载率影响：装载率越高，延误概率越高（与之前逻辑相反，更符合实际）
        weight_factor = features.usedWeightRatio * 0.3

        # 跳单影响：跳单越多，延误概率越高
        skip_factor = min(features.skippedOrderCount / 8, 1.0) * 0.25

        # 剩余高优先级影响：剩余高优先级越多，延误概率越高
        priority_factor = features.remainingHighPriorityRatio * 0.25

        # 剩余订单数量影响：剩余越多，延误概率越高
        remaining_factor = min(features.remainingOrderCount / 12, 1.0) * 0.15

        # 轮次影响：轮次越多，延误概率越高
        rounds_factor = min(features.rounds / 6, 1.0) * 0.15

        # 综合计算延误概率
        delay_prob = (base_prob + weight_factor + skip_factor +
                     priority_factor + remaining_factor + rounds_factor)

        # 限制在合理范围内
        return max(0.1, min(0.9, delay_prob))

    def _generate_mock_features(self, truck: TruckInfo) -> Features:
        """
        生成模拟特征数据（优化版本，增加特征间一致性）

        确保特征间的业务逻辑一致性：
        - 重量装载率和体积装载率应该正相关
        - 订单数量和平均重量决定总重量
        - 最大订单重量占比 = 最大单票重量 / 车辆最大载重，与Java侧语义一致

        Args:
            truck: 车辆信息，用于计算maxOrderWeightRatio

        Returns:
            Features: 模拟特征对象
        """
        # 生成基础参数
        order_count = random.randint(5, 30)
        avg_weight = random.uniform(50, 200)

        # 计算总重量相关的装载率（保持一致性）
        base_weight_ratio = random.uniform(0.3, 0.95)
        base_volume_ratio = base_weight_ratio * random.uniform(0.7, 1.3)  # 体积率与重量率相关但有波动

        used_weight_ratio = min(1.0, base_weight_ratio)
        used_volume_ratio = min(0.95, max(0.1, base_volume_ratio))

        # 标准差与平均重量相关
        std_weight = avg_weight * random.uniform(0.1, 0.4)

        # 最大订单重量占比 = 最大单票重量 / 车辆最大载重（与Java侧语义一致）
        max_order_weight = avg_weight * random.uniform(1.5, 3.0)  # 最大订单是平均的1.5-3倍
        max_order_weight_ratio = min(0.4, max_order_weight / truck.maxWeight)

        # 其他特征
        skipped_order_count = random.randint(0, min(8, order_count // 3))
        remaining_order_count = random.randint(0, 15)
        remaining_high_priority_ratio = random.uniform(0, 0.6)
        rounds = random.randint(1, 5)

        # 增加一些极端情况的样本
        if random.random() < 0.1:  # 10%概率生成极端高延误风险样本
            used_weight_ratio = random.uniform(0.85, 0.98)
            used_volume_ratio = random.uniform(0.8, 0.95)
            remaining_high_priority_ratio = random.uniform(0.4, 0.7)
            skipped_order_count = random.randint(4, 8)
            rounds = random.randint(4, 6)
            remaining_order_count = random.randint(8, 15)
        elif random.random() < 0.1:  # 10%概率生成极端低延误风险样本
            used_weight_ratio = random.uniform(0.2, 0.4)
            used_volume_ratio = random.uniform(0.15, 0.35)
            remaining_high_priority_ratio = random.uniform(0, 0.1)
            skipped_order_count = 0
            rounds = 1
            remaining_order_count = random.randint(0, 2)

        return Features(
            usedWeightRatio=used_weight_ratio,
            usedVolumeRatio=used_volume_ratio,
            orderCount=order_count,
            avgWeight=avg_weight,
            stdWeight=std_weight,
            maxOrderWeightRatio=max_order_weight_ratio,
            skippedOrderCount=skipped_order_count,
            remainingOrderCount=remaining_order_count,
            remainingHighPriorityRatio=remaining_high_priority_ratio,
            rounds=rounds
        )

    def _generate_mock_result(self, features: Features) -> Result:
        """
        根据特征生成模拟结果

        Args:
            features (Features): 调度特征

        Returns:
            Result: 模拟结果对象
        """
        # 计算延误概率
        delay_prob = self._calculate_delay_probability(features)

        # 根据概率决定是否延误
        is_delayed = random.random() < delay_prob

        # 手动调整概率（一般较少）
        manual_adjusted = random.random() < 0.1

        # 装载完成概率与装载率相关
        is_full_loaded = features.usedWeightRatio > 0.7

        return Result(
            isFullLoaded=is_full_loaded,
            isDelayed=is_delayed,
            manualAdjusted=manual_adjusted
        )

    def _generate_single_record(self, index: int) -> DispatchDecisionRecord:
        """
        生成单条模拟记录

        Args:
            index (int): 记录索引

        Returns:
            DispatchDecisionRecord: 模拟记录
        """
        # 生成唯一ID
        dispatch_decision_id = f"mock_{index:08d}_{int(datetime.now().timestamp())}"

        # 生成车辆信息
        truck = TruckInfo(
            maxWeight=random.uniform(5000, 15000),
            maxVolume=random.uniform(5, 15)
        )

        # 生成特征
        features = self._generate_mock_features(truck)

        # 生成结果
        result = self._generate_mock_result(features)

        # 生成时间戳
        created_at = datetime.now() - timedelta(days=random.randint(0, 30))
        updated_at = created_at + timedelta(minutes=random.randint(10, 120))

        return DispatchDecisionRecord(
            dispatchDecisionId=dispatch_decision_id,
            truck=truck,
            features=features,
            result=result,
            status="COMPLETED",
            createdAt=created_at,
            updatedAt=updated_at
        )

    def generate_mock_data(self, count: int = 1000) -> int:
        """
        生成指定数量的模拟数据

        Args:
            count (int): 生成数量

        Returns:
            int: 成功生成的记录数量
        """
        logger.info(f"开始生成 {count} 条模拟数据...")

        success_count = 0
        batch_size = 100

        try:
            # 切换到mock集合
            original_collection = dispatch_decision_repository.collection
            mock_database = dispatch_decision_repository.client[
                config_manager.get("mongo.database.ccAi")
            ]
            mock_collection = mock_database[self.mock_collection_name]
            dispatch_decision_repository.collection = mock_collection

            # 创建索引
            mock_collection.create_index([("dispatchDecisionId", 1)], unique=True)

            for i in range(0, count, batch_size):
                batch_count = min(batch_size, count - i)
                batch_records = []

                # 生成一批记录
                for j in range(batch_count):
                    try:
                        record = self._generate_single_record(i + j)
                        batch_records.append(record.to_dict())
                    except Exception as e:
                        logger.error(f"生成单条记录失败 (index={i+j}): {e}")
                        continue

                # 批量插入
                if batch_records:
                    try:
                        result = mock_collection.insert_many(batch_records, ordered=False)
                        success_count += len(result.inserted_ids)
                        logger.info(f"已生成 {success_count}/{count} 条记录")
                    except Exception as e:
                        # 处理重复键等错误
                        for record_data in batch_records:
                            try:
                                mock_collection.insert_one(record_data)
                                success_count += 1
                            except Exception as single_error:
                                logger.warning(f"插入单条记录失败: {single_error}")

            logger.info(f"模拟数据生成完成，成功插入 {success_count} 条记录")

        except Exception as e:
            logger.error(f"生成模拟数据时发生错误: {e}")
        finally:
            # 恢复原始集合
            dispatch_decision_repository.collection = original_collection

        return success_count

    def clear_mock_data(self) -> int:
        """
        清除所有模拟数据

        Returns:
            int: 删除的记录数量
        """
        try:
            # 切换到mock集合
            mock_database = dispatch_decision_repository.client[
                config_manager.get("mongo.database.ccAi")
            ]
            mock_collection = mock_database[self.mock_collection_name]

            # 删除所有数据
            result = mock_collection.delete_many({})
            logger.info(f"已清除 {result.deleted_count} 条模拟数据")

            return result.deleted_count

        except Exception as e:
            logger.error(f"清除模拟数据时发生错误: {e}")
            return 0


def generate_mock_dataset(count: int = 1000):
    """
    生成模拟数据集的便捷函数

    Args:
        count (int): 生成数量
    """
    generator = MockDataGenerator()
    return generator.generate_mock_data(count)


def clear_mock_dataset():
    """
    清除模拟数据集的便捷函数
    """
    generator = MockDataGenerator()
    return generator.clear_mock_data()


if __name__ == "__main__":
    # 生成10000条模拟数据
    generator = MockDataGenerator()
    success_count = generator.generate_mock_data(10000)

    logger.info(f"模拟数据生成完成，共生成 {success_count} 条记录")