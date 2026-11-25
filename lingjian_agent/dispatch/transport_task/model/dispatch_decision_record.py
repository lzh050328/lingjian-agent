"""
调度决策记录数据模型

该模型对应MongoDB中的dispatch_decision_record集合
"""

from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from bson import ObjectId


@dataclass
class TruckInfo:
    """车辆信息"""
    maxWeight: float
    maxVolume: float


@dataclass
class Features:
    """调度特征数据"""
    usedWeightRatio: float
    usedVolumeRatio: float
    orderCount: int
    avgWeight: float
    stdWeight: float
    maxOrderWeightRatio: float
    skippedOrderCount: int
    remainingOrderCount: int
    remainingHighPriorityRatio: float
    rounds: int


@dataclass
class Result:
    """调度结果"""
    isFullLoaded: bool = False
    isDelayed: Optional[bool] = None
    manualAdjusted: Optional[bool] = None


@dataclass
class DispatchDecisionRecord:
    """
    调度决策记录数据模型

    该模型用于存储调度决策的完整信息，包括：
    - 车辆信息
    - 调度特征
    - 调度结果
    - 时间戳
    """

    dispatchDecisionId: str
    """调度决策唯一ID"""

    truck: TruckInfo
    """车辆信息"""

    features: Features
    """调度特征数据"""

    result: Result = field(default_factory=Result)
    """调度结果"""

    status: str = "PENDING"
    """状态：PENDING(待完成) 或 COMPLETED(已完成)"""

    createdAt: datetime = field(default_factory=datetime.utcnow)
    """创建时间"""

    updatedAt: datetime = field(default_factory=datetime.utcnow)
    """更新时间"""

    _id: Optional[ObjectId] = None
    """MongoDB文档ID"""

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式，用于MongoDB存储

        Returns:
            Dict[str, Any]: 字典格式的记录数据
        """
        data = {
            "dispatchDecisionId": self.dispatchDecisionId,
            "truck": {
                "maxWeight": self.truck.maxWeight,
                "maxVolume": self.truck.maxVolume
            },
            "features": {
                "usedWeightRatio": self.features.usedWeightRatio,
                "usedVolumeRatio": self.features.usedVolumeRatio,
                "orderCount": self.features.orderCount,
                "avgWeight": self.features.avgWeight,
                "stdWeight": self.features.stdWeight,
                "maxOrderWeightRatio": self.features.maxOrderWeightRatio,
                "skippedOrderCount": self.features.skippedOrderCount,
                "remainingOrderCount": self.features.remainingOrderCount,
                "remainingHighPriorityRatio": self.features.remainingHighPriorityRatio,
                "rounds": self.features.rounds
            },
            "result": {
                "isFullLoaded": self.result.isFullLoaded,
                "isDelayed": self.result.isDelayed,
                "manualAdjusted": self.result.manualAdjusted
            },
            "status": self.status,
            "createdAt": self.createdAt,
            "updatedAt": self.updatedAt
        }

        if self._id:
            data["_id"] = self._id

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DispatchDecisionRecord':
        """
        从字典数据创建DispatchDecisionRecord实例

        Args:
            data (Dict[str, Any]): 字典格式的记录数据

        Returns:
            DispatchDecisionRecord: 数据模型实例
        """
        record = cls(
            dispatchDecisionId=data["dispatchDecisionId"],
            truck=TruckInfo(
                maxWeight=data["truck"]["maxWeight"],
                maxVolume=data["truck"]["maxVolume"]
            ),
            features=Features(
                usedWeightRatio=data["features"]["usedWeightRatio"],
                usedVolumeRatio=data["features"]["usedVolumeRatio"],
                orderCount=data["features"]["orderCount"],
                avgWeight=data["features"]["avgWeight"],
                stdWeight=data["features"]["stdWeight"],
                maxOrderWeightRatio=data["features"]["maxOrderWeightRatio"],
                skippedOrderCount=data["features"]["skippedOrderCount"],
                remainingOrderCount=data["features"]["remainingOrderCount"],
                remainingHighPriorityRatio=data["features"]["remainingHighPriorityRatio"],
                rounds=data["features"]["rounds"]
            ),
            result=Result(
                isFullLoaded=data["result"]["isFullLoaded"],
                isDelayed=data["result"].get("isDelayed"),
                manualAdjusted=data["result"].get("manualAdjusted")
            ),
            status=data["status"],
            createdAt=data["createdAt"],
            updatedAt=data["updatedAt"]
        )

        if "_id" in data:
            record._id = data["_id"]

        return record

    @classmethod
    def from_create_message(cls, message_data: Dict[str, Any]) -> 'DispatchDecisionRecord':
        """
        从CREATE消息创建记录

        Args:
            message_data (Dict[str, Any]): CREATE消息数据

        Returns:
            DispatchDecisionRecord: 数据模型实例
        """
        # 将时间戳转换为datetime对象
        created_at = datetime.utcfromtimestamp(message_data["createdAt"] / 1000)
        updated_at = datetime.utcfromtimestamp(message_data["updatedAt"] / 1000)

        return cls(
            dispatchDecisionId=message_data["dispatchDecisionId"],
            truck=TruckInfo(
                maxWeight=message_data["truck"]["maxWeight"],
                maxVolume=message_data["truck"]["maxVolume"]
            ),
            features=Features(
                usedWeightRatio=message_data["features"]["usedWeightRatio"],
                usedVolumeRatio=message_data["features"]["usedVolumeRatio"],
                orderCount=message_data["features"]["orderCount"],
                avgWeight=message_data["features"]["avgWeight"],
                stdWeight=message_data["features"]["stdWeight"],
                maxOrderWeightRatio=message_data["features"]["maxOrderWeightRatio"],
                skippedOrderCount=message_data["features"]["skippedOrderCount"],
                remainingOrderCount=message_data["features"]["remainingOrderCount"],
                remainingHighPriorityRatio=message_data["features"]["remainingHighPriorityRatio"],
                rounds=message_data["features"]["rounds"]
            ),
            result=Result(
                isFullLoaded=message_data["result"]["isFullLoaded"]
            ),
            status=message_data["status"],
            createdAt=created_at,
            updatedAt=updated_at
        )

    def update_from_complete_message(self, message_data: Dict[str, Any]) -> None:
        """
        使用COMPLETE消息更新记录

        Args:
            message_data (Dict[str, Any]): COMPLETE消息数据
        """
        self.result.isDelayed = message_data["isDelayed"]
        self.result.manualAdjusted = message_data["manualAdjusted"]
        self.status = "COMPLETED"
        self.updatedAt = datetime.utcnow()