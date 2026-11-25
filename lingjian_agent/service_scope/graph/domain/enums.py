from enum import Enum


class AddressConfidence(str, Enum):
    """
    地址置信度枚举。
    """
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class GeocodeConfidence(str, Enum):
    """
    geocode 置信度枚举。
    """
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class StationStatus(str, Enum):
    """
    网点营业状态。
    """
    NORMAL = "NORMAL"
    LIMITED = "LIMITED"   # 半停运 / 限流 / 接单受限
    STOPPED = "STOPPED"   # 停运 / 不可接单


class LoadStatus(str, Enum):
    """
    网点负载等级。
    """
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    OVERLOAD = "OVERLOAD"


class SlaStatus(str, Enum):
    """
    近时段 SLA 表现等级。
    """
    GOOD = "GOOD" # 良好
    MEDIUM = "MEDIUM" # 中等
    POOR = "POOR" # 差


class DecisionStatus(str, Enum):
    """
    推荐决策状态。
    """
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"