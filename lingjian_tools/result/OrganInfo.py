from dataclasses import dataclass
from typing import Optional


@dataclass
class OrganInfo:
    """机构信息实体，对应组织数据定义"""
    # 机构ID
    id: Optional[str] = None
    # 机构名称
    name: Optional[str] = None
    # 机构类型
    type: Optional[int] = None
    # 联系电话
    phone: Optional[str] = None
    # 地址信息(JSON格式)
    address: Optional[str] = None
    # 纬度
    latitude: Optional[float] = None
    # 经度
    longitude: Optional[float] = None
    # 父级机构ID
    parentId: Optional[str] = None
    # 负责人姓名
    managerName: Optional[str] = None
    # 扩展信息
    extra: Optional[str] = None
    # 状态
    status: Optional[bool] = None