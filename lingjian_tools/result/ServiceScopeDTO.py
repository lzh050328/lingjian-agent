from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime


@dataclass
class Coordinate:
    """坐标点数据传输对象"""
    latitude: Optional[float] = None
    longitude: Optional[float] = None


@dataclass
class ServiceScopeDTO:
    """服务范围数据传输对象，用于服务范围查询响应"""
    # 业务id，可以是机构或快递员
    bid: Optional[int] = None
    # 类型，1-机构，2-快递员
    type: Optional[int] = None
    # 多边形坐标点，至少3个坐标点，首尾坐标必须相同
    polygon: Optional[List[Coordinate]] = None
    # 创建时间
    created: Optional[datetime] = None
    # 更新时间
    updated: Optional[datetime] = None