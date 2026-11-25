from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime


@dataclass
class CarriageDTO:
    """运费数据传输对象，用于运费计算响应"""
    # 关联城市：1-全国，2-京津冀，3-江浙沪，4-川渝，5-黑吉辽
    associatedCityList: Optional[List[str]] = None
    # 计费重量，单位：kg
    computeWeight: Optional[float] = None
    # 续重价格
    continuousWeight: Optional[float] = None
    # 创建时间
    created: Optional[datetime] = None
    # 运费
    expense: Optional[float] = None
    # 首重价格
    firstWeight: Optional[float] = None
    # 运费模板id
    id: Optional[int] = None
    # 轻抛系数
    lightThrowingCoefficient: Optional[int] = None
    # 模板类型：1-同城寄，2-省内寄，3-经济区互寄，4-跨省
    templateType: Optional[int] = None
    # 运送类型：1-普快，2-特快
    transportType: Optional[int] = None
    # 更新时间
    updated: Optional[datetime] = None