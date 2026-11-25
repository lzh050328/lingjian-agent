from dataclasses import dataclass
from typing import Optional


@dataclass
class AreaInfo:
    """行政区域信息实体，对应 AreaDto 定义"""
    # id
    id: Optional[int] = None
    # 父级行政
    parentId: Optional[int] = None
    # 行政名称
    name: Optional[str] = None
    # 行政区域编号
    areaCode: Optional[str] = None
    # 城市编码
    cityCode: Optional[str] = None
    # 合并名称
    mergerName: Optional[str] = None
    # 简称
    shortName: Optional[str] = None
    # 邮政编码
    zipCode: Optional[str] = None
    # 行政区域等级（0: 省级 1:市级 2:区）
    level: Optional[int] = None
    # 经度
    lng: Optional[str] = None
    # 纬度
    lat: Optional[str] = None
    # 拼音
    pinyin: Optional[str] = None
    # 首字母
    first: Optional[str] = None
