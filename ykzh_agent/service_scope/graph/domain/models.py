"""
服务领域图数据模型定义

该模块定义了服务推荐系统图执行过程中使用的所有核心数据模型。
这些模型用于在各个节点之间传递数据，并确保类型安全。
"""
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field

# 导入枚举类型，定义各种状态和置信度级别
from .enums import (
    AddressConfidence,    # 地址解析置信度枚举
    DecisionStatus,       # 决策状态枚举
    GeocodeConfidence,    # 地理编码置信度枚举
    LoadStatus,          # 网点负载状态枚举
    SlaStatus,           # SLA服务等级枚举
    StationStatus,       # 网点状态枚举
)

class AddressAnalysis(BaseModel):
    """
    地址分析结果模型。

    该对象由"地址分析 Agent 节点"生成，包含对原始地址的解析和理解结果。
    用于后续的地理编码和网点推荐流程。

    属性:
        normalized_address: 标准化后的地址表达，去除冗余信息，统一格式
        address_type: 地址类型分类，如 precise_address(精确地址)/poi(兴趣点)/
                     fuzzy_poi(模糊兴趣点)/road(道路地址)
        address_confidence: 地址解析的置信度等级，反映地址理解的可靠性
        risk_tags: 风险标签列表，标识地址存在的潜在问题，如 FUZZY_ADDRESS(模糊地址)
        rewrite_queries: 重写查询词列表，当主要地址解析失败时，提供备选查询方案
    """
    normalized_address: str = ""  # 标准化地址，如"北京市朝阳区望京街道"
    address_type: str = ""  # 地址类型分类
    address_confidence: AddressConfidence = AddressConfidence.MEDIUM  # 默认中等置信度
    risk_tags: List[str] = Field(default_factory=list)  # 风险标签，如["FUZZY_ADDRESS"]
    rewrite_queries: List[str] = Field(default_factory=list)  # 备选查询词，如["望京", "望京SOHO"]
    description: str = "" # 得出地址分析结果的过程描述


class Location(BaseModel):
    """
    地理定位结果模型。

    该对象记录通过地理编码服务获得的实际坐标信息，
    与地址文本分析分离，专注于空间定位数据。

    属性:
        lng: 经度坐标，使用高德坐标系(GCJ-02)
        lat: 纬度坐标，使用高德坐标系(GCJ-02)
        geocode_source: 定位服务来源，如"amap_mcp"(高德地图)、"baidu_api"等
        geocode_confidence: 定位结果的置信度，反映坐标的准确性
        geocode_success: 定位是否成功的标志位
    """
    lng: Optional[float] = None  # 经度，如 116.481488
    lat: Optional[float] = None  # 纬度，如 39.991937
    geocode_source: Optional[str] = None  # 定位服务来源
    geocode_confidence: Optional[GeocodeConfidence] = None  # 定位置信度
    geocode_success: bool = False  # 定位成功标志


class CandidateStation(BaseModel):
    """
    候选网点基础信息模型。

    该对象表示通过空间查询找到的潜在服务网点，
    由"服务范围查询"和"邻近网点查询"两个阶段的结果合并生成。
    此阶段只包含最基本的网点标识和位置关系信息。

    属性:
        station_id: 网点唯一标识符
        in_polygon: 是否在服务多边形区域内，True表示在服务范围内
        is_adjacent: 是否为邻近网点，True表示虽不在服务范围内但地理位置接近
    """
    station_id: str  # 网点ID，如"ST001"
    # station_name: str  # 网点名称(暂时注释，后续可能在其他阶段添加)
    in_polygon: bool = False  # 是否在服务区域内
    is_adjacent: bool = False  # 是否为邻近网点


class StationFact(BaseModel):
    """
    网点详细事实数据模型。

    这是推荐系统中最重要的数据对象，包含网点的完整状态信息和运营数据。
    由"网点事实丰富节点"通过多个工具查询获得，为后续的推荐规则引擎提供决策依据。
    所有字段都应基于实际查询结果，避免推测性数据。

    属性:
        station_id: 网点唯一标识符
        station_name: 网点完整名称
        in_polygon: 是否在服务多边形区域内
        is_adjacent: 是否为邻近网点
        station_status: 网点运营状态，如正常/维护/关闭等
        load_status: 网点当前负载状态，如正常/繁忙/超负荷等
        route_time_min: 到达该网点的预估时间(分钟)，999表示无法到达或超时
        route_distance_km: 到达该网点的预估距离(公里)，999.0表示无法到达
        sla_status: 网点SLA服务等级，反映服务质量和时效承诺
    """
    station_id: str  # 网点ID
    station_name: str  # 网点名称，如"望京西园网点"
    in_polygon: bool = False  # 是否在服务区域内
    is_adjacent: bool = False  # 是否为邻近网点
    station_status: StationStatus = StationStatus.NORMAL  # 网点运营状态
    load_status: LoadStatus = LoadStatus.NORMAL  # 负载状态
    route_time_min: int = 999  # 预估到达时间，默认999表示未知/不可达
    route_distance_km: float = 999.0  # 预估距离，默认999.0表示未知/不可达
    sla_status: SlaStatus = SlaStatus.MEDIUM  # SLA服务等级


class RankedStation(BaseModel):
    """
    排序后的网点结果模型。

    该对象表示经过评分算法计算后的网点排序结果，
    由"网点评分计算器"生成，用于最终的推荐决策。
    评分综合考虑了距离、时间、负载、SLA等多个维度。

    属性:
        station_id: 网点唯一标识符
        station_name: 网点名称
        score: 综合评分(0-100)，分数越高表示推荐优先级越高
    """
    station_id: str  # 网点ID
    station_name: str  # 网点名称
    score: int  # 综合评分，如85分


class Decision(BaseModel):
    """
    推荐决策结果模型。

    该对象由"规则推荐节点"生成，包含完整的推荐决策信息，
    是推荐系统的核心输出。决策基于评分结果和业务规则生成。

    属性:
        recommended_station_id: 最终推荐的网点ID，None表示无合适推荐
        recommended_station_name: 最终推荐的网点名称
        candidate_ranked: 所有候选网点的排序列表，按推荐优先级降序排列
        reason_codes: 决策理由代码列表，用于生成用户友好的解释
        decision_confidence: 决策置信度("high"/"medium"/"low")，反映推荐的可靠性
        manual_review: 是否建议人工复核，True表示系统推荐不够自信需要人工干预
        decision_status: 决策状态，SUCCESS(成功)或FAILED(失败)
    """
    recommended_station_id: Optional[str] = None  # 推荐网点ID
    recommended_station_name: Optional[str] = None  # 推荐网点名称
    candidate_ranked: List[RankedStation] = Field(default_factory=list)  # 排序后的候选网点
    reason_codes: List[str] = Field(default_factory=list)  # 理由代码，如["NEAREST", "LOW_LOAD"]
    decision_confidence: str = "low"  # 决策置信度
    manual_review: bool = False  # 是否需要人工复核
    decision_status: DecisionStatus = DecisionStatus.FAILED  # 决策状态，默认失败


class FinalResult(BaseModel):
    """
    最终人类可读结果模型。

    该对象由"理由生成节点"生成，将结构化的决策结果转换为
    人类可理解的自然语言描述，是系统对外的最终输出接口。

    属性:
        recommendation_text: 完整的推荐说明文案，面向运营人员或客户的详细解释
        risk_summary: 风险摘要说明，概述潜在的风险和限制条件
        reasons: 可拆分展示的简要原因列表，用于前端分条展示推荐理由
    """
    recommendation_text: str = ""  # 完整推荐文案，如"推荐望京网点，距离较近且负载正常"
    risk_summary: str = ""  # 风险说明，如"目标地址较模糊，建议人工确认"
    reasons: List[str] = Field(default_factory=list)  # 简要原因，如["距离最近", "负载较低"]


class GraphState(BaseModel):
    """
    LangGraph 全局状态对象。

    这是整个推荐图执行过程中的核心状态容器，封装了从输入到输出的所有数据。
    各个节点通过更新此状态的特定字段来传递数据，LangGraph框架负责状态的合并和管理。
    节点只返回"局部更新字典"，LangGraph会将这些更新merge回全局state。

    属性:
        request_id: 请求唯一标识符，用于日志追踪和监控
        raw_address: 用户输入的原始地址文本
        address_analysis: 地址分析结果，包含标准化地址和置信度信息
        location: 地理定位结果，包含经纬度坐标
        candidate_stations: 候选网点列表，初步筛选的潜在服务网点
        station_facts: 网点详细事实列表，包含运营状态和路由信息
        decision: 推荐决策结果，包含最终推荐和排序信息
        final_result: 最终人类可读结果，包含解释文案
        errors: 错误信息列表，记录执行过程中的异常情况
    """
    request_id: str  # 请求ID，用于追踪和日志
    raw_address: str  # 用户原始输入地址

    # 地址解析阶段
    address_analysis: AddressAnalysis = Field(default_factory=AddressAnalysis)  # 地址分析结果
    location: Location = Field(default_factory=Location)  # 定位结果

    # 网点查询阶段
    candidate_stations: List[CandidateStation] = Field(default_factory=list)  # 候选网点
    station_facts: List[StationFact] = Field(default_factory=list)  # 网点详细信息

    # 决策生成阶段
    decision: Decision = Field(default_factory=Decision)  # 推荐决策
    final_result: FinalResult = Field(default_factory=FinalResult)  # 最终结果

    # 错误处理
    errors: List[str] = Field(default_factory=list)  # 错误信息列表


class AddressAnalysisAgentResponse(BaseModel):
    """
    地址分析 Agent 结构化响应模型。

    这是地址分析节点的标准输出格式，同时承载两个层面的信息：
    1. 地址文本理解结果：标准化、分类、置信度评估
    2. 地理定位结果：通过MCP工具获得的实际坐标

    重要约束：
    - 如果 Agent 无法获得可靠坐标，必须设置 geocode_success=False，且 lng/lat=None
    - 该模型确保地址解析和地理编码结果的原子性，避免状态不一致

    属性:
        normalized_address: 标准化地址表达，去除冗余信息，统一格式
        address_type: 地址类型分类，用于后续处理策略选择
        address_confidence: 地址解析置信度，反映文本理解的可靠性
        risk_tags: 风险标签列表，标识地址质量问题
        rewrite_queries: 重写查询词，提供地理编码的备选方案
        lng: 通过地图API获得的经度坐标
        lat: 通过地图API获得的纬度坐标
        geocode_source: 定位服务来源标识
        geocode_confidence: 地理编码置信度，反映坐标的准确性
        geocode_success: 定位成功标志，False时必须清空坐标字段
    """
    normalized_address: str = Field(description="标准化后的地址表达")
    address_type: str = Field(description="地址类型，如 precise_address / poi / fuzzy_poi / road")
    address_confidence: AddressConfidence = Field(description="地址置信度")
    risk_tags: List[str] = Field(default_factory=list, description="地址风险标签")
    rewrite_queries: List[str] = Field(default_factory=list, description="后续 geocode 可尝试的候选查询词")

    lng: Optional[float] = Field(default=None, description="通过地图工具获得的经度")
    lat: Optional[float] = Field(default=None, description="通过地图工具获得的纬度")
    geocode_source: Optional[str] = Field(default=None, description="定位来源，例如 amap_mcp")
    geocode_confidence: Optional[GeocodeConfidence] = Field(
        default=None,
        description="定位置信度"
    )
    geocode_success: bool = Field(description="是否已成功获取可靠坐标")
    description:str = Field(description="描述")


class ReasonGenerationResponse(BaseModel):
    """
    理由生成节点响应模型。

    这是理由生成节点的标准输出格式，负责将结构化的决策结果
    转换为人类可读的自然语言描述。该模型确保输出的完整性和可读性。

    属性设计原则：
    1. recommendation_text: 完整的长文案，提供详细的推荐理由说明
    2. risk_summary: 风险摘要，突出显示潜在问题和限制条件
    3. reasons: 可拆分的原因列表，便于前端组件化展示

    属性:
        recommendation_text: 完整的推荐说明文案，包含所有关键信息和推理过程
        risk_summary: 风险摘要说明，概述决策中的不确定性和潜在问题
        reasons: 结构化的简要原因列表，每个元素都是一个独立的推荐理由
    """
    recommendation_text: str  # 完整推荐文案
    risk_summary: str  # 风险说明
    reasons: List[str]  # 可拆分的简要原因列表
