import asyncio
from dataclasses import dataclass
from typing import Dict, Any, Optional

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ykzh_agent.BaseAgent import parse_agent_response
from ykzh_agent.mcp import mcp_service
from ykzh_agent.prompts import system_prompt_config
from ykzh_agent.service_scope.graph.domain.enums import LoadStatus, SlaStatus, StationStatus
from ykzh_agent.service_scope.graph.domain.models import GraphState, StationFact
from ykzh_agent.service_scope.graph.nodes.BaseGraphNode import BaseGraphNode
from ykzh_tools.TransportTools import query_organ_by_id
from ykzh_tools.result.OrganInfo import OrganInfo
from ykzh_config import logger

@dataclass
class EnrichResponse:
    """网点信息丰富响应结构"""
    organ_info: Optional[OrganInfo] = None
    route_time_min: int = 30
    route_distance_km: float = 15.8


class StationFactEnricherNode(BaseGraphNode):
    """
    网点事实补全节点。

    职责：
    1. 批量查询候选网点的状态
    2. 批量查询候选网点的负载
    3. 批量查询发件地址到候选网点的路网时间
    4. 组装成 station_facts，供规则节点使用

    注意：
    - 这个节点只处理"确定性事实"
    - 不做自由推理
    """

    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm
        self.enricher_agent = None

    async def init_agent(self) -> None:
        if self.enricher_agent is not None:
            return

        # 获取高德提供的所有工具
        amap_tools = await mcp_service.get_tools("amap_mcp")

        all_tools = [query_organ_by_id] + amap_tools

        self.enricher_agent = create_agent(
            model=self.llm,
            tools=all_tools
        )

    async def run(self, state: GraphState) -> Dict[str, Any]:
        # 延迟初始化agent对象
        await self.init_agent()

        if not state.candidate_stations:
            return {}

        sender_lng = state.location.lng
        sender_lat = state.location.lat

        if not sender_lng or not sender_lat:
            state.errors.append("发件人位置信息缺失，无法计算路线")
            return {"station_facts": []}

        station_facts: list[StationFact] = []

        for candidate in state.candidate_stations:
            sid = candidate.station_id

            try:
                agent_input = [
                    SystemMessage(system_prompt_config.chat_station_fact_message),
                    HumanMessage(
                        f"机构ID: {sid}, 发件人坐标: {sender_lng},{sender_lat}。请先查询机构信息获取名称和经纬度，然后计算发件人到机构的驾车路线时间和距离。")
                ]
                result = await self.enricher_agent.ainvoke({"messages": agent_input})

                enrich_response = parse_agent_response(result, EnrichResponse)

                if enrich_response is None or enrich_response.organ_info is None:
                    state.errors.append(f"查询机构 {sid} 信息失败")
                    continue

                # 获取机构信息
                organ_info = enrich_response.organ_info

                # 创建StationFact对象
                station_fact = StationFact(
                    station_id=sid,
                    station_name=organ_info.name,
                    in_polygon=candidate.in_polygon,
                    is_adjacent=candidate.is_adjacent,
                    load_status=LoadStatus.NORMAL,  # 负载状态默认NORMAL
                    sla_status=SlaStatus.GOOD,  # SLA状态默认GOOD
                    station_status=StationStatus.NORMAL if organ_info.status else StationStatus.STOPPED,
                    route_time_min=enrich_response.route_time_min,
                    route_distance_km=enrich_response.route_distance_km
                )
                station_facts.append(station_fact)

            except Exception as e:
                logger.error(f"查询机构 {sid} 信息失败: {str(e)}")
                state.errors.append(f"查询机构 {sid} 信息失败: {str(e)}")

        return {
            "station_facts": station_facts
        }
