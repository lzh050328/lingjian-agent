from typing import Dict, Any

from ykzh_agent.service_scope.graph.domain.models import CandidateStation, GraphState
from ykzh_agent.service_scope.graph.nodes.BaseGraphNode import BaseGraphNode

from ykzh_tools import query_service_scope_list, query_adjacent_service_scope_list


class CandidateStationFetcherNode(BaseGraphNode):
    """
    候选网点召回节点。

    职责：
    1. 根据坐标查询命中服务范围的网点
    2. 在低置信度地址场景下，补充邻近网点
    3. 合并去重后生成 candidate_stations

    说明：
    - 这个节点只负责“召回候选”，不做推荐
    - 推荐动作由后面的规则节点完成
    """

    def __init__(self) -> None:
        pass

    async def run(self, state: GraphState) -> Dict[str, Any]:
        if not state.location.geocode_success or state.location.lng is None or state.location.lat is None:
            return {}

        lng = state.location.lng
        lat = state.location.lat

        # =========================
        # 1. 查询命中服务范围的网点
        # =========================
        matched_stations = query_service_scope_list(1, lng, lat)

        candidate_map: dict[str, CandidateStation] = {}

        for item in matched_stations:
            candidate = CandidateStation(
                station_id=item.bid,
                in_polygon=True,
                is_adjacent=False,
            )
            candidate_map[candidate.station_id] = candidate

        # =========================
        # 2. 如果地址低置信度，则补充邻近网点
        # 这样可以降低 geocode 漂移导致误分的风险
        # =========================
        if state.address_analysis.address_confidence.value == "low":
            nearby_stations = query_adjacent_service_scope_list(1, lng, lat)

            for item in nearby_stations:
                sid = item.bid
                if sid not in candidate_map:
                    candidate_map[sid] = CandidateStation(
                        station_id=sid,
                        in_polygon=False,
                        is_adjacent=True,
                    )

        return {
            "candidate_stations": list(candidate_map.values())
        }
