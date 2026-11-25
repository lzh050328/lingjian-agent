"""
图节点路由逻辑。

定义各个节点间的条件跳转逻辑。
"""

from ykzh_agent.service_scope.graph.domain.models import GraphState
from ykzh_agent.service_scope.graph.domain.enums import DecisionStatus


def route_after_address_analysis(state: GraphState) -> str:
    """
    地址分析节点后的路由逻辑。

    根据是否成功获取坐标决定下一步：
    - 有坐标：继续到候选网点获取
    - 无坐标：直接结束流程

    Args:
        state: 当前图状态

    Returns:
        下一个节点名称或"end"
    """
    if state.location and state.location.geocode_success:
        return "candidate_station_fetcher"
    else:
        return "end"


def route_after_recommender(state: GraphState) -> str:
    """
    推荐节点后的路由逻辑。

    根据推荐结果决定是否生成理由：
    - 推荐成功：生成推荐理由
    - 推荐失败：直接结束

    Args:
        state: 当前图状态

    Returns:
        下一个节点名称或"end"
    """
    if state.decision and state.decision.decision_status == DecisionStatus.SUCCESS:
        return "reason_generator"
    else:
        return "end"