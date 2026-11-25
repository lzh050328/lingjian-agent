from typing import Dict, Any, List

from lingjian_agent.service_scope.graph.domain.enums import DecisionStatus, LoadStatus, StationStatus
from lingjian_agent.service_scope.graph.domain.models import Decision, GraphState, RankedStation, StationFact
from lingjian_agent.service_scope.graph.domain.ReasonCodes import ReasonCodes
from lingjian_agent.service_scope.graph.domain.StationScoreCalculator import StationScoreCalculator
from lingjian_agent.service_scope.graph.nodes.BaseGraphNode import BaseGraphNode


class StationRecommenderNode(BaseGraphNode):
    """
    规则推荐节点。

    这个节点是整个流程里最关键的“确定性决策层”。

    职责：
    1. 过滤不可用网点
    2. 对候选网点打分
    3. 选出推荐网点
    4. 输出结构化 reason codes

    设计原则：
    - 不把最终推荐完全交给 LLM
    - 保证稳定、可审计、可解释
    """

    def __init__(self, score_calculator: StationScoreCalculator) -> None:
        """
        初始化推荐节点。

        参数:
        - score_calculator: 网点评分计算器实例
        """
        self.score_calculator = score_calculator

    def _build_reason_codes(self, top_station: StationFact, state: GraphState) -> List[str]:
        """
        根据最终推荐网点和整体上下文构造结构化理由码。
        """
        reason_codes: List[str] = []

        if top_station.in_polygon:
            reason_codes.append(ReasonCodes.IN_SERVICE_RANGE)

        if top_station.is_adjacent:
            reason_codes.append(ReasonCodes.ADJACENT_AREA)

        if top_station.route_time_min <= 20:
            reason_codes.append(ReasonCodes.SHORTER_ROUTE_TIME)

        if top_station.load_status in (LoadStatus.LOW, LoadStatus.NORMAL):
            reason_codes.append(ReasonCodes.BETTER_LOAD_STATUS)

        if state.address_analysis.address_confidence.value == "low":
            reason_codes.append(ReasonCodes.LOW_ADDRESS_CONFIDENCE)

        return reason_codes

    async def run(self, state: GraphState) -> Dict[str, Any]:
        """
        执行网点推荐逻辑。

        流程:
        1. 过滤不可用网点（停运状态）
        2. 对候选网点进行评分排序
        3. 选择最优网点并生成推荐结果
        4. 生成结构化理由码
        5. 判断是否需要人工复核

        参数:
        - state: 图状态对象，包含网点事实数据

        返回:
        - 包含决策结果的字典
        """
        if not state.station_facts:
            return {
                "decision": Decision(
                    decision_status=DecisionStatus.FAILED,
                    manual_review=True,
                    reason_codes=[ReasonCodes.NO_AVAILABLE_STATION],
                )
            }

        # =========================
        # 1. 过滤不可用网点
        # 停运网点直接剔除，不参与推荐
        # =========================
        available_facts: List[StationFact] = [
            fact for fact in state.station_facts
            if fact.station_status != StationStatus.STOPPED
        ]

        if not available_facts:
            return {
                "decision": Decision(
                    decision_status=DecisionStatus.FAILED,
                    manual_review=True,
                    reason_codes=[ReasonCodes.NO_AVAILABLE_STATION],
                )
            }

        # =========================
        # 2. 计算得分并排序
        # 使用评分计算器对每个网点进行打分，按分数降序排列
        # =========================
        scored_items: List[tuple[StationFact, int]] = []

        for fact in available_facts:
            score = self.score_calculator.calculate(fact)
            scored_items.append((fact, score))

        scored_items.sort(key=lambda x: x[1], reverse=True)

        top_fact, top_score = scored_items[0]

        # =========================
        # 3. 生成排序列表
        # 为前端展示和后续处理生成排序后的网点列表
        # =========================
        ranked_list = [
            RankedStation(
                station_id=fact.station_id,
                station_name=fact.station_name,
                score=score,
            )
            for fact, score in scored_items
        ]

        # =========================
        # 4. 生成 reason codes
        # 基于最优网点的特征生成结构化推荐理由
        # =========================
        reason_codes = self._build_reason_codes(top_fact, state)

        # =========================
        # 5. 生成决策置信度
        # 基于最优网点的得分确定推荐置信度等级
        # =========================
        if top_score >= 80:
            decision_confidence = "high"
        elif top_score >= 60:
            decision_confidence = "medium"
        else:
            decision_confidence = "low"

        # =========================
        # 6. 判断是否需要人工复核
        # 基于以下规则触发人工复核:
        # - 最优网点得分过低(<60分)
        # - 地址置信度低且推荐的是邻近网点(边界区域)
        # =========================
        manual_review = False
        if top_score < 60:
            manual_review = True

        if (
            state.address_analysis.address_confidence.value == "low"
            and top_fact.is_adjacent
        ):
            manual_review = True

        decision = Decision(
            recommended_station_id=top_fact.station_id,
            recommended_station_name=top_fact.station_name,
            candidate_ranked=ranked_list,
            reason_codes=reason_codes,
            decision_confidence=decision_confidence,
            manual_review=manual_review,
            decision_status=DecisionStatus.SUCCESS,
        )

        return {"decision": decision}