from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph

from lingjian_agent.BaseAgent import *
from lingjian_agent.prompts import system_prompt_config
from lingjian_agent.mcp import mcp_service
from lingjian_agent.service_scope.graph import RecommendationGraphBuilder
from lingjian_agent.service_scope.graph.domain.models import GraphState
from lingjian_config import logger

@dataclass
class ServiceScopeResponse:
    """服务范围分析响应结构"""
    request_id: str  # 请求ID
    recommended_station: Dict[str, Any]  # 推荐站点信息
    candidate_stations: List[Dict[str, Any]]  # 候选站点排名列表
    decision_confidence: float  # 决策置信度 (0.00-1.00)
    manual_review: bool  # 是否需要人工审核
    reason_codes: List[str]  # 决策原因代码列表
    recommendation_text: str  # 推荐文本说明
    risk_summary: str  # 风险摘要
    reasons: List[str]  # 详细原因列表
    errors: List[str]  # 错误信息列表


class ServiceScopeAgent(BaseAgent):

    def __init__(self, provider: str = "openai"):
        self.provider = provider
        self.llm: Optional[BaseChatModel] = None  # LLM 实例
        self.graph: Optional[CompiledStateGraph] = None  # LangGraph 图实例

    async def init(self):
        """
        初始化 ServiceScopeAgent：
        1. 创建 LLM 实例
        2. 构建推荐图
        """
        # 初始化 LLM（统一走 openai-compatible 接口）
        self.llm = super().init_llm(self.provider)

        # 构建推荐图
        graph_builder = RecommendationGraphBuilder(self.llm)
        self.graph = graph_builder.build()

    async def analyze_service_scope(self, raw_address: str, request_id: str = "") -> ServiceScopeResponse:
        """
        执行服务范围分析。

        Args:
            raw_address: 原始地址字符串
            request_id: 请求ID

        Returns:
            ServiceScopeResponse: 分析结果
        """

        # 准备图输入状态
        initial_state = GraphState(
            request_id=request_id or "default_request",
            raw_address=raw_address
        )

        try:
            # 执行图
            final_state = await self.graph.ainvoke(initial_state, subgraphs=True)

            # 转换为响应格式
            # 由于GraphState是BaseModel，而ainvoke返回的是dict，需要转换为GraphState实例
            try:
                if isinstance(final_state, dict):
                    graph_state = GraphState(**final_state)
                else:
                    # 如果已经是GraphState实例，直接使用
                    graph_state = final_state
                return self._convert_to_response(graph_state)
            except Exception as e:
                logger.error(f"Graph state conversion error: {str(e)}")
                logger.error(f"final_state type: {type(final_state)}, keys: {list(final_state.keys()) if isinstance(final_state, dict) else 'Not a dict'}")
                raise

        except Exception as e:
            logger.error(f"ServiceScopeAgent 执行错误: {str(e)}")
            # 处理执行错误
            return ServiceScopeResponse(
                request_id=request_id,
                recommended_station={},
                candidate_stations=[],
                decision_confidence=0.0,
                manual_review=True,
                reason_codes=["SYSTEM_ERROR"],
                recommendation_text=f"服务范围分析失败: {str(e)}",
                risk_summary="系统错误，需要人工介入",
                reasons=["系统错误"],
                errors=[str(e)]
            )

    def _convert_to_response(self, state: GraphState) -> ServiceScopeResponse:
        """
        将图状态转换为响应格式。

        Args:
            state: 图执行完成后的状态

        Returns:
            ServiceScopeResponse: 标准化的响应
        """
        # 转换推荐网点
        recommended_station = {}
        if state.decision and state.decision.recommended_station_id:
            recommended_station = {
                "station_id": state.decision.recommended_station_id,
                "station_name": state.decision.recommended_station_name,
            }

        # 转换候选网点列表
        candidate_stations = []
        if state.decision and state.decision.candidate_ranked:
            candidate_stations = [
                {
                    "station_id": station.station_id,
                    "station_name": station.station_name,
                    "score": station.score
                }
                for station in state.decision.candidate_ranked
            ]

        # 转换决策置信度
        confidence_map = {"high": 1.0, "medium": 0.7, "low": 0.3}
        decision_confidence = confidence_map.get(state.decision.decision_confidence, 0.0)

        # 获取最终结果文本
        recommendation_text = state.final_result.recommendation_text if state.final_result else ""
        risk_summary = state.final_result.risk_summary if state.final_result else ""
        reasons = state.final_result.reasons if state.final_result else []

        return ServiceScopeResponse(
            request_id=state.request_id,
            recommended_station=recommended_station,
            candidate_stations=candidate_stations,
            decision_confidence=decision_confidence,
            manual_review=state.decision.manual_review if state.decision else False,
            reason_codes=state.decision.reason_codes if state.decision else [],
            recommendation_text=recommendation_text,
            risk_summary=risk_summary,
            reasons=reasons,
            errors=state.errors or []
        )

    async def execute(self, question: str, session_id: str, user_token: str):
        raise NotImplementedError("暂不提供实现.")

    async def execute_one_shot(self, question: str, session_id: str, user_token: str):
        """
        执行服务范围分析（一次性输出）

        Args:
            question: 用户问题，预期包含地址信息
            session_id: 会话ID
            user_token: 用户令牌

        Returns:
            ServiceScopeResponse: 分析结果
        """
        # 从问题中提取地址（简单实现，实际可能需要更复杂的解析）
        # 假设question直接就是地址
        raw_address = question.strip()

        # 使用新的图执行方法
        return await self.analyze_service_scope(raw_address, session_id)

    def is_streaming(self) -> bool:
        return False

    def id(self) -> int:
        """ServiceScopeAgent 的唯一 ID，用于系统标识"""
        return 1007

    # --- 以下两个方法 ServiceScopeAgent 不支持，会话不需要状态保存 ---

    async def session_detail(self, user_id: int, session_id: str) -> list:
        """ServiceScopeAgent 不进行会话持久化，因此无需实现 session_detail"""
        raise NotImplementedError("暂不提供实现.")

    async def delete_session(self, session_id: str):
        """ServiceScopeAgent 不具备持久化能力，因此无需删除会话"""
        raise NotImplementedError("暂不提供实现.")


# 全局智能体实例
service_scope_agent = ServiceScopeAgent()