import asyncio
from typing import Any, Dict

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ykzh_agent.BaseAgent import parse_agent_response
from ykzh_agent.mcp import mcp_service
from ykzh_agent.prompts import system_prompt_config
from ykzh_agent.service_scope.graph.domain.models import (
    AddressAnalysis,
    AddressAnalysisAgentResponse,
    GraphState,
    Location,

)
from ykzh_agent.service_scope.graph.nodes import BaseGraphNode
from ykzh_config import logger

class AddressAnalyzerAgentNode(BaseGraphNode):
    """
    地址分析 Agent 节点。

    职责：
    1. 使用 LangChain Agent 对原始地址做语义理解
    2. 必要时通过高德 MCP tools 调地图能力
    3. 输出：
       - address_analysis
       - location

    """

    def __init__(self, llm: BaseChatModel) -> None:
        # 保留LLM 实例
        self.llm = llm
        self.address_agent = None

    async def init_agent(self) -> None:
        if self.address_agent is not None:
            return

        # 获取高德提供的所有工具
        amap_tools = await mcp_service.get_tools("amap_mcp")

        # 创建智能体
        self.address_agent = create_agent(
            model=self.llm,
            tools=amap_tools
        )

    async def run(self, state: GraphState) -> Dict[str, Any]:
        """
        执行地址分析。

        流程：
        1. 构造 agent 输入
        2. 调用 Agent
        3. 从 structured_response 中拿到结构化结果
        4. 写回 address_analysis 和 location

        注意：
        - Agent 如果没拿到可靠坐标，会返回 geocode_success=False
        - 后续图中会根据 geocode_success 决定是否走 geocode fallback
        """
        # 延迟初始化agent对象
        await self.init_agent()

        raw_address = state.raw_address

        agent_input = [
            SystemMessage(system_prompt_config.chat_address_analysis_message),
            HumanMessage(f"请分析并定位这个物流发件地址：{raw_address}")
        ]

        result = await self.address_agent.ainvoke({"messages": agent_input})

        agent_response = parse_agent_response(result, AddressAnalysisAgentResponse)

        logger.info(f"地址分析结果：{agent_response}")

        address_analysis = AddressAnalysis(
            normalized_address=agent_response.normalized_address,
            address_type=agent_response.address_type,
            address_confidence=agent_response.address_confidence,
            risk_tags=agent_response.risk_tags,
            rewrite_queries=agent_response.rewrite_queries,
            description=agent_response.description
        )

        location = Location(
            lng=agent_response.lng,
            lat=agent_response.lat,
            geocode_source=agent_response.geocode_source,
            geocode_confidence=agent_response.geocode_confidence,
            geocode_success=agent_response.geocode_success,
        )

        return {
            "address_analysis": address_analysis,
            "location": location,
        }
