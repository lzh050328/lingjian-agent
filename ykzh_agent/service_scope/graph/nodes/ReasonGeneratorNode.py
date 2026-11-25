import asyncio
from typing import Dict, Any

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ykzh_agent.BaseAgent import parse_agent_response
from ykzh_agent.prompts import system_prompt_config
from ykzh_agent.service_scope.graph.domain.models import GraphState, ReasonGenerationResponse
from ykzh_agent.service_scope.graph.nodes.BaseGraphNode import BaseGraphNode


class ReasonGeneratorNode(BaseGraphNode):
    """
    推荐理由生成节点。

    职责：
    1. 将结构化事实和决策结果组织成自然语言
    2. 给调用方/运营提供可读解释
    3. 严格基于事实，不允许编造

    说明：
    - 这个节点体现的是 AI 在"解释"上的价值
    - 真正的推荐结论仍然来自规则节点
    """

    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm
        self.reason_agent = None

    async def init_agent(self) -> None:
        if self.reason_agent is not None:
            return

        # 创建理由生成智能体，不需要额外工具，只需LLM
        self.reason_agent = create_agent(
            model=self.llm
        )

    async def run(self, state: GraphState) -> Dict[str, Any]:
        """
        执行推荐理由生成。

        流程：
        1. 构造agent输入数据
        2. 调用Agent生成推荐理由
        3. 解析结构化响应并返回
        """
        await self.init_agent()

        # 构造输入数据
        payload = {
            "raw_address": state.raw_address,
            "address_analysis": state.address_analysis.model_dump(),
            "location": state.location.model_dump(),
            "station_facts": [item.model_dump() for item in state.station_facts],
            "decision": state.decision.model_dump(),
        }

        # 构造agent输入消息
        agent_input = [
            SystemMessage(system_prompt_config.chat_reason_generation_message),
            HumanMessage(
                f"请根据以下结构化事实生成物流网点推荐说明：\n{str(payload)}"
            )
        ]

        # 调用agent
        result = await self.reason_agent.ainvoke({"messages": agent_input})

        # 解析响应
        structured = parse_agent_response(result, ReasonGenerationResponse)

        if structured is None:
            # 如果解析失败，返回默认的空结果
            structured = ReasonGenerationResponse(
                recommendation_text="推荐理由生成失败，请稍后重试。",
                risk_summary="系统暂时无法生成推荐说明。",
                reasons=["系统错误"]
            )

        return {
            "final_result": structured.model_dump()
        }
