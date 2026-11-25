from dataclasses import dataclass
from typing import Optional

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph

from lingjian_agent.BaseAgent import *
from lingjian_agent.prompts import system_prompt_config
from lingjian_agent.dispatch.courier.tools import query_courier_info_single, query_courier_info_batch, predict_courier_dispatch_single, predict_courier_dispatch_batch

@dataclass
class CourierDispatchResponse:
    """快递员调度响应结构"""
    courierId: str
    reason: str


class CourierDispatchAgent(BaseAgent):

    def __init__(self, provider: str = "openai"):
        self.provider = provider
        self.llm: Optional[BaseChatModel] = None  # LLM 实例
        self.agent: Optional[CompiledStateGraph] = None  # LangChain agent 实例

    async def init(self):
        """
        初始化 CourierDispatchAgent：
        1. 创建 LLM 实例
        2. 创建 LangChain Agent
        """
        # 初始化 LLM（统一走 openai-compatible 接口）
        self.llm = super().init_llm(self.provider)

        # 创建一个智能体，包含快递员特征查询和预测工具
        self.agent = create_agent(
            model=self.llm,
            tools=[query_courier_info_single, query_courier_info_batch,
                  predict_courier_dispatch_single, predict_courier_dispatch_batch]
        )

    async def execute(self, question: str, session_id: str, user_token: str) -> AsyncIterable[str]:
        raise NotImplementedError("暂不提供实现.")

    async def execute_one_shot(self, question: str, session_id: str, user_token: str):
        """
        执行快递员调度决策（一次性输出）
        """
        prompts = [
            SystemMessage(system_prompt_config.chat_courier_dispatch_message),
            HumanMessage(question)
        ]

        result = await self.agent.ainvoke({"messages": prompts})
        return parse_agent_response(result, CourierDispatchResponse)

    def is_streaming(self) -> bool:
        return False

    def id(self) -> int:
        """CourierDispatchAgent 的唯一 ID，用于系统标识"""
        return 1008

    # --- 以下两个方法 CourierDispatchAgent 不支持，会话不需要状态保存 ---

    async def session_detail(self, user_id: int, session_id: str) -> list:
        """CourierDispatchAgent 不进行会话持久化，因此无需实现 session_detail"""
        raise NotImplementedError("暂不提供实现.")

    async def delete_session(self, session_id: str):
        """CourierDispatchAgent 不具备持久化能力，因此无需删除会话"""
        raise NotImplementedError("暂不提供实现.")


# 全局智能体实例
courier_dispatch_agent = CourierDispatchAgent()