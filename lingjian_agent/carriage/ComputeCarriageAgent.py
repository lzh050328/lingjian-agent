from dataclasses import dataclass
from typing import Optional

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph

from lingjian_agent.BaseAgent import *
from lingjian_agent.prompts import system_prompt_config
from lingjian_tools import query_area_by_merger_name, query_area_by_short_name, compute_carriage


@dataclass
class ComputeResponse:
    """运费计算响应结构"""
    computeWeight: float
    continuousWeight: float
    expense: float
    firstWeight: float
    lightThrowingCoefficient: int
    info: str


class ComputeCarriageAgent(BaseAgent):

    def __init__(self, provider: str = "openai"):
        self.provider = provider
        self.llm: Optional[BaseChatModel] = None  # LLM 实例
        self.agent: Optional[CompiledStateGraph] = None  # LangChain agent 实例

    async def init(self):
        """
        初始化 ComputeCarriageAgent：
        1. 创建 LLM 实例
        2. 创建 LangChain Agent
        """
        # 初始化 LLM（统一走 openai-compatible 接口）
        self.llm = super().init_llm(self.provider)

        # 创建一个智能体
        self.agent = create_agent(
            model=self.llm,
            tools=[query_area_by_merger_name, query_area_by_short_name, compute_carriage]
        )

    async def execute(self, question: str, session_id: str, user_token: str) -> AsyncIterable[str]:
        raise NotImplementedError("暂不提供实现.")

    async def execute_one_shot(self, question: str, session_id: str, user_token: str):
        """
        执行运费计算（一次性输出）
        """
        prompts = [
            SystemMessage(system_prompt_config.chat_compute_message),
            HumanMessage(question)
        ]

        result = await self.agent.ainvoke({"messages": prompts})
        return parse_agent_response(result, ComputeResponse)

    def is_streaming(self) -> bool:
        return False

    def id(self) -> int:
        """ComputeCarriageAgent 的唯一 ID，用于系统标识"""
        return 1003

    # --- 以下两个方法 ComputeCarriageAgent 不支持，会话不需要状态保存 ---

    async def session_detail(self, user_id: int, session_id: str) -> list:
        """ComputeCarriageAgent 不进行会话持久化，因此无需实现 session_detail"""
        raise NotImplementedError("暂不提供实现.")

    async def delete_session(self, session_id: str):
        """ComputeCarriageAgent 不具备持久化能力，因此无需删除会话"""
        raise NotImplementedError("暂不提供实现.")


# 全局智能体实例
compute_carriage_agent = ComputeCarriageAgent()
