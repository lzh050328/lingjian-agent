from dataclasses import dataclass
from typing import Optional

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph

from lingjian_agent.BaseAgent import *
from lingjian_agent.prompts import system_prompt_config


@dataclass
class OrganTypeResponse:
    """机构类型识别响应结构"""
    organType: str  # OLT|TLT|AGENCY|UNKNOWN
    confidence: float  # 0-1之间的小数
    reason: str  # 简短原因，30字以内
    suggestReview: bool  # 是否需要审核
    normalizedName: str  # 标准化名称


class OrganTypeAgent(BaseAgent):

    def __init__(self, provider: str = "openai"):
        self.provider = provider
        self.llm: Optional[BaseChatModel] = None  # LLM 实例
        self.agent: Optional[CompiledStateGraph] = None  # LangChain agent 实例

    async def init(self):
        """
        初始化 OrganTypeAgent：
        1. 创建 LLM 实例
        2. 创建 LangChain Agent（无工具）
        """
        # 初始化 LLM（统一走 openai-compatible 接口）
        self.llm = super().init_llm(self.provider)

        # 创建一个无工具的通用智能体
        self.agent = create_agent(
            model=self.llm
        )

    async def execute(self, question: str, session_id: str, user_token: str):
        raise NotImplementedError("暂不提供实现.")

    async def execute_one_shot(self, question: str, session_id: str, user_token: str):
        """
        执行机构类型识别（一次性输出）
        """
        prompts = [
            SystemMessage(system_prompt_config.chat_organ_type_message),
            HumanMessage(question)
        ]

        result = await self.agent.ainvoke({"messages": prompts})
        return parse_agent_response(result, OrganTypeResponse)

    def is_streaming(self) -> bool:
        return False

    def id(self) -> int:
        """OrganTypeAgent 的唯一 ID，用于系统标识"""
        return 1005

    # --- 以下两个方法 OrganTypeAgent 不支持，会话不需要状态保存 ---

    async def session_detail(self, user_id: int, session_id: str) -> list:
        """OrganTypeAgent 不进行会话持久化，因此无需实现 session_detail"""
        raise NotImplementedError("暂不提供实现.")

    async def delete_session(self, session_id: str):
        """OrganTypeAgent 不具备持久化能力，因此无需删除会话"""
        raise NotImplementedError("暂不提供实现.")


# 全局智能体实例
organ_type_agent = OrganTypeAgent()