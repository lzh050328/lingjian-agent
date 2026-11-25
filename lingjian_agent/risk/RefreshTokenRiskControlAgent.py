from dataclasses import dataclass
from enum import Enum
from typing import Optional

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph

from lingjian_agent.BaseAgent import *
from lingjian_agent.mcp import mcp_service
from lingjian_agent.prompts import system_prompt_config


class RiskLevel(Enum):
    """风险等级枚举"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass
class RefreshTokenRiskResponse:
    """登录风险响应"""
    riskLevel: RiskLevel
    riskScore: float
    reason: str


class RefreshTokenRiskControlAgent(BaseAgent):

    def __init__(self, provider: str = "openai"):
        self.provider = provider
        self.llm: Optional[BaseChatModel] = None  # LLM 实例
        self.agent: Optional[CompiledStateGraph] = None  # LangChain agent 实例

    async def init(self):
        """
        初始化 RiskControlAgent：
        1. 创建 LLM 实例
        2. 创建 LangChain Agent
        """
        # 初始化 LLM（统一走 openai-compatible 接口）
        self.llm = super().init_llm(self.provider)

        ip_tool = await mcp_service.get_tool("maps_ip_location")
        distance_tool = await mcp_service.get_tool("maps_distance")

        # 通用智能体(包含了2个工具)
        self.agent = create_agent(
            model=self.llm,
            tools=[ip_tool, distance_tool],
        )

    async def execute(self, question: str, session_id: str, user_token: str) -> AsyncIterable[str]:
        pass

    async def execute_one_shot(self, question: str, session_id: str, user_token: str):
        prompts = [
            SystemMessage(system_prompt_config.chat_refresh_token_risk_message),
            HumanMessage(question)
        ]

        result = await self.agent.ainvoke({"messages": prompts})
        return parse_agent_response(result, RefreshTokenRiskResponse)

    def is_streaming(self) -> bool:
        return False

    def id(self) -> int:
        """RefreshTokenRiskControlAgent 的唯一 ID，用于系统标识"""
        return 1002

    # --- 以下两个方法 RefreshTokenRiskControlAgent 不支持，会话不需要状态保存 ---

    def session_detail(self, user_id: int, session_id: str) -> list:
        """RefreshTokenRiskControlAgent 不进行会话持久化，因此无需实现 session_detail"""
        raise NotImplementedError("暂不提供实现.")

    def delete_session(self, session_id: str):
        """RefreshTokenRiskControlAgent 不具备持久化能力，因此无需删除会话"""
        raise NotImplementedError("暂不提供实现.")


# 全局智能体实例
refresh_token_risk_control_agent = RefreshTokenRiskControlAgent()
