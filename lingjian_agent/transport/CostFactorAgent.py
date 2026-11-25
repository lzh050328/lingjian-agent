from dataclasses import dataclass
from typing import Optional, List

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph

from lingjian_agent.BaseAgent import *
from lingjian_agent.prompts import system_prompt_config
from lingjian_agent.mcp import mcp_service


@dataclass
class CostFactorResponse:
    """路线成本因子分析响应结构"""
    regionFactor: float  # 区域关系修正因子 (0.95-1.10)
    distanceSegmentFactor: float  # 距离分段修正因子 (0.95-1.10)
    efficiencyFactor: float  # 路径效率修正因子 (0.95-1.10)
    tollFactor: float  # 收费道路修正因子 (0.95-1.10)
    detourFactor: float  # 绕行修正因子 (0.95-1.10)
    routeComplexityFactor: float  # 路径复杂度修正因子 (0.95-1.10)
    corridorFactor: float  # 成熟通道修正因子 (0.95-1.10)
    similarityCalibrationFactor: float  # 相似路线经验校准因子 (0.95-1.10)
    accessFactor: float  # 起终点接入难度修正因子 (0.95-1.10)
    confidence: float  # 整体可信度评估 (0.00-1.00)
    usedTools: List[str]  # 实际调用的MCP工具名称列表
    reason: str  # 判断依据的简短说明 (不超过300字符)


class CostFactorAgent(BaseAgent):

    def __init__(self, provider: str = "openai"):
        self.provider = provider
        self.llm: Optional[BaseChatModel] = None  # LLM 实例
        self.agent: Optional[CompiledStateGraph] = None  # LangChain agent 实例

    async def init(self):
        """
        初始化 CostFactorAgent：
        1. 创建 LLM 实例
        2. 创建 LangChain Agent（无工具）
        """
        # 初始化 LLM（统一走 openai-compatible 接口）
        self.llm = super().init_llm(self.provider)

        # 获取高德提供的所有工具
        amap_tools = await mcp_service.get_tools("amap_mcp")

        # 创建智能体
        self.agent = create_agent(
            model=self.llm,
            tools=amap_tools
        )

    async def execute(self, question: str, session_id: str, user_token: str):
        raise NotImplementedError("暂不提供实现.")

    async def execute_one_shot(self, question: str, session_id: str, user_token: str):
        """
        执行路线成本因子分析（一次性输出）
        """
        prompts = [
            SystemMessage(system_prompt_config.chat_cost_factor_message),
            HumanMessage(question)
        ]

        result = await self.agent.ainvoke({"messages": prompts})
        return parse_agent_response(result, CostFactorResponse)

    def is_streaming(self) -> bool:
        return False

    def id(self) -> int:
        """CostFactorAgent 的唯一 ID，用于系统标识"""
        return 1006

    # --- 以下两个方法 CostFactorAgent 不支持，会话不需要状态保存 ---

    async def session_detail(self, user_id: int, session_id: str) -> list:
        """CostFactorAgent 不进行会话持久化，因此无需实现 session_detail"""
        raise NotImplementedError("暂不提供实现.")

    async def delete_session(self, session_id: str):
        """CostFactorAgent 不具备持久化能力，因此无需删除会话"""
        raise NotImplementedError("暂不提供实现.")


# 全局智能体实例
cost_factor_agent = CostFactorAgent()