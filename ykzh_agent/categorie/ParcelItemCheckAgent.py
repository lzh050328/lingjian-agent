from dataclasses import dataclass
from typing import Optional

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph

from ykzh_agent.BaseAgent import *
from ykzh_agent.categorie import category_predictor
from ykzh_agent.prompts import system_prompt_config
from ykzh_config import logger


@dataclass
class ParcelItemCheckResponse:
    """物品检查响应数据结构"""
    itemName: str  # 物品名称
    categoryId: int  # 物品类别 ID
    categoryName: str  # 物品类别名称
    isProhibited: bool  # 是否违禁
    reason: str  # 原因说明


class ParcelItemCheckAgent(BaseAgent):
    def __init__(self, provider: str = "openai"):
        self.provider = provider
        self.llm: Optional[BaseChatModel] = None  # LLM 实例
        self.agent: Optional[CompiledStateGraph] = None  # LangChain agent 实例

    async def init(self):
        """
        初始化 ParcelItemCheckAgent：
        1. 创建 LLM 实例
        2. 创建 LangChain Agent（无工具）
        """

        # 初始化 LLM（统一走 openai-compatible 接口）
        self.llm = super().init_llm(self.provider)

        # 创建一个无工具的通用智能体
        self.agent = create_agent(
            model=self.llm
        )

    async def execute(self, question: str, session_id: str, user_token: str) -> AsyncIterable[str]:
        raise NotImplementedError("暂不提供实现.")

    async def execute_one_shot(self, question: str, session_id: str, user_token: str):
        """
        执行物品分类检查（非流式/一次性输出）

        :param question: 用户输入的物品名称
        :param session_id: 会话 ID
        :param user_token: 用户令牌
        :return: 包含完整响应的字符串
        """

        # 预测物品类别
        predict_result = category_predictor.predict(question)

        # 判断置信度是否大于等于90%，则返回结果，否在调用LLM
        if predict_result.confidence >= 0.9:
            return ParcelItemCheckResponse(
                itemName=predict_result.text,
                categoryId=predict_result.category_id,
                categoryName=predict_result.category_name,
                isProhibited=predict_result.category_id == 1020,
                reason=f"该物品属于{predict_result.category_name}，置信度{predict_result.confidence:.2%}"
            )

        logger.info(f"物品分类置信度低于90%，预测结果为{predict_result}，调用LLM进行判断")

        prompts = [
            SystemMessage(system_prompt_config.chat_parcel_item_check_message),
            HumanMessage(question)
        ]

        result = await self.agent.ainvoke({"messages": prompts})
        return parse_agent_response(result, ParcelItemCheckResponse)

    def id(self) -> int:
        return 1004

    async def session_detail(self, user_id: int, session_id: str) -> list:
        raise NotImplementedError("暂不提供实现.")

    async def delete_session(self, session_id: str):
        raise NotImplementedError("暂不提供实现.")

    def is_streaming(self) -> bool:
        return False


# 全局智能体实例
parcel_item_check_agent = ParcelItemCheckAgent()
