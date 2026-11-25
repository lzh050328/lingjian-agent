from dataclasses import dataclass
from typing import Optional, Dict, Any

from ykzh_agent.BaseAgent import *
from ykzh_agent.dispatch.transport_order.model import transport_order_dispatch_predictor
from ykzh_util import JsonUtil

@dataclass
class TransportOrderDispatchResponse:
    """运输订单调度响应结构"""
    score: float
    reason: str


class TransportOrderDispatchAgent(BaseAgent):

    def __init__(self):
        # 不需要provider参数，因为不调用LLM
        self.predictor = transport_order_dispatch_predictor  # 预测器实例

    async def init(self):
        """
        初始化 TransportOrderDispatchAgent：
        该类不调用LLM，只需要确保预测器已加载
        """
        # 不需要初始化LLM，预测器已在模块导入时初始化
        pass

    async def execute(self, question: str, session_id: str, user_token: str):
        raise NotImplementedError("暂不提供实现.")

    async def execute_one_shot(self, question: str, session_id: str, user_token: str):
        """
        执行运输订单调度决策（一次性输出）

        Args:
            question: 包含运输订单数据的JSON字符串
            session_id: 会话ID
            user_token: 用户令牌

        Returns:
            TransportOrderDispatchResponse对象，包含评分和原因
        """
        try:
            # 解析输入数据
            data = JsonUtil.to_obj(question)

            # 使用预测器的score方法获取评分
            score_result = self.predictor.score(data)

            # 根据评分生成原因说明
            score = round(score_result.get("score", 0.0), 5)
            if score >= 0.85:
                reason = f"运输订单优先级评分较高 ({score})，建议优先调度"
            elif score >= 0.5:
                reason = f"运输订单优先级评分中等 ({score})，可正常排队调度"
            else:
                reason = f"运输订单优先级评分较低 ({score})，可延后处理"

            # 构造响应对象
            response = TransportOrderDispatchResponse(
                score=score,
                reason=reason
            )

            return response

        except Exception as e:
            # 错误处理
            error_response = TransportOrderDispatchResponse(
                score=0.0,
                reason=f"评分计算失败: {str(e)}"
            )
            return error_response

    def is_streaming(self) -> bool:
        return False

    def id(self) -> int:
        """TransportOrderDispatchAgent 的唯一 ID，用于系统标识"""
        return 1009

    # --- 以下两个方法 TransportOrderDispatchAgent 不支持，会话不需要状态保存 ---

    async def session_detail(self, user_id: int, session_id: str) -> list:
        """TransportOrderDispatchAgent 不进行会话持久化，因此无需实现 session_detail"""
        raise NotImplementedError("暂不提供实现.")

    async def delete_session(self, session_id: str):
        """TransportOrderDispatchAgent 不具备持久化能力，因此无需删除会话"""
        raise NotImplementedError("暂不提供实现.")


# 全局智能体实例
transport_order_dispatch_agent = TransportOrderDispatchAgent()