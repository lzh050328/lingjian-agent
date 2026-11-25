from dataclasses import dataclass
from typing import Optional, Dict, Any

from lingjian_agent.BaseAgent import *
from lingjian_agent.dispatch.transport_task.service.PredictService import predict_service
from lingjian_util import JsonUtil

@dataclass
class TransportTaskDispatchResponse:
    """运输任务调度响应结构"""
    prediction_label: str
    confidence: float
    reason: str


class TransportTaskDispatchAgent(BaseAgent):

    def __init__(self):
        # 不需要provider参数，因为不调用LLM
        self.predict_service = predict_service  # 预测服务实例

    async def init(self):
        """
        初始化 TransportTaskDispatchAgent：
        该类不调用LLM，只需要确保预测服务已加载
        """
        # 不需要初始化LLM，预测服务已在模块导入时初始化
        pass

    async def execute(self, question: str, session_id: str, user_token: str):
        raise NotImplementedError("暂不提供实现.")

    async def execute_one_shot(self, question: str, session_id: str, user_token: str):
        """
        执行运输任务调度决策（一次性输出）

        Args:
            question: 包含运输任务数据的JSON字符串
            session_id: 会话ID
            user_token: 用户令牌

        Returns:
            TransportTaskDispatchResponse对象，包含预测标签、置信度和原因
        """
        try:
            # 解析输入数据
            data = JsonUtil.to_obj(question)

            # 使用预测服务的predict_delay方法获取预测结果
            prediction_result = self.predict_service.predict_delay(data)

            if not prediction_result.get("success", False):
                error_msg = prediction_result.get("error", "未知错误")
                return TransportTaskDispatchResponse(
                    prediction_label="error",
                    confidence=0.0,
                    reason=f"预测失败: {error_msg}"
                )

            # 提取预测结果
            prediction = prediction_result.get("prediction", {})
            prediction_label = prediction.get("prediction_label", "unknown")
            confidence = round(prediction.get("confidence", 0.0), 4)

            # 置信度等级划分和业务建议
            confidence_level = self._get_confidence_level(confidence)
            reason = self._generate_reason(prediction_label, confidence, confidence_level)

            # 构造响应对象
            response = TransportTaskDispatchResponse(
                prediction_label=prediction_label,
                confidence=confidence,
                reason=reason
            )

            return response

        except Exception as e:
            # 错误处理
            error_response = TransportTaskDispatchResponse(
                prediction_label="error",
                confidence=0.0,
                reason=f"预测计算失败: {str(e)}"
            )
            return error_response

    def _get_confidence_level(self, confidence: float) -> str:
        """
        根据置信度值返回置信度等级

        Args:
            confidence: 置信度值 (0-1)

        Returns:
            置信度等级描述
        """
        if confidence >= 0.9:
            return "极高"
        elif confidence >= 0.8:
            return "高"
        elif confidence >= 0.7:
            return "中等"
        elif confidence >= 0.6:
            return "较低"
        else:
            return "低"

    def _generate_reason(self, prediction_label: str, confidence: float, confidence_level: str) -> str:
        """
        根据预测结果和置信度生成详细的业务建议

        Args:
            prediction_label: 预测标签
            confidence: 置信度值
            confidence_level: 置信度等级

        Returns:
            详细的业务建议说明
        """
        if prediction_label == "delayed":
            if confidence >= 0.8:
                # 高置信度延误预测
                return f"预测结果为延误 (置信度: {confidence:.4f}, {confidence_level}置信度)。建议立即调整调度策略：考虑重新分配订单、增加配送资源或优化路线规划。"
            elif confidence >= 0.6:
                # 中等置信度延误预测
                return f"预测结果为延误 (置信度: {confidence:.4f}, {confidence_level}置信度)。建议密切关注并准备应急预案，可考虑部分调度优化。"
            else:
                # 低置信度延误预测
                return f"预测结果为延误 (置信度: {confidence:.4f}, {confidence_level}置信度)。数据存在不确定性，建议人工复核输入数据的合理性。"

        elif prediction_label == "not_delayed":
            if confidence >= 0.8:
                # 高置信度正常预测
                return f"预测结果为正常 (置信度: {confidence:.4f}, {confidence_level}置信度)。当前调度计划可行，可保持现有策略。"
            elif confidence >= 0.6:
                # 中等置信度正常预测
                return f"预测结果为正常 (置信度: {confidence:.4f}, {confidence_level}置信度)。当前调度计划基本可行，建议持续监控。"
            else:
                # 低置信度正常预测
                return f"预测结果为正常 (置信度: {confidence:.4f}, {confidence_level}置信度)。预测可靠性较低，建议检查数据质量或人工复核。"

        else:
            # 未知预测结果
            return f"预测结果为 {prediction_label} (置信度: {confidence:.4f}, {confidence_level}置信度)。建议检查输入数据和模型状态。"

    def is_streaming(self) -> bool:
        return False

    def id(self) -> int:
        """TransportTaskDispatchAgent 的唯一 ID，用于系统标识"""
        return 1010

    # --- 以下两个方法 TransportTaskDispatchAgent 不支持，会话不需要状态保存 ---

    async def session_detail(self, user_id: int, session_id: str) -> list:
        """TransportTaskDispatchAgent 不进行会话持久化，因此无需实现 session_detail"""
        raise NotImplementedError("暂不提供实现.")

    async def delete_session(self, session_id: str):
        """TransportTaskDispatchAgent 不具备持久化能力，因此无需删除会话"""
        raise NotImplementedError("暂不提供实现.")


# 全局智能体实例
transport_task_dispatch_agent = TransportTaskDispatchAgent()