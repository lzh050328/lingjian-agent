from lingjian_agent.BaseAgent import BaseAgent
from lingjian_agent.risk import risk_control_agent, refresh_token_risk_control_agent
from lingjian_agent.carriage import compute_carriage_agent
from lingjian_agent.categorie import parcel_item_check_agent
from lingjian_agent.transport import organ_type_agent, cost_factor_agent
from lingjian_agent.service_scope import service_scope_agent
from lingjian_agent.dispatch import courier_dispatch_agent, transport_order_dispatch_agent, transport_task_dispatch_agent

# 智能体id 与 对应的智能体实例
AGENTS: dict[int, BaseAgent] = {
    risk_control_agent.id(): risk_control_agent,  # 1001
    refresh_token_risk_control_agent.id(): refresh_token_risk_control_agent,  # 1002
    compute_carriage_agent.id(): compute_carriage_agent,  # 1003
    parcel_item_check_agent.id(): parcel_item_check_agent,  # 1004
    organ_type_agent.id(): organ_type_agent,  # 1005
    cost_factor_agent.id(): cost_factor_agent,  # 1006
    service_scope_agent.id(): service_scope_agent,  # 1007
    courier_dispatch_agent.id(): courier_dispatch_agent,  # 1008
    transport_order_dispatch_agent.id(): transport_order_dispatch_agent,  # 1009
    transport_task_dispatch_agent.id(): transport_task_dispatch_agent,  # 1010
}
