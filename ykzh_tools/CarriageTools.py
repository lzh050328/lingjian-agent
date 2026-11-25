from langchain.tools import tool

from ykzh_config import logger, config_manager
from ykzh_tools.result import CarriageDTO
from ykzh_util import HttpClientUtil, JsonUtil
from ykzh_util.NacosServiceSelector import NacosServiceSelector

"""运费计算工具 - 提供运费计算的 LangChain 工具"""


def _compute_carriage_data(url: str, tool_name: str, request_data: dict):
    """
    通用的运费计算数据请求方法

    Args:
        url: 请求 URL
        tool_name: 工具名称（用于日志）
        request_data: 请求数据

    Returns:
        JSON 序列化的运费计算结果
    """

    # 使用 NacosServiceSelector 选择服务实例并构建 URL
    server_name = config_manager.get("cc.services.carriage")
    full_url = NacosServiceSelector.get_url(server_name, url)
    if not full_url:
        return None

    # 发起 HTTP POST 请求计算运费
    response_data = HttpClientUtil.post(full_url, "", json=request_data)

    logger.debug("【Tool】 %s url=%s, request_data=%s, response_data=%s", tool_name, full_url, request_data,
                 response_data)

    if not response_data:
        logger.error(f"Failed to compute carriage from {full_url}")
        return None

    # 转换为 CarriageDTO 对象
    carriage_info = JsonUtil.to_obj(response_data, CarriageDTO)

    # 将结果序列化 json，返回给大模型
    return JsonUtil.to_str(carriage_info)


@tool
def compute_carriage(senderCityId,
                     receiverCityId,
                     weight,
                     *,
                     volume: int = 1,
                     measureHigh: int = 1,
                     measureLong: int = 1,
                     measureWidth: int = 1):
    """
    根据运单信息计算运费

    Args:
        senderCityId: 寄件地址id，市级 (必填)
        receiverCityId: 收件地址id，市级 (必填)
        weight: 重量，单位kg (必填)
        volume: 体积，单位cm^3 (可选)
        measureHigh: 高，单位cm (可选)
        measureLong: 长，单位cm (可选)
        measureWidth: 宽，单位cm (可选)

    Returns:
        JSON 字符串，包含运费计算结果。包含以下字段：
        - associatedCityList: 关联城市：1-全国，2-京津冀，3-江浙沪，4-川渝，5-黑吉辽
        - computeWeight: 计费重量，单位：kg
        - continuousWeight: 续重价格
        - created: 创建时间
        - expense: 运费
        - firstWeight: 首重价格
        - id: 运费模板id
        - lightThrowingCoefficient: 轻抛系数
        - templateType: 模板类型：1-同城寄，2-省内寄，3-经济区互寄，4-跨省
        - transportType: 运送类型：1-普快，2-特快
        - updated: 更新时间
    """

    # 构建请求数据
    request_data = {
        "measureHigh": measureHigh,
        "measureLong": measureLong,
        "measureWidth": measureWidth,
        "receiverCityId": receiverCityId,
        "senderCityId": senderCityId,
        "volume": volume,
        "weight": weight
    }

    # 移除None值的字段
    request_data = {k: v for k, v in request_data.items() if v is not None}

    url = "/carriages/compute"
    return _compute_carriage_data(url, "compute_carriage", request_data)

# if __name__ == "__main__":
#     # 测试运费计算
#     result = compute_carriage(receiverCityId=13268, senderCityId=7363, weight=2.5)
#     print(result)
