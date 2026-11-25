from langchain.tools import tool

from lingjian_config import logger, config_manager
from lingjian_tools.result import ServiceScopeDTO
from lingjian_util import HttpClientUtil, JsonUtil
from lingjian_util.NacosServiceSelector import NacosServiceSelector

"""服务范围查询工具 - 提供基于位置的查询服务范围的 LangChain 工具"""


def _query_service_scope_data(url: str, tool_name: str, params: dict):
    """
    通用的服务范围查询数据请求方法

    Args:
        url: 请求 URL
        tool_name: 工具名称（用于日志）
        params: 请求参数

    Returns:
        JSON 序列化的服务范围查询结果
    """

    # 使用 NacosServiceSelector 选择服务实例并构建 URL
    server_name = config_manager.get("cc.services.service-scope")
    full_url = NacosServiceSelector.get_url(server_name, url)
    if not full_url:
        return None

    # 发起 HTTP GET 请求查询服务范围
    response_data = HttpClientUtil.get(full_url, "", params)

    logger.debug("【Tool】 %s url=%s, params=%s, response_data=%s", tool_name, full_url, params, response_data)

    if not response_data:
        logger.error(f"Failed to query service scope from {full_url}")
        return None

    # 转换为 ServiceScopeDTO 对象列表
    service_scope_list = []
    if isinstance(response_data, list):
        for item in response_data:
            service_scope_info = JsonUtil.to_obj(item, ServiceScopeDTO)
            service_scope_list.append(service_scope_info)

    # 将结果序列化 json，返回给大模型
    return service_scope_list


def query_service_scope_list(type: int, longitude: float, latitude: float):
    """
    根据位置信息查询服务范围

    Args:
        type: 类型，1-机构，2-快递员 (必填)
        longitude: 经度 (必填)
        latitude: 纬度 (必填)

    Returns:
        ServiceScopeDTO 列表。字段：
        - bid: 业务id，可以是机构或快递员
        - type: 类型，1-机构，2-快递员
        - polygon: 多边形坐标点列表，至少3个坐标点，首尾坐标必须相同
        - created: 创建时间
        - updated: 更新时间
    """

    # 构建请求参数
    params = {
        "type": type,
        "longitude": longitude,
        "latitude": latitude
    }

    url = "/scopes/location"
    return _query_service_scope_data(url, "query_service_scope_list", params)

def query_adjacent_service_scope_list(type: int, longitude: float, latitude: float, limit: int = 10):
    """
    根据位置信息查询服务范围

    Args:
        type: 类型，1-机构，2-快递员 (必填)
        longitude: 经度 (必填)
        latitude: 纬度 (必填)
        limit: 数量，默认：10

    Returns:
        ServiceScopeDTO 列表。字段：
        - bid: 业务id，可以是机构或快递员
        - type: 类型，1-机构，2-快递员
        - polygon: 多边形坐标点列表，至少3个坐标点，首尾坐标必须相同
        - created: 创建时间
        - updated: 更新时间
    """

    # 构建请求参数
    params = {
        "type": type,
        "longitude": longitude,
        "latitude": latitude,
        "limit": limit
    }

    url = "/scopes/adjacent"
    return _query_service_scope_data(url, "query_adjacent_service_scope_list", params)

@tool
def query_service_scope_tool(type: int, longitude: float, latitude: float):
    """
    根据位置信息查询服务范围

    Args:
        type: 类型，1-机构，2-快递员 (必填)
        longitude: 经度 (必填)
        latitude: 纬度 (必填)

    Returns:
        JSON 字符串，包含服务范围查询结果列表。每个结果包含以下字段：
        - bid: 业务id，可以是机构或快递员
        - type: 类型，1-机构，2-快递员
        - polygon: 多边形坐标点列表，至少3个坐标点，首尾坐标必须相同
        - created: 创建时间
        - updated: 更新时间
    """

    return JsonUtil.to_str(query_service_scope_list(type, longitude, latitude))

# if __name__ == "__main__":
#     # 测试服务范围查询
#     result = query_service_scope_by_location(type=1, longitude=121.612519, latitude=31.034925)
#     print(result)
