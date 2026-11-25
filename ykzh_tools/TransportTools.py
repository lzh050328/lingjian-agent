import random

from langchain.tools import tool

from ykzh_tools.result import OrganInfo
from ykzh_config import config_manager, logger
from ykzh_util import HttpClientUtil, JsonUtil
from ykzh_util.NacosServiceSelector import NacosServiceSelector


"""运输机构信息查询工具 - 提供按ID查询机构数据的 LangChain 工具"""


def _query_organ_data(url: str, tool_name: str):
    """
    通用的机构数据查询方法

    Args:
        url: 请求 URL
        tool_name: 工具名称（用于日志）

    Returns:
        JSON 序列化的机构信息
    """

    # 使用 NacosServiceSelector 选择服务实例并构建 URL
    full_url = NacosServiceSelector.get_url(config_manager.get("cc.services.transport"), url)
    if not full_url:
        return None

    # 发起 HTTP GET 请求获取机构数据
    response_data = HttpClientUtil.get(full_url, "")

    logger.debug("【Tool】 %s url=%s, data=%s", tool_name, full_url, response_data)

    if not response_data:
        logger.error(f"Failed to fetch organ data from {full_url}")
        return None

    # 转换为 OrganInfo 对象
    organ_info = JsonUtil.to_obj(response_data, OrganInfo)

    # 将结果序列化 json，返回给大模型
    return JsonUtil.to_str(organ_info)


@tool
def query_organ_by_id(organ_id):
    """
    根据机构ID查询机构详细信息

    Args:
        organ_id : 机构ID

    Returns:
        JSON 字符串，包含机构信息。机构对象包含以下字段：
        - id: 机构ID
        - name: 机构名称
        - type: 机构类型
        - phone: 联系电话
        - address: 地址信息(JSON格式)，包含省、市、区县信息和详细地址
        - latitude: 纬度
        - longitude: 经度
        - parentId: 父级机构ID
        - managerName: 负责人姓名
        - extra: 扩展信息
        - status: 状态（true表示启用，false表示禁用）
    """
    url = f"/organs/{organ_id}"
    return _query_organ_data(url, "query_organ_by_id")


if __name__ == "__main__":
    print(_query_organ_data(f"/organs/1024981295454874273", "query_organ_by_id"))