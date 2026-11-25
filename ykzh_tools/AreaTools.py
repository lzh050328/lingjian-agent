import random

from langchain.tools import tool

from ykzh_tools.result import AreaInfo
from ykzh_config import config_manager, logger
from ykzh_util import HttpClientUtil, JsonUtil
from ykzh_util.NacosServiceSelector import NacosServiceSelector


"""行政区域信息查询工具 - 提供按合并名称或简称查询区域数据的 LangChain 工具"""


def _query_area_data(url: str, tool_name: str):
    """
    通用的区域数据查询方法

    Args:
        url: 请求 URL
        tool_name: 工具名称（用于日志）

    Returns:
        JSON 序列化的区域信息列表
    """

    # 使用 NacosServiceSelector 选择服务实例并构建 URL
    full_url = NacosServiceSelector.get_url(config_manager.get("cc.services.base"), url)
    if not full_url:
        return None

    # 发起 HTTP GET 请求获取区域数据
    response_data = HttpClientUtil.get(full_url, "") or []

    logger.debug("【Tool】 %s url=%s, data=%s", tool_name, full_url, response_data)

    if not response_data:
        logger.error(f"Failed to fetch area data from {full_url}")
        return None

    # 转换为 AreaInfo 对象列表
    area_info_list = [JsonUtil.to_obj(area_data, AreaInfo) for area_data in response_data]

    # 将结果序列化 json，返回给大模型
    return JsonUtil.to_str(area_info_list)


@tool
def query_area_by_merger_name(merger_name):
    """
    根据合并名称模糊查询行政区域数据，并将结果存储到 ToolResultHolder。

    Args:
        merger_name : 行政区域合并名称

    Returns:
        JSON 字符串，包含区域信息列表。每个区域对象包含以下字段：
        - id: 区域 ID
        - parentId: 父级行政区 ID
        - name: 行政名称
        - areaCode: 行政区域编号
        - cityCode: 城市编码
        - mergerName: 合并名称
        - shortName: 简称
        - zipCode: 邮政编码
        - level: 行政区域等级（0: 省级，1: 市级，2: 区级）- 用于标识行政级别
        - lng: 经度
        - lat: 纬度
        - pinyin: 拼音
        - first: 首字母
    """
    url = f"/area/mergerName/{merger_name}"
    return _query_area_data(url, "query_area_by_merger_name")


@tool
def query_area_by_short_name(short_name):
    """
    根据简称获取行政区域详情，并将结果存储到 ToolResultHolder。

    Args:
        short_name : 行政区域简称
    
    Returns:
        JSON 字符串，包含区域信息列表。每个区域对象包含以下字段：
        - id: 区域 ID
        - parentId: 父级行政区 ID
        - name: 行政名称
        - areaCode: 行政区域编号
        - cityCode: 城市编码
        - mergerName: 合并名称
        - shortName: 简称
        - zipCode: 邮政编码
        - level: 行政区域等级（0: 省级，1: 市级，2: 区级）- 用于标识行政级别
        - lng: 经度
        - lat: 纬度
        - pinyin: 拼音
        - first: 首字母
    """
    url = f"/area/shortName/{short_name}"
    return _query_area_data(url, "query_area_by_short_name")

if __name__ == "__main__":
    print(_query_area_data(f"/area/shortName/上海", "query_area_by_short_name"))
