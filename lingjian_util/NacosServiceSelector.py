import random
from typing import List, Dict, Optional

from lingjian_config import nacos_config, logger


class NacosServiceSelector:
    """
    Nacos 服务选择器 - 提供负载均衡的服务实例选择功能
    """

    @staticmethod
    def select_random_instance(service_name: str) -> Optional[Dict]:
        """
        从 Nacos 获取指定服务的实例列表，并随机选择一个实例

        Args:
            service_name: 服务名称

        Returns:
            选中的服务实例信息，包含 ip 和 port，如果没有可用实例则返回 None
        """
        # 从 Nacos 获取微服务列表
        instances = nacos_config.get_discovery_client().list_naming_instance(service_name).get("hosts", [])
        if not instances:
            logger.error(f"No {service_name} instances found")
            return None

        # 随机选择一个网关实例发起请求，分散负载
        instance = random.choice(instances)
        return instance

    @staticmethod
    def build_service_url(instance: Dict, path: str) -> str:
        """
        构建完整的服务 URL

        Args:
            instance: 服务实例信息
            path: API 路径

        Returns:
            完整的 URL
        """
        return f"http://{instance['ip']}:{instance['port']}{path}"

    @staticmethod
    def get_url(service_name: str, path: str) -> Optional[str]:
        """
        选择服务实例并构建完整 URL

        Args:
            service_name: 服务名称
            path: API 路径

        Returns:
            完整的 URL，如果没有可用实例则返回 None
        """
        instance = NacosServiceSelector.select_random_instance(service_name)
        if not instance:
            return None

        return NacosServiceSelector.build_service_url(instance, path)