import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient

from ykzh_config import config_manager, logger

class MCPService:
    """
    封装MCP服务的客户端
    """

    def __init__(self):
        servers_config = {}
        for item in config_manager.get("mcp-servers", []):
            servers_config[item["name"]] = {
                "transport": item["transport"],
                "url": item["url"]
            }

        self.client = MultiServerMCPClient(servers_config)

    async def get_tools(self, server_name: str | None = None):
        """
        获取所有工具列表
        """
        return await self.client.get_tools(server_name = server_name)

    async def get_tool(self, name: str):
        """
        根据名字获取工具
        """
        try:
            tools = await self.get_tools()
        except Exception as e:
            # 记录日志或处理异常
            logger.error(f"获取工具列表失败: {str(e)}")
            return None

        for tool in tools:
            if tool.name == name:
                return tool
        return None


mcp_service = MCPService()

if __name__ == "__main__":
    # 测试代码
    tools = asyncio.run(mcp_service.get_tools())
    print(f"tools:{tools}")

    print("-" * 100)

    tools = asyncio.run(mcp_service.get_tools("amap_mcp"))
    print(f"tools:{tools}")

    print("-"*100)


    tool = asyncio.run(mcp_service.get_tool("maps_ip_location"))
    print(f"maps_ip_location:{tool}")

    tool = asyncio.run(mcp_service.get_tool("maps_distance"))
    print(f"maps_distance:{tool}")
