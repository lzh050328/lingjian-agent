# ============================
# server 配置
# ============================
SERVER_PORT = "server.port"  # 服务端口
SERVER_HOST = "server.host"  # 监听所有网络接口
SERVER_LOGGER_FILE = "server.logger.file"  # 日志输出文件
SERVER_LOGGER_LEVEL = "server.logger.level"  # 日志级别

# ============================
# nacos 配置
# ============================
NACOS_SERVER_ADDR = "nacos.server-addr"  # NACOS 注册中心地址
NACOS_USERNAME = "nacos.username"  # NACOS 用户名
NACOS_PASSWORD = "nacos.password"  # NACOS 密码

NACOS_CONFIG_NAMESPACE = "nacos.config.namespace"  # NACOS 配置 namespace
NACOS_CONFIG_GROUP = "nacos.config.group"  # NACOS 配置 group

NACOS_DISCOVERY_NAME = "nacos.discovery.name"  # 服务注册名称
NACOS_DISCOVERY_NAMESPACE = "nacos.discovery.namespace"  # 服务发现 namespace
NACOS_DISCOVERY_IP = "nacos.discovery.ip"  # 服务注册 IP

# ============================
# AI OpenAI 配置
# ============================
AI_OPENAI_MODEL = "ai.openai.model"  # 模型名称
AI_OPENAI_API_KEY = "ai.openai.api-key"  # API Key
AI_OPENAI_BASE_URL = "ai.openai.base-url"  # Base URL
AI_OPENAI_TEMPERATURE = "ai.openai.temperature"  # Temperature 参数
AI_OPENAI_TIMEOUT = "ai.openai.timeout"  # 请求超时时间

# ============================
# prompt 配置
# ============================
PROMPT_RISK_CHAT_DATA_ID = "prompt.risk"
PROMPT_REFRESH_TOKEN_RISK_CHAT_DATA_ID = "prompt.refresh-token-risk"
PROMPT_COMPUTE_CHAT_DATA_ID = "prompt.compute"
PROMPT_PARCEL_ITEM_CHECK_CHAT_DATA_ID = "prompt.parcel-item-check"
PROMPT_ORGAN_TYPE_CHAT_DATA_ID = "prompt.organ-type"
PROMPT_COST_FACTOR_CHAT_DATA_ID = "prompt.cost-factor"
PROMPT_ADDRESS_ANALYSIS_CHAT_DATA_ID = "prompt.address-analysis"
PROMPT_STATION_FACT_CHAT_DATA_ID = "prompt.station-fact"
PROMPT_REASON_GENERATION_CHAT_DATA_ID = "prompt.reason-generation"
PROMPT_COURIER_DISPATCH_CHAT_DATA_ID = "prompt.courier-dispatch"