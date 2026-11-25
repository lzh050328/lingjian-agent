"""
JsonUtil - JSON 序列化/反序列化工具类

依赖：
    必须：orjson
    可选：pydantic（支持 BaseModel 映射）
          dacite（更健壮的 dataclass 映射）
          json-repair（从脏文本中提取/修复 JSON）

安装：
    pip install orjson
    pip install pydantic dacite json-repair   # 可选增强
"""

import re
import typing
import logging
import orjson
from dataclasses import is_dataclass, asdict
from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import Any, Type, TypeVar, Union, get_type_hints
from uuid import UUID

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ---------------------------------------------------------------------------
# 可选依赖：模块级缓存，只 import 一次
# ---------------------------------------------------------------------------

try:
    import dacite as _dacite
except ImportError:
    _dacite = None

try:
    import json_repair as _json_repair
except ImportError:
    _json_repair = None

try:
    from pydantic import BaseModel as _BaseModel
except ImportError:
    _BaseModel = None


# ---------------------------------------------------------------------------
# 序列化默认处理器
# ---------------------------------------------------------------------------

def _default_serializer(obj: Any) -> Any:
    """
    orjson default 钩子，处理标准库不支持的类型。
    按出现频率排序，减少不必要的 isinstance 检查。
    """
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        # 整数时返回 int，有小数时返回 str 保留精度（避免浮点误差）
        # 注意：str 在 JSON 中是字符串而非数字，确认下游可接受再使用
        if obj == obj.to_integral_value():
            return int(obj)
        return str(obj)
    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    raise TypeError(f"类型 {type(obj)!r} 不可序列化")


# ---------------------------------------------------------------------------
# JSON 提取核心逻辑
# ---------------------------------------------------------------------------

def _extract_json_from_text(text: str) -> str:
    """
    从任意文本中提取第一个有效 JSON 片段（对象或数组）。

    策略（按优先级）：
    1. 文本本身就是合法 JSON → 直接返回
    2. 剥离 Markdown 代码块 ``` 后尝试解析
    3. json-repair 库修复（如果已安装）
    4. 状态机扫描：正确处理字符串内的括号和转义字符

    Args:
        text: 可能包含 JSON 的任意字符串

    Returns:
        合法的 JSON 字符串

    Raises:
        ValueError: 无法提取有效 JSON
    """
    if not isinstance(text, str):
        raise TypeError(f"期望 str，收到 {type(text)!r}")

    text = text.strip()

    # 快速路径：已经是合法 JSON
    try:
        orjson.loads(text)
        return text
    except orjson.JSONDecodeError:
        pass

    # 剥离 Markdown 代码块：```json ... ``` 或 ``` ... ```
    md_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if md_match:
        candidate = md_match.group(1).strip()
        try:
            orjson.loads(candidate)
            return candidate
        except orjson.JSONDecodeError:
            pass

    # 使用 json-repair（能处理缺逗号、单引号、未加引号的 key 等）
    if _json_repair is not None:
        try:
            repaired = _json_repair.repair_json(text, return_objects=False)
            if repaired and repaired not in ("", '""', "null"):
                orjson.loads(repaired)  # 校验修复结果合法
                return repaired
        except Exception:
            pass  # 修复失败，继续下一策略

    # 状态机提取
    result = _state_machine_extract(text)
    if result:
        return result

    raise ValueError(
        f"无法从文本中提取有效 JSON：{text[:120]}{'...' if len(text) > 120 else ''}"
    )


def _state_machine_extract(text: str) -> str | None:
    """
    用状态机扫描文本，正确区分字符串内容和结构括号。
    相比简单的括号计数，能正确处理：
    - {"key": "val{ue"} 中值里的括号
    - 转义字符 \\" 不提前结束字符串
    - 同时寻找 { } 和 [ ] 两种容器

    修复说明：
    - break 移入 try 块内，只有 orjson.loads 成功才终止内层扫描。
      失败时继续寻找更靠后的闭合位置（可能存在有效的子结构）。
    """
    openers = {'{': '}', '[': ']'}
    best: str | None = None  # 保留最长有效片段

    for start, char in enumerate(text):
        if char not in openers:
            continue

        closer = openers[char]
        depth = 0
        in_string = False
        escape_next = False

        for end in range(start, len(text)):
            c = text[end]

            if escape_next:
                escape_next = False
                continue

            if c == '\\' and in_string:
                escape_next = True
                continue

            if c == '"':
                in_string = not in_string
                continue

            if in_string:
                continue

            if c == char:
                depth += 1
            elif c == closer:
                depth -= 1
                if depth == 0:
                    candidate = text[start:end + 1]
                    try:
                        orjson.loads(candidate)
                        if best is None or len(candidate) > len(best):
                            best = candidate
                        break  # 解析成功：找到最外层有效闭合，终止内层扫描
                    except orjson.JSONDecodeError:
                        pass   # 解析失败：继续扫描，寻找更靠后的闭合位置
                elif depth < 0:
                    # depth 为负说明当前 start 对应的括号已无法配对，
                    # 继续扫描无意义，提前终止本轮内层循环
                    break

    return best


# ---------------------------------------------------------------------------
# dataclass 递归映射
# ---------------------------------------------------------------------------

def _map_to_dataclass(cls: type, data: dict) -> Any:
    """
    将 dict 递归映射到 dataclass，支持：
    - Optional[X]
    - List[X] / list[X]（X 为 dataclass 时递归）
    - 嵌套 dataclass
    - 优先使用 dacite（如已安装）
    """
    if _dacite is not None:
        try:
            return _dacite.from_dict(
                data_class=cls,
                data=data,
                config=_dacite.Config(strict=False),
            )
        except Exception as e:
            logger.debug("dacite 转换失败，回退到内置逻辑: %s", e)

    # 内置递归实现
    type_hints = get_type_hints(cls)
    converted: dict[str, Any] = {}

    for key, raw_value in data.items():
        if key not in type_hints:
            continue
        converted[key] = _coerce_value(raw_value, type_hints[key])

    return cls(**converted)


def _coerce_value(value: Any, field_type: Any) -> Any:
    """
    将 value 强制转换为 field_type，支持嵌套 dataclass / List / Optional。

    注：field_type 使用 Any 标注，因为泛型（Optional[X]、List[X]）
        不是 type 的实例，用 type 标注会引发类型检查告警。
    """
    # 解包 Optional[X] → X
    # 仅处理 Optional[X]（即 Union[X, None]）这一种单非None情况；
    # Union[str, int] 等多类型联合无法安全推断，保持原始 value 不做转换。
    origin = typing.get_origin(field_type)
    if origin is Union:
        args = [a for a in typing.get_args(field_type) if a is not type(None)]
        if len(args) == 1:
            field_type = args[0]
            origin = typing.get_origin(field_type)
        # len(args) > 1：Union[str, int] 等复合类型，跳过转换，直接返回原始 value

    # List[X] 递归处理列表元素
    if origin in (list, typing.List):
        item_args = typing.get_args(field_type)
        if item_args and isinstance(value, list):
            inner = item_args[0]
            if is_dataclass(inner):
                return [
                    _map_to_dataclass(inner, v) if isinstance(v, dict) else v
                    for v in value
                ]
        return value

    # 嵌套 dataclass
    if is_dataclass(field_type) and isinstance(value, dict):
        return _map_to_dataclass(field_type, value)

    return value


# ---------------------------------------------------------------------------
# 主工具类
# ---------------------------------------------------------------------------

class JsonUtil:

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------

    @staticmethod
    def to_str(data: Any, *, indent: bool = False, sort_keys: bool = False) -> str:
        """
        将 Python 对象序列化为 JSON 字符串。

        支持类型：dict / list / dataclass / datetime / date /
                  Decimal / UUID / Enum / set / bytes

        Args:
            data:       要序列化的对象
            indent:     是否格式化（美化）输出，默认 False
            sort_keys:  是否按 key 排序，默认 False

        Returns:
            JSON 字符串

        Examples:
            JsonUtil.to_str({"name": "test", "ts": datetime.now()})
            JsonUtil.to_str(my_dataclass, indent=True)
        """
        option = 0
        if indent:
            option |= orjson.OPT_INDENT_2
        if sort_keys:
            option |= orjson.OPT_SORT_KEYS

        return orjson.dumps(
            data,
            default=_default_serializer,
            option=option or None,
        ).decode("utf-8")

    @staticmethod
    def to_bytes(data: Any) -> bytes:
        """序列化为 bytes（比 to_str 少一次 decode，性能更好）。"""
        return orjson.dumps(data, default=_default_serializer)

    # ------------------------------------------------------------------
    # 反序列化
    # ------------------------------------------------------------------

    @staticmethod
    def to_obj(json_str: str | dict | list | Any, cls: Type[T] = None) -> Any:
        """
        将 JSON 字符串（或已解析的 dict/list）反序列化为 Python 对象。

        支持从包含 JSON 的"脏文本"中自动提取，例如 LLM 输出。
        支持目标类型：dataclass、Pydantic BaseModel，或返回原生 dict/list。

        Args:
            json_str: JSON 字符串、包含 JSON 的文本、或已经是 dict/list；
                      其他类型会先经 str() 强转并记录 warning 日志
            cls:      目标类型（可选）

        Returns:
            反序列化后的 Python 对象

        Examples:
            # 标准 JSON
            JsonUtil.to_obj('{"name": "test"}')

            # 含噪声的文本（LLM 输出场景）
            JsonUtil.to_obj('结果是：{"score": 99} 处理完毕')

            # 映射到 dataclass
            JsonUtil.to_obj('{"name": "Alice"}', cls=MyDataClass)

            # 映射到 Pydantic model
            JsonUtil.to_obj('{"name": "Alice"}', cls=MyModel)
        """
        if json_str is None:
            return None

        # 已经是结构化数据
        if isinstance(json_str, (dict, list)):
            obj = json_str
        else:
            if not isinstance(json_str, str):
                # 接受类型声明之外的输入，强转并记录日志，方便排查调用方问题
                logger.warning(
                    "to_obj 收到非预期类型 %s，已强转为 str，建议调用方修正传参",
                    type(json_str),
                )
            raw_json = _extract_json_from_text(str(json_str))
            obj = orjson.loads(raw_json)

        if cls is None:
            return obj

        # Pydantic BaseModel
        if _BaseModel is not None and isinstance(cls, type) and issubclass(cls, _BaseModel):
            if hasattr(cls, "model_validate"):      # pydantic v2
                return cls.model_validate(obj)
            if hasattr(cls, "parse_obj"):           # pydantic v1
                return cls.parse_obj(obj)
            return cls(**obj)

        # dataclass
        if is_dataclass(cls) and isinstance(obj, dict):
            return _map_to_dataclass(cls, obj)

        # 其他类型：尝试直接构造，失败则返回原始对象
        try:
            if isinstance(obj, dict):
                return cls(**obj)
            return cls(obj)
        except Exception as e:
            logger.warning("无法将 obj 转换为 %s: %s，返回原始对象", cls, e)
            return obj

    @staticmethod
    def extract(text: str) -> Any:
        """
        仅做 JSON 提取 + 解析，不做类型映射。
        适合只需要 dict/list 的场景，语义更清晰。

        Args:
            text: 包含 JSON 的任意字符串

        Returns:
            dict 或 list

        Raises:
            ValueError: 无法提取有效 JSON
        """
        raw = _extract_json_from_text(text)
        return orjson.loads(raw)