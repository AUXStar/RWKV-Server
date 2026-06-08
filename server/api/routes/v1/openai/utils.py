import re
import json
import time
import uuid
from typing import List, Union, Optional, Tuple

from .schemas import ChatMessage, Usage, ToolCall, FunctionCall
from .....task.task import Task


def generate_chat_id() -> str:
    """生成符合OpenAI规范的聊天补全ID"""
    return f"chatcmpl-{uuid.uuid4().hex}"


def generate_completion_id() -> str:
    """生成符合OpenAI规范的文本补全ID"""
    return f"cmpl-{uuid.uuid4().hex}"


def normalize_stop_sequences(stop: Union[str, List[str], None]) -> List[str]:
    """标准化停止序列为列表格式"""
    if stop is None:
        return []
    if isinstance(stop, str):
        return [stop]
    return stop


def calculate_usage(task: Task, prompt: str) -> Usage:
    """计算聊天补全的Token使用量"""
    prompt_tokens = len(task.tokenize(prompt))
    completion_tokens = len(task._generated_tokens)
    
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


def calculate_completion_usage(task: Task, prompt: str, echo: bool = False) -> Usage:
    """计算文本补全的Token使用量"""
    prompt_tokens = len(task.tokenize(prompt))
    completion_tokens = len(task._generated_tokens)

    if echo:
        completion_tokens += prompt_tokens

    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


def parse_tool_call(text: str) -> Tuple[Optional[str], Optional[List[ToolCall]]]:
    """
    解析RWKV7-G1原生的<tool_call>格式，转换为OpenAI标准工具调用格式
    
    Args:
        text: 模型生成的原始文本
        
    Returns:
        Tuple[纯文本内容, 工具调用列表]
        如果没有工具调用，工具调用列表为None
    """
    # 匹配<tool_call>标签（支持跨多行）
    tool_call_pattern = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)
    matches = tool_call_pattern.findall(text)
    
    if not matches:
        return text.strip() if text.strip() else None, None
    
    # 提取标签之外的纯文本内容
    plain_text = tool_call_pattern.sub('', text).strip()
    
    tool_calls = []
    for match in matches:
        try:
            # 解析JSON内容
            tool_data = json.loads(match.strip())
            
            # 处理单个工具调用
            if isinstance(tool_data, dict):
                tool_calls.append(
                    ToolCall(
                        id=f"call_{uuid.uuid4().hex[:9]}",
                        function=FunctionCall(
                            name=tool_data.get("name", ""),
                            arguments=json.dumps(tool_data.get("arguments", {}), ensure_ascii=False)
                        )
                    )
                )
            # 处理多个工具调用
            elif isinstance(tool_data, list):
                for item in tool_data:
                    if isinstance(item, dict):
                        tool_calls.append(
                            ToolCall(
                                id=f"call_{uuid.uuid4().hex[:9]}",
                                function=FunctionCall(
                                    name=item.get("name", ""),
                                    arguments=json.dumps(item.get("arguments", {}), ensure_ascii=False)
                                )
                            )
                        )
        except json.JSONDecodeError:
            # JSON解析失败时，将整个内容作为纯文本返回
            return text, None
    print("plain_text:")
    print(plain_text)
    print("tool_calls:")
    print(tool_calls)
    print("end")
    return plain_text if plain_text else None, tool_calls


def messages_to_prompt(messages: List[ChatMessage]) -> str:
    """
    将OpenAI messages格式转换为RWKV7-G1官方标准Prompt格式
    支持字符串和多模态列表格式的content
    默认启用官方推荐的快思考模式，性能和质量最佳
    
    官方快思考模式格式：Assistant: \n</think
    """
    prompt = ""
    system_prompt = ""

    for message in messages:
        # 处理content字段（支持字符串和多模态列表）
        content = ""
        if isinstance(message.content, str):
            content = message.content
        elif isinstance(message.content, list):
            text_parts = []
            for part in message.content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
                elif hasattr(part, "text"):
                    text_parts.append(part.text)
            content = "".join(text_parts)
        elif message.content is None:
            content = ""

        if message.role == "system":
            system_prompt = content
        elif message.role == "user":
            prompt += f"User: {content}\n"
        elif message.role == "assistant":
            prompt += f"Assistant: {content}\n"
        elif message.role == "tool":
            # 处理工具返回结果
            prompt += f"Tool: {content}\n"

    # 系统提示词放在最前面
    if system_prompt:
        prompt = f"{system_prompt}\n\n{prompt}"

    # 添加官方标准快思考模式标签（精确格式，不能有任何改动）
    prompt += "Assistant: <think>\n</think>"
    
    return prompt