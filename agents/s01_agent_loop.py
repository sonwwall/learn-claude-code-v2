#!/usr/bin/env python3
# 这是最小可用的 agent loop 示例：
# 用户输入 -> 调用模型 -> 如果模型请求工具，则执行工具 ->
# 把工具结果重新喂给模型 -> 继续下一轮，直到模型给出最终自然语言回复。
"""
s01_agent_loop.py - The Agent Loop

This file teaches the smallest useful coding-agent pattern:

    user message
      -> model reply
      -> if tool_use: execute tools
      -> write tool_result back to messages
      -> continue

It intentionally keeps the loop small, but still makes the loop state explicit
so later chapters can grow from the same structure.
"""

import os
import subprocess
from dataclasses import dataclass

try:
    import readline
    # macOS 上默认的 libedit 对 UTF-8 输入和退格支持不稳定，
    # 这里通过 readline 参数做最小修正，避免中文或特殊字符输入时体验异常。
    readline.parse_and_bind('set bind-tty-special-chars off')
    readline.parse_and_bind('set input-meta on')
    readline.parse_and_bind('set output-meta on')
    readline.parse_and_bind('set convert-meta off')
    readline.parse_and_bind('set enable-meta-keybindings on')
except ImportError:
    pass

from anthropic import Anthropic
from dotenv import load_dotenv

# 从当前工作目录加载 .env，override=True 表示环境变量以 .env 中的值为准。
load_dotenv(override=True)

# 如果用户配置了兼容 Anthropic 的代理 / 网关地址，
# 则移除可能冲突的认证 token，避免底层 SDK 同时读取两套鉴权配置。
if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

# 初始化 Anthropic 客户端。
# 如果设置了 ANTHROPIC_BASE_URL，就把请求发到指定端点；否则使用官方默认地址。
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
# 模型名从环境变量读取，便于在 .env 中切换不同模型而不改代码。
MODEL = os.environ["MODEL_ID"]

# 系统提示词定义了 agent 的基本身份和工作方式。
# 这里明确告知模型：
# 1. 当前工作目录是什么
# 2. 可以用 bash 查看和修改工作区
# 3. 优先行动，再汇报结果
SYSTEM = (
    f"You are a coding agent at {os.getcwd()}. "
    "Use bash to inspect and change the workspace. Act first, then report clearly."
)

# 这里声明给模型使用的工具列表。
# s01 只暴露一个最简单的工具：bash。
# 模型如果想执行命令，需要返回一个 tool_use block，其输入必须符合这里的 JSON schema。
TOOLS = [{
    "name": "bash",
    "description": "Run a shell command in the current workspace.",
    "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
}]


@dataclass
class LoopState:
    # messages:
    #   完整对话历史。这里既包含用户消息，也包含 assistant 的回复，
    #   还会包含“以 user 角色回传给模型的 tool_result”。
    #
    # turn_count:
    #   记录当前 agent loop 跑了多少轮。s01 中主要起到教学和可观察性作用，
    #   后续章节可以基于它做限流、调试、日志分析等。
    #
    # transition_reason:
    #   记录本轮为什么继续循环。例如 tool_result 表示：
    #   上一轮模型请求了工具，工具已经执行，需要把结果送回模型继续推理。
    messages: list
    turn_count: int = 1
    transition_reason: str | None = None


def run_bash(command: str) -> str:
    # 非严格安全策略：只做最小危险命令拦截。
    # 这里是教学示例，不是完整沙箱；目的是避免明显破坏性的命令直接运行。
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(item in command for item in dangerous):
        return "Error: Dangerous command blocked"
    try:
        # shell=True 让模型可以直接运行普通 shell 命令字符串。
        # cwd 固定为当前工作目录，确保命令在项目根目录上下文里执行。
        # capture_output=True 收集 stdout/stderr，后续作为 tool_result 返回给模型。
        # timeout 防止命令无限卡住，阻塞整个 agent loop。
        result = subprocess.run(
            command,
            shell=True,
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"

    # 把标准输出和标准错误拼接起来，方便模型拿到完整上下文。
    # strip() 去掉首尾空白，减少无意义噪声。
    output = (result.stdout + result.stderr).strip()
    # 对输出长度做上限裁剪，避免一次命令返回过多内容，
    # 导致后续发给模型的消息过大。
    return output[:50000] if output else "(no output)"


def extract_text(content) -> str:
    # Anthropic 的 content 通常是 block 列表，其中可能混合 text、tool_use 等多种块。
    # 这个函数只提取带 text 属性的块，并把它们拼成最终展示给用户的字符串。
    if not isinstance(content, list):
        return ""
    texts = []
    for block in content:
        # getattr 的写法更稳妥：即使某个 block 没有 text 字段，也不会直接抛异常。
        text = getattr(block, "text", None)
        if text:
            texts.append(text)
    return "\n".join(texts).strip()


def execute_tool_calls(response_content) -> list[dict]:
    # 遍历模型回复中的所有 block，执行其中的 tool_use。
    # 返回值是 tool_result 列表，稍后会被重新追加到消息历史中。
    results = []
    for block in response_content:
        # 只处理工具调用块，普通文本块在这里忽略。
        if block.type != "tool_use":
            continue
        # 根据当前工具 schema，bash 的输入格式是 {"command": "..."}。
        command = block.input["command"]
        # 用黄色把实际执行的命令打印到终端，方便人类观察 agent 在做什么。
        print(f"\033[33m$ {command}\033[0m")
        output = run_bash(command)
        # 只预览前 200 个字符，避免终端被大段输出刷屏。
        print(output[:200])
        # tool_use_id 必须和模型发来的 tool_use block.id 对应，
        # 这样模型才能把这个结果关联回那次工具调用。
        results.append({
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": output,
        })
    return results


def run_one_turn(state: LoopState) -> bool:
    # 发起一次模型调用。这里把当前所有消息历史、系统提示词、工具定义一并传入。
    response = client.messages.create(
        model=MODEL,
        system=SYSTEM,
        messages=state.messages,
        tools=TOOLS,
        max_tokens=8000,
    )
    # 无论模型这次返回的是纯文本还是工具调用，都先原样记录进历史。
    # 这是对话状态持续演进的核心。
    state.messages.append({"role": "assistant", "content": response.content})

    # 如果 stop_reason 不是 tool_use，说明模型本轮没有要求继续执行工具，
    # 一般意味着它已经给出了最终答复，循环到此结束。
    if response.stop_reason != "tool_use":
        state.transition_reason = None
        return False

    # 执行模型请求的所有工具调用。
    results = execute_tool_calls(response.content)
    # 极端情况下，模型声称要 tool_use，但没有可执行结果，这里安全退出。
    if not results:
        state.transition_reason = None
        return False

    # 关键点：
    # 工具结果要以“user 消息”的形式送回模型。
    # 在 Anthropic 的消息协议里，tool_result 属于用户侧继续提供的新上下文。
    state.messages.append({"role": "user", "content": results})
    state.turn_count += 1
    state.transition_reason = "tool_result"
    # 返回 True 表示循环继续，模型还需要基于工具结果再思考一轮。
    return True


def agent_loop(state: LoopState) -> None:
    # 只要 run_one_turn 返回 True，就持续推进 agent loop。
    # 也就是：模型调用 -> 工具执行 -> 工具结果回传 -> 再调用模型。
    while run_one_turn(state):
        pass


if __name__ == "__main__":
    # history 是跨多次用户输入保留的会话历史。
    # 这意味着你在命令行里连续问多个问题时，模型会记住上文。
    history = []
    while True:
        try:
            # 青色提示符，表示当前是 s01 示例。
            query = input("\033[36ms01 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        # 输入 q / exit / 空字符串时退出交互。
        if query.strip().lower() in ("q", "exit", ""):
            break

        # 把本轮用户输入追加到共享历史中。
        history.append({"role": "user", "content": query})
        # 用当前完整历史初始化一次新的 loop state。
        # 注意 messages 直接引用 history，因此 loop 中的追加会回写到 history。
        state = LoopState(messages=history)
        agent_loop(state)

        # agent_loop 结束后，history[-1] 通常是最后一条 assistant 消息。
        # 这里把其中的纯文本部分提取出来打印给终端用户。
        final_text = extract_text(history[-1]["content"])
        if final_text:
            print(final_text)
        print()
