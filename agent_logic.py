import json
import re
from typing import Annotated, TypedDict, List

from langchain_core.messages import ToolMessage
from langchain_ollama import ChatOllama
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END
import os
from PIL import Image
import pandas as pd


# 1. 定义工具 (Tools)
@tool
def convert_images_to_pdf(image_paths: List[str], output_filename: str):
    """当用户想要将多张图片、照片合并或转换为一个 PDF 文件时调用此工具。"""
    images = [Image.open(path).convert("RGB") for path in image_paths]
    if not output_filename.endswith('.pdf'):
        output_filename += '.pdf'
    output_path = f"temp_files/{output_filename}.pdf"
    images[0].save(output_path, save_all=True, append_images=images[1:])
    return {
        "messages": [f"已成功生成 PDF：{output_path}"],
        "latest_file": output_path
    }

@tool
def modify_excel_data(file_path: str, operation_desc: str):
    """当用户想要修改、统计或处理 Excel 表格数据时调用此工具。"""
    output, output_path = handle_excel_with_agent(file_path, operation_desc)
    return {
        "messages": [f"已成功生成处理文件，结果为：{output}"],
        "latest_file": output_path
    }

tools = [convert_images_to_pdf, modify_excel_data]
tool_node = ToolNode(tools)

# 2. 设置状态与模型
class AgentState(TypedDict):
    messages: Annotated[List, "messages"]
    file_list: List[str]  # 存储上传的文件路径
    latest_file: str

# model = ChatOpenAI(model="gpt-4-turbo").bind_tools(tools)
model = ChatOllama(
    model="qwen3.5:9b",
    temperature=0,
    # format="json" # 有些模型需要强制 JSON 格式，但在 bind_tools 下通常不需要
).bind_tools(tools)

def handle_excel_with_agent(file_path: str, user_query: str):
    # 1. 加载数据
    df = None
    output_path = f"temp_files/processed_{os.path.basename(file_path)}"
    if not file_path.lower().endswith(('.xlsx', '.xls', '.csv')):
        return '输入文件格式有问题', output_path

    if file_path.lower().endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_path)
    elif file_path.lower().endswith('.csv'):
        df = pd.read_csv(file_path)

    prefix = """
    你是一个资深 A 股量化数据分析师。
    注意：
    1. **必须原位修改**：所有的筛选或处理操作，必须赋值回变量 `df`。例如：`df = df[df['low'] > 6.6]`。
    2. 你可以直接使用 pd.to_numeric 等函数，因为 pandas 已预先导入
    """

    # 3. 创建 Pandas Agent
    # allow_dangerous_code=True 是必须的，因为 Agent 需要运行生成的 Python 代码
    agent = create_pandas_dataframe_agent(
        model,
        df,
        prefix=prefix,  # 注入专业指令
        verbose=True,
        allow_dangerous_code=True,
        agent_type="tool-calling",
        include_df_in_prompt=True,  # 让 AI 明确看到列名，减少尝试
        max_iterations=9,
        max_execution_time=30   # 强制单次任务最长 30 秒
    )

    # 4. 执行并获取结果
    # 它会自动理解“合计”、“加100”、“最大值”等语义
    response = agent.invoke(user_query)
    print("response:", response)

    # 从 Agent 的 tool 状态中提取修改后的 df
    # 实际上，create_pandas_dataframe_agent 的内部 tools[0] 就是那个 python_repl
    # 我们可以通过以下方式获取它执行后的 locals 变量表
    for tool_item in agent.tools:
        if hasattr(tool_item, "globals") and "df" in tool_item.globals:
            df = tool_item.globals["df"]
        elif hasattr(tool_item, "locals") and "df" in tool_item.locals:
            df = tool_item.locals["df"]

    # 5. 保存结果（如果是修改操作）
    # 注意：Agent 默认在内存中操作 df，我们需要保存
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.csv':
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
    else:
        df.to_excel(file_path, index=False)

    return response["output"], output_path

# 3. 定义逻辑流
def router(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "execute_tool"
    return END

def call_model(state: AgentState):
    # 将文件列表上下文注入到提示词中
    prompt = f"当前用户上传的文件有: {state['file_list']}。请根据用户要求调用工具。"
    response = model.invoke([{"role": "system", "content": prompt}] + state["messages"])
    latest_file = state.get("latest_file", "")
    for msg in reversed(state["messages"]):
        # 只要是工具消息，就尝试解析
        if isinstance(msg, ToolMessage):
            content = msg.content
            try:
                # 尝试解析 JSON 字符串
                data = json.loads(content)
                if isinstance(data, dict) and "latest_file" in data:
                    latest_file = data["latest_file"]
                    break
            except (json.JSONDecodeError, TypeError):
                # 如果不是 JSON 字符串（比如是纯文本回复），尝试正则兜底
                match = re.search(r'temp_files/[\w\.-]+', str(content))
                if match:
                    latest_file = match.group(0)
                    break
    return {
        "messages": [response],
        "latest_file": latest_file
    }

# 4. 构建工作流图
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("execute_tool", tool_node)

workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", router, {"execute_tool": "execute_tool", END: END})
workflow.add_edge("execute_tool", "agent")

app_agent = workflow.compile()