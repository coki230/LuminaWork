from typing import Annotated, TypedDict, List

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END
import os
from PIL import Image

# 1. 定义工具 (Tools)
@tool
def convert_images_to_pdf(image_paths: List[str], output_filename: str):
    """当用户想要将多张图片、照片合并或转换为一个 PDF 文件时调用此工具。"""
    images = [Image.open(path).convert("RGB") for path in image_paths]
    if not output_filename.endswith('.pdf'):
        output_filename += '.pdf'
    output_path = f"temp_files/{output_filename}.pdf"
    images[0].save(output_path, save_all=True, append_images=images[1:])
    return f"成功生成 PDF，文件路径为: {output_path}"

@tool
def modify_excel_data(file_path: str, operation_desc: str):
    """当用户想要修改、统计或处理 Excel 表格数据时调用此工具。"""
    # 这里可以接入更复杂的 pandas 逻辑
    return f"已根据指令 '{operation_desc}' 处理 Excel: {file_path}"

tools = [convert_images_to_pdf, modify_excel_data]
tool_node = ToolNode(tools)

# 2. 设置状态与模型
class AgentState(TypedDict):
    messages: Annotated[List, "messages"]
    file_list: List[str]  # 存储上传的文件路径

# model = ChatOpenAI(model="gpt-4-turbo").bind_tools(tools)
model = ChatOllama(
    model="qwen3.5:9b",
    temperature=0,
    # format="json" # 有些模型需要强制 JSON 格式，但在 bind_tools 下通常不需要
).bind_tools(tools)

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
    return {"messages": [response]}

# 4. 构建工作流图
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("execute_tool", tool_node)

workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", router, {"execute_tool": "execute_tool", END: END})
workflow.add_edge("execute_tool", "agent")

app_agent = workflow.compile()