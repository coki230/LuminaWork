from datetime import datetime
import os
import traceback

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from starlette.middleware.cors import CORSMiddleware
from agent_logic import app_agent
from typing import List
import uuid
import shutil
import re

app = FastAPI()

# 允许跨域（方便前端测试）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
UPLOAD_DIR = "temp_files"

# 1. 确保目录存在
if not os.path.exists("static"):
    os.makedirs("static")

# 2. 挂载静态文件目录 (这样你可以通过 /static/文件名 访问其他资源)
app.mount("/static", StaticFiles(directory="static"), name="static")

# 3. 根目录直接返回 HTML 页面
@app.get("/")
async def read_index():
    # 假设你的 HTML 文件名是 static/index.html
    return FileResponse('static/main.html')

@app.post("/process")
async def process_request(
        files: List[UploadFile] = File(...),
        prompt: str = Form(...)
):
    job_id = str(uuid.uuid4())
    saved_paths = []

    try:
        # 1. 保存上传的文件到临时目录
        for file in files:
            file_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_paths.append(file_path)

        # 2. 调用 LangGraph Agent
        # 输入状态包含：消息列表 和 上传的文件路径列表
        inputs = {
            "messages": [("user", prompt)],
            "file_list": saved_paths,
            "latest_file": ""
        }

        # 运行 Agent
        final_state = await app_agent.ainvoke(inputs)

        # 3. 获取 Agent 的最后一条回复内容
        # 假设 Agent 已经通过工具生成了文件，并返回了消息
        ai_response = final_state["messages"][-1].content

        download_url = final_state.get("latest_file", "")

        return {
            "status": "success",
            "message": ai_response,
            "job_id": job_id,
            "download_url": download_url,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }

    except Exception as e:
        print("---------- DETAILED ERROR START ----------")
        traceback.print_exc()
        print("---------- DETAILED ERROR END ----------")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download/{file_name}")
async def download_file(file_name: str):
    file_path = os.path.join(UPLOAD_DIR, file_name)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return {"error": "文件不存在"}

if __name__ == "__main__":
    import uvicorn
    # 启动后访问 http://127.0.0.1:8000 即可看到页面
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False, workers=1)