import streamlit as st
from typing import List, Dict
from urllib.parse import urlencode

def process_files(files):
    """处理文件, 转换图片, 音频和视频为base64格式"""
    result = {"images": [], "audios": [], "videos": []}
    for file in files:
        file_name = file.name
        file_data = file.read()
        file_type = file.name.split(".")[-1]
        if file_type in ["png", "jpg", "jpeg"]:
            st.image(file_data)
            result["images"].append((file_name, file_data))
        elif file_type in ["mp3", "wav"]:
            st.audio(file_data)
            result["audios"].append((file_name, file_data))
        elif file_type in ["mp4", "avi"]:
            st.video(file_data)
            result["videos"].append((file_name, file_data))
    return result

class AgentStatus:
    llm_start: int = 1
    llm_new_token: int = 2
    llm_end: int = 3
    agent_action: int = 4
    agent_finish: int = 5
    tool_start: int = 6
    tool_end: int = 7
    error: int = 8

def format_reference(kb_name: str, docs: List[Dict], api_base_url: str="") -> List[Dict]:
    """将知识库检索结果格式化为参考文档的格式"""
    source_documents = []
    for inum, doc in enumerate(docs):
        filename = doc.get("metadata", {}).get("source")
        parameters = urlencode({"knowledge_base_name": kb_name, "file_name": filename})
        url = f"{api_base_url}/knowledge_base/download_doc?" + parameters
        ref = f"""出处 [{inum + 1}] [{filename}]({url}) \n\n{doc.get("page_content")}\n\n"""
        source_documents.append(ref)
    return source_documents