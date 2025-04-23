import openai
import streamlit as st
from chatchat.settings.settings_manager import get_api_address

api_address = get_api_address()

@st.cache_data
def upload_chat_image(file_name: str, file_data: bytes) -> dict:
    """上传聊天图片，用于多模态大模型"""
    client = openai.Client(
        base_url=f"{api_address}/v1", api_key="NONE")
    return client.files.create(
        file=(file_name, file_data), 
        purpose="assistants").to_dict()

def create_chat_completions(params):
    client = openai.Client(
         base_url=f"{api_address}/chat", api_key="NONE")
    return client.chat.completions.create(**params)