import streamlit as st
import streamlit_antd_components as sac

from chatchat import __version__
from chatchat.web.dialogue import dialogue_page

if __name__ == "__main__":

    st.set_page_config(
        page_title="LangChain-ChatChat WebUI",
        page_icon="chatchat/web/img/chatchat_icon.png",
        menu_items={
            "About": f"欢迎使用LangChain-ChatChat WebUI {__version__}!",
            "Get help": "https://github.com/AI3721/LangChain-ChatChat-Refactor",
            "Report a bug": "https://github.com/AI3721/LangChain-ChatChat-Refactor/issues",
        }
    )
    
    with st.sidebar:
        st.logo(
            size="large",
            image="chatchat/web/img/chatchat_long.png",
            icon_image="chatchat/web/img/chatchat_short.png")
        selected_page = sac.menu(
            items=[
                sac.MenuItem(label="多功能对话", icon="chat-dots"),
                sac.MenuItem(label="RAG 对话", icon="database"),
                sac.MenuItem(label="知识库管理", icon="hdd-stack"),
            ]
        )
        sac.divider() # 水平分割线
    
    if selected_page == "多功能对话":
        dialogue_page()
        pass
    elif selected_page == "RAG 对话":
        # rag_page()
        pass
    elif selected_page == "知识库管理":
        # kb_page()
        pass