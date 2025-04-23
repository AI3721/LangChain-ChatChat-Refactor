import uuid
from typing import Dict, List
from chatchat.settings.settings import Settings
from chatchat.web.utils import process_files, AgentStatus, format_reference
from chatchat.web.openai_client import upload_chat_image, create_chat_completions
from chatchat.settings.settings_manager import get_config_platforms, get_config_models, get_default_llm, get_api_address

import streamlit as st
import streamlit_antd_components as sac
from streamlit_extras.bottom_container import bottom
from streamlit_chatbox import ChatBox, Markdown, Image

chat_box = ChatBox(assistant_avatar="chatchat/web/img/chatchat_icon.png")

def restore_session(conv_name: str = None):
    """恢复会话"""
    chat_box.context_to_session(
        chat_name=conv_name,
        exclude=["prompt", "selected_page", "cur_conv_name", "last_conv_name"],
    )

def save_session(conv_name: str = None):
    """保存会话"""
    chat_box.context_from_session(
        chat_name=conv_name,
        exclude=["prompt", "selected_page", "cur_conv_name", "last_conv_name"],
    )

def rerun_session():
    """重启，刷新会话"""
    save_session()
    st.rerun()

def create_session(conv_name: str = None):
    """新建会话"""
    conv_names = chat_box.get_chat_names()
    if not conv_name:
        i = len(conv_names)
        conv_name = f"新建会话{i}"
        while conv_name in conv_names:
            i = i + 1
            conv_name = f"新建会话{i}"
    
    if conv_name in conv_names:
        sac.alert(
            label="新键会话失败",
            description=f"“{conv_name}”会话已存在",
            color="error", closable=True)
    else:
        chat_box.use_chat_name(conv_name)
        st.session_state["cur_conv_name"] = conv_name

def delete_session(conv_name: str = None):
    """删除会话"""
    conv_names = chat_box.get_chat_names()
    conv_name = conv_name or chat_box.cur_chat_name
    if len(conv_names) == 1:
        sac.alert(
            label="删除会话失败", 
            description=f"至少保留一个会话",
            color="error", closable=True)
    elif conv_name not in conv_names:
        sac.alert(
            label="删除会话出失败",
            description=f"“{conv_name}”会话不存在",
            color="error", closable=True)
    else:
        chat_box.del_chat_name(conv_name)
    st.session_state["cur_conv_name"] = chat_box.cur_chat_name

def clear_session_history(conv_name: str = None):
    """清空会话历史"""
    chat_box.reset_history(conv_name)
    rerun_session()

def get_messages_history(history_len: int, content_in_expander: bool = False) -> List[Dict]:
    """
    获取历史消息
    content_in_expander: 是否返回expander中的内容
    一般在导出的时候可以选上, 传入LLM的history不需要
    """
    def filter(msg):
        content = [
            x for x in msg["elements"] 
            if x._output_method in ["markdown", "text"]]
        if not content_in_expander:
            content = [x for x in content if not x._in_expander]
        content = [x.content for x in content]
        return {"role": msg["role"], "content": "\n\n".join(content)}
    
    messages = chat_box.filter_history(history_len=history_len, filter=filter)
    if sys_msg := chat_box.context.get("system_message"):
        # := 海象运算符, 如果system_message不存在, 则sys_msg为None
        messages = [{"role": "system", "content": sys_msg}] + messages
    return messages

@st.dialog("重命名会话", width="small")
def rename_session():
    old_name = chat_box.cur_chat_name
    name = st.text_input("新的会话名", value=old_name)
    if st.button("OK"):
        names = chat_box.get_chat_names()
        if name == old_name:
            sac.alert(
                label="重命名失败",
                color="error", closable=True,
                description=f"与原会话名相同，请输入新的会话名！")
        elif name in names:
            sac.alert(
                label="重命名失败",
                color="error", closable=True,
                description=f"“{name}”会话已存在，请重新输入！")
        else:
            st.session_state["last_conv_name"] = name
            st.session_state["cur_conv_name"] = name
            chat_box.change_chat_name(name)
            restore_session()
            rerun_session()

@st.dialog("模型配置", width="large")
def model_setting():
    cols = st.columns(3)
    platforms = ["所有"] + list(get_config_platforms())
    platform = cols[0].selectbox("模型平台", platforms, key="platform")
    platform_name = None if platform == "所有" else platform # 默认所有
    llm_models = list(get_config_models(platform_name, model_type="llm"))
    llm_model = cols[1].selectbox("LLM模型", llm_models, key="llm_model")
    temperature = cols[2].slider("Temperature", 0.,1., key="temperature")
    system_message = st.text_area("System Message", key="system_message")
    if st.button("OK"): # 按下OK刷新会话
        rerun_session()

def dialogue_page():
    context = chat_box.context
    context.setdefault("uid", uuid.uuid4().hex)
    context.setdefault("llm_model", get_default_llm())
    context.setdefault("temperature",Settings.model_settings.TEMPERATURE)
    st.session_state.setdefault("cur_conv_name", chat_box.cur_chat_name)
    st.session_state.setdefault("last_conv_name", chat_box.cur_chat_name)

    if st.session_state.cur_conv_name != st.session_state.last_conv_name:
        # 确保用户在切换会话时保存上一个会话恢复当前会话
        save_session(st.session_state.last_conv_name)
        restore_session(st.session_state.cur_conv_name)
        st.session_state.last_conv_name = st.session_state.cur_conv_name
    
    with st.sidebar: # 侧边栏
        tab1, tab2 = st.tabs(["功能设置", "会话管理"])

        with tab1: # 功能设置
            agent_process = st.checkbox("显示Agent过程", key="agent_process")
            # tools = list_tools(api) # 工具列表
            tools={'arxiv': {'name': 'arxiv', 'title': 'ARXIV论文', 'description': 'A wrapper around Arxiv.org for searching and retrieving scientific articles in various fields.', 'args': {'query': {'title': 'Query', 'description': 'The search query title', 'type': 'string'}}, 'config': {'use': False}}, 'calculate': {'name': 'calculate', 'title': '数学计算器', 'description': ' Useful to answer questions about simple calculations. translate user question to a math expression that can be evaluated by numexpr. ', 'args': {'text': {'title': 'Text', 'description': 'a math expression', 'type': 'string'}}, 'config': {'use': False}}, 'search_internet': {'name': 'search_internet', 'title': '互联网搜索', 'description': 'Use this tool to use bing search engine to search the internet and get information.', 'args': {'query': {'title': 'Query', 'description': 'query for Internet search', 'type': 'string'}}, 'config': {'use': False, 'search_engine_name': 'duckduckgo', 'search_engine_config': {'bing': {'bing_search_url': 'https://api.bing.microsoft.com/v7.0/search', 'bing_key': ''}, 'metaphor': {'metaphor_api_key': '', 'split_result': False, 'chunk_size': 500, 'chunk_overlap': 0}, 'duckduckgo': {}, 'searx': {'host': 'https://metasearx.com', 'engines': [], 'categories': [], 'language': 'zh-CN'}}, 'top_k': 5, 'verbose': 'Origin', 'conclude_prompt': '<指令>这是搜索到的互联网信息，请你根据这些信息进行提取并有调理，简洁的回答问题。如果无法从中得到答案，请说 “无法搜索到能回答问题的内容”。 </指令>\n<已知信息>{{ context }}</已知信息>\n<问题>\n{{ question }}\n</问题>\n'}}, 'search_local_knowledgebase': {'name': 'search_local_knowledgebase', 'title': '本地知识库', 'description': "Use local knowledgebase from one or more of these: samples: 关于本项目issue的解答 三体: 关于三体的知识库 to get information，Only local data on this knowledge use this tool. The 'database' should be one of the above [samples 三体].", 'args': {'database': {'title': 'Database', 'description': 'Database for Knowledge Search', 'choices': ['samples', '三体'], 'type': 'string'}, 'query': {'title': 'Query', 'description': 'Query for Knowledge Search', 'type': 'string'}}, 'config': {'use': False, 'top_k': 3, 'score_threshold': 2.0, 'conclude_prompt': {'with_result': '<指令>根据已知信息，简洁和专业的来回答问题。如果无法从中得到答案，请说 "根据已知信息无法回答该问题"，不允许在答案中添加编造成分，答案请使用中文。 </指令>\n<已知信息>{{ context }}</已知信息>\n<问题>{{ question }}</问题>\n', 'without_result': '请你根据我的提问回答我的问题:\n{{ question }}\n请注意，你必须在回答结束后强调，你的回答是根据你的经验回答而不是参考资料回答的。\n'}}}, 'search_youtube': {'name': 'search_youtube', 'title': '油管视频', 'description': 'use this tools_factory to search youtube videos', 'args': {'query': {'title': 'Query', 'description': 'Query for Videos search', 'type': 'string'}}, 'config': {'use': False}}, 'shell': {'name': 'shell', 'title': '系统命令', 'description': 'Use Shell to execute system shell commands', 'args': {'query': {'title': 'Query', 'description': 'The command to execute', 'type': 'string'}}, 'config': {}}, 'text2images': {'name': 'text2images', 'title': '文生图', 'description': '根据用户的描述生成图片', 'args': {'prompt': {'title': 'Prompt', 'type': 'string'}, 'n': {'title': 'N', 'description': '需生成图片的数量', 'default': 1, 'type': 'integer'}, 'width': {'title': 'Width', 'description': '生成图片的宽度', 'default': 512, 'enum': [256, 512, 1024], 'type': 'integer'}, 'height': {'title': 'Height', 'description': '生成图片的高度', 'default': 512, 'enum': [256, 512, 1024], 'type': 'integer'}}, 'config': {'use': False, 'model': 'stable-diffusion-xl-base-1.0', 'size': '256*256'}}, 'text2sql': {'name': 'text2sql', 'title': '数据库对话', 'description': 'Use this tool to chat with  database,Input natural language, then it will convert it into SQL and execute it in the database, then return the execution result.', 'args': {'query': {'title': 'Query', 'description': 'No need for SQL statements,just input the natural language that you want to chat with database', 'type': 'string'}}, 'config': {'model_name': 'qwen-plus', 'use': False, 'sqlalchemy_connect_str': 'mysql+pymysql://用户名:密码@主机地址/数据库名称', 'read_only': False, 'top_k': 50, 'return_intermediate_steps': True, 'table_names': [], 'table_comments': {}}}, 'weather_check': {'name': 'weather_check', 'title': '天气查询', 'description': 'Use this tool to check the weather at a specific city', 'args': {'city': {'title': 'City', 'description': "City name,include city and county,like '厦门'", 'type': 'string'}}, 'config': {'use': False, 'api_key': ''}}, 'wolfram': {'name': 'wolfram', 'title': 'Wolfram', 'description': 'Useful for when you need to calculate difficult formulas', 'args': {'query': {'title': 'Query', 'description': 'The formula to be calculated', 'type': 'string'}}, 'config': {'use': False, 'appid': ''}}, 'amap_poi_search': {'name': 'amap_poi_search', 'title': '高德地图POI搜索', 'description': ' A wrapper that uses Amap to search.', 'args': {'location': {'title': 'Location', 'description': "'实际地名'或者'具体的地址',不能使用简称或者别称", 'type': 'string'}, 'types': {'title': 'Types', 'description': 'POI类型，比如商场、学校、医院等等', 'type': 'string'}}, 'config': {}}, 'amap_weather': {'name': 'amap_weather', 'title': '高德地图天气查询', 'description': 'A wrapper that uses Amap to get weather information.', 'args': {'city': {'title': 'City', 'description': '城市名', 'type': 'string'}}, 'config': {}}, 'wikipedia_search': {'name': 'wikipedia_search', 'title': '维基百科搜索', 'description': ' A wrapper that uses Wikipedia to search.', 'args': {'query': {'title': 'Query', 'description': 'The search query', 'type': 'string'}}, 'config': {}}, 'text2promql': {'name': 'text2promql', 'title': 'Prometheus对话', 'description': 'Use this tool to chat with prometheus, Input natural language, then it will convert it into PromQL and execute it in the prometheus, then return the execution result.', 'args': {'query': {'title': 'Query', 'description': 'Tool for querying a Prometheus server, No need for PromQL statements, just input the natural language that you want to chat with prometheus', 'type': 'string'}}, 'config': {'use': False, 'prometheus_endpoint': 'http://127.0.0.1:9090', 'username': '', 'password': ''}}, 'url_reader': {'name': 'url_reader', 'title': 'URL内容阅读', 'description': 'Use this tool to get the clear content of a URL.', 'args': {'url': {'title': 'Url', 'description': 'The URL to be processed, so that its web content can be made more clear to read. Then provide a detailed description of the content in about 500 words. As structured as possible. ONLY THE LINK SHOULD BE PASSED IN.', 'type': 'string'}}, 'config': {'use': False, 'timeout': '10000'}}}
            selected_tools = st.multiselect(
                label="选择工具",
                options=list(tools),
                format_func=lambda x: tools[x]["title"],
                placeholder="(可选)",
                key="selected_tools",
            )

            selected_tools_args = {}
            for tool_name in selected_tools:
                selected_tool = tools[tool_name]
                selected_tools_args[tool_name] = {}
                with st.expander(f"{selected_tool['title']}", False):
                    for k, v in selected_tool["args"].items():
                        key = f"{tool_name}_{k}" # 确保每个组件的key唯一
                        if choices := v.get("choices", v.get("enum")):
                            selected_tools_args[tool_name][k] = st.selectbox(v["title"], options=choices, index=0, key=key)
                        elif v["type"] == "integer":
                            selected_tools_args[tool_name][k] = st.slider(v["title"], value=v.get("default"), step=1, key=key)
                        elif v["type"] == "number":
                            selected_tools_args[tool_name][k] = st.slider(v["title"], value=v.get("default"), step=0.1, key=key)
                        elif v["type"] == "string":
                            selected_tools_args[tool_name][k] = st.text_input(v["title"], value=v.get("default"), key=key)
            
            uploaded_files = st.file_uploader("上传附件", accept_multiple_files=True)
            processed_files = process_files(files=uploaded_files)
        
        with tab2: # 会话管理
            cols = st.columns(3)
            selected_conv_name = sac.buttons(
                items=chat_box.get_chat_names(),
                key="cur_conv_name",
                label="当前会话",
            )
            chat_box.use_chat_name(selected_conv_name)
            if cols[0].button("新建", on_click=create_session):
                st.success("新建成功！", icon="✅")
            if cols[1].button("重命名"):
                rename_session() # 创建一个小会话框
            if cols[2].button("删除", on_click=delete_session):
                st.success("删除成功！", icon="✅")
    
    chat_box.output_messages() # 聊天框输出历史消息

    with bottom(): # 底部输入框
        cols = st.columns([1, 0.3, 15, 1])
        if cols[0].button(":gear:", help="模型配置"):
            chat_box.context_to_session(
                include=["llm_model", "temperature"])
            model_setting() # 模型配置之前先获取默认配置
        if cols[-1].button(":wastebasket:", help="清空对话"):
            clear_session_history()
        prompt = cols[2].chat_input(placeholder="请输入对话内容, 换行请使用Shift+Enter.", key="prompt")
    
    if prompt:
        llm_model = context.get("llm_model")
        temperature = context.get("temperature")
        llm_model_config = Settings.model_settings.LLM_MODEL_CONFIG
        chat_model_config = {key: {} for key in llm_model_config.keys()}
        for key in llm_model_config:
            if c := llm_model_config[key]:
                model = c.get("model") or llm_model
                chat_model_config[key][model] = llm_model_config[key]
                if key == "llm_model":
                    chat_model_config[key][model]["temperature"] = temperature
                    history_len = chat_model_config[key][model].get("history_len", 10)
        
        history = get_messages_history(history_len=history_len) # 获取历史消息

        image_info = None
        if processed_files["images"] and llm_model in get_config_models(model_type="image2text") and not selected_tools:
            image_info = upload_chat_image(processed_files["images"][0][0], processed_files["images"][0][1])
            image_url = f'{get_api_address()}/v1/files/{image_info.get("id")}/content'
            chat_box.user_say([Image(image_url, use_column_width=None), Markdown(prompt)])
            messages = history + [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}}]}]
        else:
            chat_box.user_say(prompt)
            messages = history + [{"role": "user", "content": prompt}]
        
        chat_box.ai_say("正在思考...")
        started = False

        extra_body = dict(
            image_info=image_info,
            chat_model_config=chat_model_config,
            selected_tools_args=selected_tools_args,
            conversation_id=chat_box.context.get("uid"),
        )
        params = dict(
            stream=True,
            model=llm_model,
            messages=messages,
            extra_body=extra_body,
        )
        if selected_tools:
            params["tools"] = list(selected_tools)
        if Settings.model_settings.MAX_TOKENS:
            params["max_tokens"] = Settings.model_settings.MAX_TOKENS
        
        try:
            for d in create_chat_completions(params):
                metadata = {"message_id": d.message_id}
                if not started:
                    chat_box.update_msg("", streaming=False)
                    started = True
                if d.status == AgentStatus.error:
                    st.error(d.choices[0].delta.content)
                elif d.status == AgentStatus.llm_start:
                    if not agent_process:
                        continue
                    chat_box.insert_msg("正在解读工具输出结果...")
                    text = d.choices[0].delta.content or ""
                elif d.status == AgentStatus.llm_new_token:
                    if not agent_process:
                        continue
                    text += d.choices[0].delta.content or ""
                    chat_box.update_msg(text.replace("\n", "\n\n"), streaming=True, metadata=metadata)
                elif d.status == AgentStatus.llm_end:
                    if not agent_process:
                        continue
                    text += d.choices[0].delta.content or ""
                    chat_box.update_msg(text.replace("\n", "\n\n"), streaming=False, metadata=metadata)
                elif d.status == AgentStatus.agent_finish:
                    text = d.choices[0].delta.content or ""
                    chat_box.update_msg(text.replace("\n", "\n\n"))
                elif d.status is None:
                    if getattr(d, "is_ref", False):
                        context = str(d.tool_output)
                        if isinstance(d.tool_output, dict):
                            source_documents = format_reference(
                                docs=d.tool_output.get("docs", []),
                                kb_name=d.tool_output.get("knowledge_base"),
                                api_base_url=get_api_address(is_public=True))
                            context = "\n".join(source_documents)
                        chat_box.insert_msg(
                            Markdown(context, title="参考资料", in_expander=True, state="complete"))
                        chat_box.insert_msg("")
                    elif getattr(d, "tool_call", None) == "text2images":
                        for img in d.tool_output.get("images", []):
                            chat_box.insert_msg(Image(f"{get_api_address()}/media/{img}"), pos=-2)
                    else:
                        text += d.choices[0].delta.content or ""
                        chat_box.update_msg(text.replace("\n", "\n\n"), streaming=True, metadata=metadata)
                chat_box.update_msg(text, streaming=False, metadata=metadata)
        except Exception as e:
            st.error(e)

