import requests
from urllib.parse import urlparse
from chatchat.utils import build_logger
from chatchat.settings.settings import Settings
from typing import Dict, List, Literal, Optional
from memoization import cached, CachingAlgorithmFlag

logger = build_logger() # 创建日志记录器

def get_config_platforms() -> Dict[str, Dict]:
    """获取配置的模型平台, 转换为字典"""
    platforms = [p.model_dump() for p in Settings.model_settings.MODEL_PLATFORMS]
    return {p["platform_name"]: p for p in platforms}

def get_base_url(url):
    """获取 api url 的根地址"""
    parsed_url = urlparse(url) # 解析url
    return f"{parsed_url.scheme}://{parsed_url.netloc}"

@cached(max_size=10, ttl=60, algorithm=CachingAlgorithmFlag.LRU)
def detect_xf_models(xf_url: str) -> Dict[str, List[str]]:
    """检测xinference模型列表, 并缓存结果"""
    xf_model_type_maps = {
        "llm_models": lambda xf_models: [ # 模型类型为LLM, 且不具备视图能力
            k for k, v in xf_models.items() if v["model_type"] == "LLM"],
        "embed_models": lambda xf_models: [ # 模型类型为embedding
            k for k, v in xf_models.items() if v["model_type"] == "embedding"],
        "rerank_models": lambda xf_models: [ # 模型类型为rerank
            k for k, v in xf_models.items() if v["model_type"] == "rerank" ],
        "image2text_models": lambda xf_models: [ # 模型类型为LLM, 且具备视图能力
            k for k, v in xf_models.items() if v["model_type"] == "LLM" and "vision" in v["model_ability"]],
        "text2image_models": lambda xf_models: [ # 模型类型为image
            k for k, v in xf_models.items() if v["model_type"] == "image"],
        "image2image_models": lambda xf_models: [ # 模型类型为image
            k for k, v in xf_models.items() if v["model_type"] == "image"],
        "speech2text_models": lambda xf_models: [ # 模型家族为whisper
            k for k, v in xf_models.items() if v.get("model_family") in ["whisper"]],
        "text2speech_models": lambda xf_models: [ # 模型家族为ChatTTS
            k for k, v in xf_models.items() if v.get("model_family") in ["ChatTTS"]],
    }
    models = {}
    try:
        from xinference_client import RESTfulClient
        xf_client = RESTfulClient(xf_url)
        xf_models = xf_client.list_models()
        for model_type, model_filter in xf_model_type_maps.items():
            models[model_type] = model_filter(xf_models)
    except ImportError: # 引用错误
        logger.warning("自动检测模型需要下载xinference-client, 可以尝试'pip install xinference-client'")
    except requests.exceptions.ConnectionError:
        logger.warning(f"无法连接xinference接口: {xf_url}, 请检查你的配置")
    except Exception as e: # 其他错误
        logger.warning(f"xinference服务连接出错: {e}")
    return models

def get_config_models(
        platform_name: str = None,
        model_type: Optional[Literal["llm", "embed", "text2image", "image2image", "image2text", "rerank", "speech2text", "text2speech"]] = None,
        model_name: str = None,
    ) -> Dict[str, Dict]:
    """
    获取配置的模型列表，返回值为:
    {model_name: {
        "platform_type": xx,
        "platform_name": xx,
        "model_type": xx,
        "model_name": xx,
        "api_base_url": xx,
        "api_key": xx,
        "api_proxy": xx,
    }}
    """
    result = {}
    if model_type:
        model_types = [f"{model_type}_models"]
    else:
        model_types = [
            "llm_models", "embed_models",
            "rerank_models", "image2text_models",
            "text2image_models", "image2image_models",
            "speech2text_models", "text2speech_models"]

    for p in list(get_config_platforms().values()):
        if platform_name is not None and platform_name != p.get("platform_name"):
            continue
        if p.get("auto_detect_model"):
            if p.get("platform_type") not in ["xinference"]: # TODO: 当前仅支持xinference自动检测模型
                logger.warning(f"自动检测模型不支持{p.get('platform_type')}平台")
                continue
            xf_url = get_base_url(p.get("api_base_url"))
            xf_models = detect_xf_models(xf_url)
            for type in model_types:
                p[type] = xf_models.get(type, [])

        for type in model_types:
            model_list = p.get(type, [])
            if not model_list:
                continue
            for name in model_list:
                if model_name is None or model_name == name:
                    result[name] = {
                        "platform_type": p.get("platform_type"),
                        "platform_name": p.get("platform_name"),
                        "model_type": type.split("_")[0],
                        "model_name": name,
                        "api_base_url": p.get("api_base_url"),
                        "api_key": p.get("api_key"),
                        "api_proxy": p.get("api_proxy"),
                    }
    return result

def get_default_llm():
    """获取可用的默认llm模型"""
    available_llms = list(get_config_models(model_type="llm").keys())
    if Settings.model_settings.DEFAULT_LLM_MODEL in available_llms:
        return Settings.model_settings.DEFAULT_LLM_MODEL
    else:
        logger.warning(f"没有在平台的可用模型中发现你设置的默认llm模型{Settings.model_settings.DEFAULT_LLM_MODEL}, 将使用{available_llms[0]}模型代替")
        return available_llms[0]

def get_default_embedding():
    """获取可用的默认embedding模型"""
    available_embeddings = list(get_config_models(model_type="embed").keys())
    if Settings.model_settings.DEFAULT_EMBEDDING_MODEL in available_embeddings:
        return Settings.model_settings.DEFAULT_EMBEDDING_MODEL
    else:
        logger.warning(f"没有在平台的可用模型中发现你设置的默认embedding模型{Settings.model_settings.DEFAULT_EMBEDDING_MODEL}, 将使用{available_embeddings[0]}模型代替")
        return available_embeddings[0]

def get_api_address(is_public: bool = False) -> str:
    """获取api地址"""
    api_info = Settings.basic_settings.API_SERVER
    if is_public:
        host = api_info.get("public_host", "127.0.0.1")
        port = api_info.get("public_port", "7861")
    else:
        host = api_info.get("host", "127.0.0.1")
        port = api_info.get("port", "7861")
        if host == "0.0.0.0":
            host = "127.0.0.1"
    return f"http://{host}:{port}"
