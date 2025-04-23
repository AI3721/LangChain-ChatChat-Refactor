from __future__ import annotations

import os
import ruamel.yaml
from io import StringIO
from pathlib import Path
from functools import cached_property
from pydantic import BaseModel, ConfigDict
from ruamel.yaml.comments import CommentedBase
from memoization import cached, CachingAlgorithmFlag
from typing import Type, Dict, TypedDict, TypeVar, Literal, Any
from pydantic_settings import BaseSettings, SettingsConfigDict, PydanticBaseSettingsSource, YamlConfigSettingsSource

__all__ = ["MyBaseModel", "BaseFileSettings", "SettingsConfigDict", "cached_property", "settings_property"]

def import_yaml() -> ruamel.yaml.YAML:
    """创建一个配置过的YAML对象"""
    yaml = ruamel.yaml.YAML()
    yaml.sequence_dash_offset = 2 # 破折号偏移量
    yaml.block_seq_indent = 2 # 块级序列缩进
    yaml.sequence_indent = 4 # 序列缩进
    yaml.map_indent = 2 # 映射缩进
    return yaml


class SubModelComment(TypedDict):
    """子模型注释模板"""
    model_obj: BaseModel
    dump_kwds: Dict
    is_entire_comment: bool = False
    sub_comments: Dict[str, "SubModelComment"]


class YamlTemplate:
    """创建YAML文件配置模板"""
    def __init__(self, model_obj: BaseModel, dump_kwds: Dict={}, sub_comments: Dict[str, SubModelComment]={}):
        self.model_obj = model_obj
        self.dump_kwds = dump_kwds
        self.sub_comments = sub_comments

    @cached_property
    def model_cls(self):
        """
        定义一个方法获取模型的类,
        并转换为一个属性缓存结果
        """
        return self.model_obj.__class__

    def _create_yaml_object(self) -> CommentedBase:
        """将pydantic对象转换为yaml对象"""
        data = self.model_obj.model_dump(**self.dump_kwds)
        yaml = import_yaml()
        buffer = StringIO()
        yaml.dump(data, buffer)
        buffer.seek(0)
        obj = yaml.load(buffer)
        return obj

    def get_class_comment(self, model_cls: Type[BaseModel] | BaseModel=None) -> str | None:
        """获取类的描述性注释, 可以重写"""
        if model_cls is None:
            model_cls = self.model_cls
        return model_cls.model_json_schema().get("description")

    def get_field_comment(self, field_name: str, model_obj: BaseModel=None) -> str | None:
        """获取字段的描述性注释, 可以重写"""
        if model_obj is None:
            schema = self.model_cls.model_json_schema().get("properties", {})
        else:
            schema = model_obj.model_json_schema().get("properties", {})
        if field := schema.get(field_name):
            # := 海象运算符, 如果field_name不存在, 则field为None
            lines = [field.get("description", "")]
            if enum := field.get("enum"):
                # := 海象运算符, 如果enum不存在, 则enum为None
                lines.append(f"可选值：{enum}")
            return "\n".join(lines)

    def create_yaml_template(self, write_to: str | Path | bool = False, indent: int = 0) -> str:
        """创建一个包含默认对象和注释的YAML模板"""
        cls = self.model_cls
        obj = self._create_yaml_object()

        # 添加类注释
        cls_comment = self.get_class_comment()
        if cls_comment:
            obj.yaml_set_start_comment(cls_comment + "\n\n", indent)
        
        # 定义一个递归函数, 用于设置子字段注释
        def _set_subfield_comment(
            o: CommentedBase,# obj ruamel.yaml.YAML对象
            m: BaseModel, # model_obj pydantic对象
            n: str, # subfield_name 子字段名
            sub_comment: SubModelComment, indent: int):
            if sub_comment:
                if sub_comment.get("is_entire_comment"):
                    comment = (
                        YamlTemplate( # 递归
                            sub_comment["model_obj"],
                            dump_kwds=sub_comment.get("dump_kwds", {}),
                            sub_comments=sub_comment.get("sub_comments", {}),
                        ).create_yaml_template()
                    )
                    if comment:
                        o.yaml_set_comment_before_after_key(n, "\n"+comment, indent=indent)
                elif sub_model_obj := sub_comment.get("model_obj"):
                    comment = self.get_field_comment(n, m) or self.get_class_comment(sub_model_obj)
                    if comment:
                        o.yaml_set_comment_before_after_key(n, "\n"+comment, indent=indent)
                    for name in sub_model_obj.model_fields:
                        subsub_comment = sub_comment.get("sub_comments", {}).get(name, {})
                        _set_subfield_comment(o[n], sub_model_obj, name, subsub_comment, indent+2)
            else:
                comment = self.get_field_comment(n, m)
                if comment:
                    o.yaml_set_comment_before_after_key(n, "\n"+comment, indent=indent)

        for name in cls.model_fields:
            sub_comment = self.sub_comments.get(name, {})
            _set_subfield_comment(obj, self.model_obj, name, sub_comment, indent)

        yaml = import_yaml()
        buffer = StringIO()
        yaml.dump(obj, buffer)
        template = buffer.getvalue()

        if write_to is True:
            write_to = self.model_cls.model_config.get("yaml_file")
        if write_to:
            with open(write_to, "w", encoding="utf-8") as f:
                f.write(template)

        return template


class MyBaseModel(BaseModel):
    """自定义pydantic模型类"""
    model_config = ConfigDict(
        use_attribute_docstrings=True,
        env_file_encoding="utf-8",
        extra="allow",
    )


class BaseFileSettings(BaseSettings):
    """定义一个加载文件配置的类"""
    model_config = SettingsConfigDict(
        use_attribute_docstrings=True,
        yaml_file_encoding="utf-8",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 一个模型初始化后调用的方法，在这里设置自动重载
    def model_post_init(self, __context: Any) -> None:
        self._auto_reload = True
        return super().model_post_init(__context)

    @property
    def auto_reload(self) -> bool:
        return self._auto_reload
    
    @auto_reload.setter
    def auto_reload(self, val: bool):
        self._auto_reload = val
    
    # 一个类方法，用于自定义配置来源
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return init_settings, env_settings, dotenv_settings, YamlConfigSettingsSource(settings_cls)

    def create_template_file(
        self,
        model_obj: BaseFileSettings=None,
        dump_kwds: Dict={},
        sub_comments: Dict[str, SubModelComment]={},
        write_file: bool | str | Path = False,
        file_format: Literal["yaml", "json"] = "yaml",
    ) -> str:
        if model_obj is None:
            model_obj = self
        if file_format == "yaml":
            template = YamlTemplate(model_obj=model_obj, dump_kwds=dump_kwds, sub_comments=sub_comments)
            return template.create_yaml_template(write_to=write_file)
        else:
            dump_kwds.setdefault("indent", 4)
            data = model_obj.model_dump_json(**dump_kwds)
            if write_file:
                write_file = self.model_config.get("json_file")
                with open(write_file, "w", encoding="utf-8") as f:
                    f.write(data)
            return data


# 定义一个泛型类型变量
_T = TypeVar("_T", bound=BaseFileSettings)

# 生成唯一的键, 用来识别配置文件何时发生变化
def _lazy_load_key(settings: BaseSettings):
    keys = [settings.__class__]
    for n in ["env_file", "json_file", "yaml_file", "toml_file"]:
        key = None
        if file := settings.model_config.get(n):
            if os.path.isfile(file) and os.path.getsize(file) > 0:
                key = int(os.path.getmtime(file))
        keys.append(key)
    return tuple(keys)

# 创建一个缓存机制, 缓存settings文件生成的唯一的键, 创建一个泛型函数, 当键发生变化时, 重新加载配置文件
@cached(max_size=1, algorithm=CachingAlgorithmFlag.LRU, thread_safe=True, custom_key_maker=_lazy_load_key)
def _cached_settings(settings: _T) -> _T:
    if settings.auto_reload:
        settings.__init__()
    return settings

# 装饰器, 配置装换为属性
def settings_property(settings: _T):
    def wrapper(self) -> _T:
        return _cached_settings(settings)
    return property(wrapper)
