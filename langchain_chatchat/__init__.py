import sys
import types
import importlib

# 动态导入chatchat模块
chatchat = importlib.import_module("chatchat")

# 创建新的langchain_chatchat模块
module = types.ModuleType("langchain_chatchat")
sys.modules["langchain_chatchat"] = module

# 复制chatchat的所有属性到langchain_chatchat
for attr in dir(chatchat):
    if not attr.startswith("_"):
        setattr(module, attr, getattr(chatchat, attr))


# 动态导入chatchat子模块
def import_submodule(name):
    full_name = f"chatchat.{name}"
    submodule = importlib.import_module(full_name)
    sys.modules[f"langchain_chatchat.{name}"] = submodule
    for attr in dir(submodule):
        if not attr.startswith("_"):
            setattr(module, attr, getattr(submodule, attr))

# 导入需要导入的子模块
submodules = ["client", "web"]
for submodule in submodules:
    import_submodule(submodule)
