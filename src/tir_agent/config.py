"""
配置管理模块
"""
import os
from typing import Optional
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import Field

# 加载环境变量
load_dotenv()


class Settings(BaseSettings):
    """应用配置"""
    
    # LLM配置
    model_name: str = Field(default="qwen3.6-plus", alias="MODEL_NAME")
    model_type: str = Field(default="qwen_dashscope", alias="MODEL_TYPE")
    
    # DashScope配置
    dashscope_api_key: Optional[str] = Field(default=None, alias="DASHSCOPE_API_KEY")
    
    # OpenAI兼容API配置
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openai_api_base: Optional[str] = Field(default=None, alias="OPENAI_API_BASE")
    
    # VLM配置
    vlm_model_name: str = Field(default="qwen-vl-max", alias="VLM_MODEL_NAME")
    vlm_api_key: Optional[str] = Field(default=None, alias="VLM_API_KEY")
    vlm_api_base: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        alias="VLM_API_BASE"
    )
    
    # 代码执行配置
    code_executor_timeout: int = Field(default=60, alias="CODE_EXECUTOR_TIMEOUT")
    
    # 文件处理配置
    max_file_size_mb: int = Field(default=50, alias="MAX_FILE_SIZE_MB")
    temp_dir: str = Field(default="/tmp/tir_agent", alias="TEMP_DIR")
    
    # Memory配置
    memory_enabled: bool = Field(default=True, alias="MEMORY_ENABLED")
    memory_db_path: str = Field(default="~/.tir_agent/memory.db", alias="MEMORY_DB_PATH")
    memory_compression_threshold: int = Field(default=10, alias="MEMORY_COMPRESSION_THRESHOLD")
    memory_recall_top_k: int = Field(default=5, alias="MEMORY_RECALL_TOP_K")
    
    # Skills配置
    skill_dirs: list = Field(default_factory=lambda: ["skills"], alias="SKILL_DIRS")
    custom_skill_dir: str = Field(default="", alias="CUSTOM_SKILL_DIR")  # 改为空字符串，不再默认 ~/.tir_agent/skills
    
    # Planning配置
    planning_enabled: bool = Field(default=True, alias="PLANNING_ENABLED")
    complexity_threshold: str = Field(default="auto", alias="COMPLEXITY_THRESHOLD")
    max_tir_loops: int = Field(default=5, alias="MAX_TIR_LOOPS")  # 单步 TIR 循环最大次数
    
    # ReAct执行模式配置
    execution_mode: str = Field(default="auto", alias="EXECUTION_MODE")  # auto(双层模式), react(纯ReAct), planning(纯Planning)
    react_max_iterations: int = Field(default=10, alias="REACT_MAX_ITERATIONS")  # ReAct最大迭代次数
    max_react_per_step: int = Field(default=5, alias="MAX_REACT_PER_STEP")  # 双层模式下每步ReAct最大迭代次数
    
    # 工具输出配置
    max_tool_output_chars: int = Field(default=4000, alias="MAX_TOOL_OUTPUT_CHARS")
    
    # ContextManager配置
    max_context_tokens: int = Field(default=30000, alias="MAX_CONTEXT_TOKENS")
    
    # Prompt配置
    prompt_version: str = Field(default="default", alias="PROMPT_VERSION")
    prompts_dir: str = Field(default="prompts", alias="PROMPTS_DIR")
    
    class Config:
        env_file = ".env"
        extra = "ignore"
    
    def get_llm_config(self) -> dict:
        """获取LLM配置"""
        if self.model_type == "qwen_dashscope":
            # 使用DashScope的OpenAI兼容模式
            return {
                "model": self.model_name,
                "model_server": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "api_key": self.dashscope_api_key,
                "generate_cfg": {
                    "top_p": 0.8,
                    "temperature": 0.7,
                }
            }
        else:
            return {
                "model": self.model_name,
                "model_server": self.openai_api_base,
                "api_key": self.openai_api_key,
                "generate_cfg": {
                    "top_p": 0.8,
                    "temperature": 0.7,
                }
            }
    
    def get_vlm_config(self) -> dict:
        """获取VLM配置"""
        # 优先使用VLM专用配置
        api_key = self.vlm_api_key or self.dashscope_api_key or self.openai_api_key
        base_url = self.vlm_api_base
        
        # 如果设置了VLM_MODEL_NAME环境变量，使用它
        vlm_model = os.getenv("VLM_MODEL_NAME", self.vlm_model_name)
        
        return {
            "model": vlm_model,
            "api_key": api_key,
            "base_url": base_url,
        }


# 全局配置实例
# 尝试加载配置，如果失败则使用默认值
try:
    settings = Settings()
except Exception:
    # 如果pydantic_settings不可用，使用简单配置
    class SimpleSettings:
        def __init__(self):
            load_dotenv()
            self.model_name = os.getenv("MODEL_NAME", "qwen-max-latest")
            self.model_type = os.getenv("MODEL_TYPE", "qwen_dashscope")
            self.dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
            self.openai_api_base = os.getenv("OPENAI_API_BASE")
            self.vlm_model_name = os.getenv("VLM_MODEL_NAME", "qwen-vl-max")
            self.vlm_api_key = os.getenv("VLM_API_KEY")
            self.vlm_api_base = os.getenv("VLM_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
            self.code_executor_timeout = int(os.getenv("CODE_EXECUTOR_TIMEOUT", "60"))
            self.max_file_size_mb = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
            self.temp_dir = os.getenv("TEMP_DIR", "/tmp/tir_agent")
            # Memory配置
            self.memory_enabled = os.getenv("MEMORY_ENABLED", "true").lower() == "true"
            self.memory_db_path = os.getenv("MEMORY_DB_PATH", "~/.tir_agent/memory.db")
            self.memory_compression_threshold = int(os.getenv("MEMORY_COMPRESSION_THRESHOLD", "10"))
            self.memory_recall_top_k = int(os.getenv("MEMORY_RECALL_TOP_K", "5"))
            # Skills配置
            self.skill_dirs = ["skills"]
            self.custom_skill_dir = os.getenv("CUSTOM_SKILL_DIR", "")  # 改为空字符串
            # Planning配置
            self.planning_enabled = os.getenv("PLANNING_ENABLED", "true").lower() == "true"
            self.complexity_threshold = os.getenv("COMPLEXITY_THRESHOLD", "auto")
            self.max_tir_loops = int(os.getenv("MAX_TIR_LOOPS", "5"))  # 单步 TIR 循环最大次数
            
            # ReAct执行模式配置
            self.execution_mode = os.getenv("EXECUTION_MODE", "auto")  # auto(双层模式), react(纯ReAct), planning(纯Planning)
            self.react_max_iterations = int(os.getenv("REACT_MAX_ITERATIONS", "10"))  # ReAct最大迭代次数
            self.max_react_per_step = int(os.getenv("MAX_REACT_PER_STEP", "5"))  # 双层模式下每步ReAct最大迭代次数
            # 工具输出配置
            self.max_tool_output_chars = int(os.getenv("MAX_TOOL_OUTPUT_CHARS", "4000"))
            # ContextManager配置
            self.max_context_tokens = int(os.getenv("MAX_CONTEXT_TOKENS", "30000"))
            # Prompt配置
            self.prompt_version = os.getenv("PROMPT_VERSION", "default")
            self.prompts_dir = os.getenv("PROMPTS_DIR", "prompts")
        
        def get_llm_config(self) -> dict:
            if self.model_type == "qwen_dashscope":
                # 使用DashScope的OpenAI兼容模式
                return {
                    "model": self.model_name,
                    "model_server": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "api_key": self.dashscope_api_key,
                    "generate_cfg": {"top_p": 0.8, "temperature": 0.7}
                }
            return {
                "model": self.model_name,
                "model_server": self.openai_api_base,
                "api_key": self.openai_api_key,
                "generate_cfg": {"top_p": 0.8, "temperature": 0.7}
            }
        
        def get_vlm_config(self) -> dict:
            api_key = self.vlm_api_key or self.dashscope_api_key or self.openai_api_key
            vlm_model = os.getenv("VLM_MODEL_NAME", self.vlm_model_name)
            return {
                "model": vlm_model,
                "api_key": api_key,
                "base_url": self.vlm_api_base,
            }
    
    settings = SimpleSettings()


def get_settings() -> Settings:
    """获取Settings实例（用于依赖注入）"""
    return settings


def reload_settings():
    """重新加载配置（在环境变量更新后调用）"""
    global settings
    try:
        settings = Settings()
    except Exception:
        settings = SimpleSettings()
    return settings
