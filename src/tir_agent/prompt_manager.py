"""
Prompt 管理模块
负责加载和管理不同版本的 prompt 模板
"""
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class PromptManager:
    """
    Prompt 管理器，支持多版本管理和变量填充
    
    目录结构:
        prompts/
            default/
                system.txt
                planning_plan.txt
                ...
            custom/
                system.txt
                ...
    """
    
    def __init__(self, prompts_dir: str = "prompts", version: str = "default"):
        """
        初始化 PromptManager
        
        Args:
            prompts_dir: prompts 根目录路径（相对于项目根目录或绝对路径）
            version: 当前使用的 prompt 版本（对应 prompts/ 下的子目录名）
        """
        self._prompts_dir = self._resolve_prompts_dir(prompts_dir)
        self._version = version
        self._cache: dict[str, str] = {}
    
    def _resolve_prompts_dir(self, prompts_dir: str) -> Path:
        """
        解析 prompts 目录路径
        
        支持绝对路径和相对路径：
        - 绝对路径：直接使用
        - 相对路径：基于项目根目录（向上查找到包含 prompts 目录的位置）
        """
        path = Path(prompts_dir)
        
        if path.is_absolute():
            return path
        
        # 相对路径：从当前文件位置向上查找项目根目录
        # 项目根目录定义为包含 prompts 目录的位置
        current_file = Path(__file__).resolve()
        
        # 从当前文件所在目录开始向上查找
        for parent in current_file.parents:
            potential_path = parent / prompts_dir
            if potential_path.exists() and potential_path.is_dir():
                return potential_path
        
        # 如果没找到，使用相对于当前文件的路径作为 fallback
        # 假设当前文件在 src/tir_agent/ 下，向上两级到项目根目录
        project_root = current_file.parent.parent.parent
        return project_root / prompts_dir
    
    def _get_prompt_path(self, name: str, version: Optional[str] = None) -> Path:
        """获取指定 prompt 文件的完整路径"""
        ver = version or self._version
        return self._prompts_dir / ver / f"{name}.txt"
    
    def _load_prompt(self, name: str) -> str:
        """
        加载指定名称的 prompt 模板
        
        优先从当前版本加载，如果失败则 fallback 到 default 版本
        """
        # 先尝试当前版本
        prompt_path = self._get_prompt_path(name, self._version)
        
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8")
        
        # Fallback 到 default 版本
        if self._version != "default":
            fallback_path = self._get_prompt_path(name, "default")
            if fallback_path.exists():
                return fallback_path.read_text(encoding="utf-8")
        
        # 文件不存在，抛出错误
        searched_paths = [str(prompt_path)]
        if self._version != "default":
            searched_paths.append(str(self._get_prompt_path(name, "default")))
        
        raise FileNotFoundError(
            f"Prompt file '{name}.txt' not found. "
            f"Searched in: {', '.join(searched_paths)}"
        )
    
    def get_raw(self, name: str) -> str:
        """
        获取原始 prompt 模板（不填充变量）
        
        Args:
            name: prompt 名称，如 'system', 'planning_plan' 等
            
        Returns:
            原始 prompt 模板字符串
            
        Raises:
            FileNotFoundError: 如果文件不存在
        """
        # 检查缓存
        if name in self._cache:
            return self._cache[name]
        
        # 加载并缓存
        content = self._load_prompt(name)
        self._cache[name] = content
        return content
    
    def get(self, name: str, **kwargs) -> str:
        """
        获取 prompt 并用 kwargs 填充变量
        
        Args:
            name: prompt 名称，如 'system', 'planning_plan' 等
            **kwargs: 用于填充模板变量的关键字参数
            
        Returns:
            填充后的 prompt 字符串
            
        Raises:
            FileNotFoundError: 如果文件不存在
            KeyError: 如果模板中有变量未在 kwargs 中提供
        """
        template = self.get_raw(name)
        
        if not kwargs:
            return template
        
        try:
            return template.format(**kwargs)
        except (KeyError, IndexError, ValueError) as e:
            # 如果 format 失败，使用 safe_format 作为 fallback
            logger.warning(f"[PromptManager] format() failed for '{name}': {e}, using safe_format")
            return self._safe_format(template, **kwargs)
    
    def _safe_format(self, template: str, **kwargs) -> str:
        """
        安全的字符串格式化，不会因为未知占位符或格式错误而崩溃。
        使用逐个替换的方式，只替换 kwargs 中提供的键。
        """
        result = template
        for key, value in kwargs.items():
            # 替换 {key} 但不替换 {{key}}（已转义的）
            placeholder = '{' + key + '}'
            result = result.replace(placeholder, str(value))
        return result
    
    def list_versions(self) -> list[str]:
        """
        列出所有可用的 prompt 版本
        
        Returns:
            prompts_dir 下的所有子目录名列表
        """
        if not self._prompts_dir.exists():
            return []
        
        versions = []
        for item in self._prompts_dir.iterdir():
            if item.is_dir():
                versions.append(item.name)
        
        return sorted(versions)
    
    def switch_version(self, version: str):
        """
        切换到指定版本，清除缓存
        
        Args:
            version: 目标版本名称
        """
        self._version = version
        self._cache.clear()
    
    @property
    def current_version(self) -> str:
        """获取当前版本"""
        return self._version
    
    @property
    def prompts_dir(self) -> Path:
        """获取 prompts 目录路径"""
        return self._prompts_dir


# 模块级便捷函数和全局实例
_default_manager: Optional[PromptManager] = None


def get_prompt_manager(
    prompts_dir: Optional[str] = None,
    version: Optional[str] = None
) -> PromptManager:
    """
    获取 PromptManager 实例
    
    如果不提供参数，使用环境变量或默认值：
    - PROMPTS_DIR: prompts 目录路径
    - PROMPT_VERSION: prompt 版本
    
    Args:
        prompts_dir: prompts 目录路径（可选）
        version: prompt 版本（可选）
        
    Returns:
        PromptManager 实例
    """
    global _default_manager
    
    # 从环境变量获取默认值
    env_prompts_dir = prompts_dir or os.getenv("PROMPTS_DIR", "prompts")
    env_version = version or os.getenv("PROMPT_VERSION", "default")
    
    # 如果参数与当前实例不同，创建新实例
    if _default_manager is None:
        _default_manager = PromptManager(
            prompts_dir=env_prompts_dir,
            version=env_version
        )
    elif prompts_dir is not None or version is not None:
        # 显式指定了参数，创建新实例
        return PromptManager(
            prompts_dir=env_prompts_dir,
            version=env_version
        )
    
    return _default_manager


def get_prompt(name: str, **kwargs) -> str:
    """
    便捷函数：获取填充后的 prompt
    
    Args:
        name: prompt 名称
        **kwargs: 模板变量
        
    Returns:
        填充后的 prompt 字符串
    """
    return get_prompt_manager().get(name, **kwargs)


def get_raw_prompt(name: str) -> str:
    """
    便捷函数：获取原始 prompt 模板
    
    Args:
        name: prompt 名称
        
    Returns:
        原始 prompt 模板字符串
    """
    return get_prompt_manager().get_raw(name)


# 导出主要类和函数
__all__ = [
    "PromptManager",
    "get_prompt_manager",
    "get_prompt",
    "get_raw_prompt",
]
