"""
BaseSkill基类定义

所有Skill的抽象基类，定义了Skill的基本接口和元数据结构。
"""
from abc import ABC, abstractmethod
from typing import Any, Optional
from dataclasses import dataclass, field


@dataclass
class SkillMetadata:
    """Skill元数据，从YAML加载"""
    name: str
    display_name: str
    version: str
    description: str
    parameters: list[dict]  # [{name, type, required, default, description, enum}]
    returns: dict  # {type, description}
    entry_point: str
    tags: list[str] = field(default_factory=list)
    enabled: bool = True


class BaseSkill(ABC):
    """所有Skill的基类"""

    def __init__(self, metadata: SkillMetadata, config: dict = None):
        self.metadata = metadata
        self.config = config or {}

    @abstractmethod
    def execute(self, **params) -> str:
        """执行Skill，返回结果字符串"""
        ...

    def validate_params(self, params: dict) -> dict:
        """验证和填充默认参数"""
        validated = {}
        for param_def in self.metadata.parameters:
            name = param_def['name']
            if name in params:
                validated[name] = params[name]
            elif param_def.get('required', False):
                raise ValueError(f"Missing required parameter: {name}")
            elif 'default' in param_def:
                validated[name] = param_def['default']
        return validated

    @property
    def name(self) -> str:
        return self.metadata.name

    @property
    def description(self) -> str:
        return self.metadata.description

    @property
    def enabled(self) -> bool:
        return self.metadata.enabled
