"""
TIR-Agent: Tool-Integrated Reasoning Agent based on Qwen-Agent
集成Memory、Skills、Planning三个子系统
"""

__version__ = "0.1.0"

from .agent import TIRAgent
from .config import Settings, get_settings, settings, reload_settings
from .context_manager import ContextManager
from .memory import MemoryManager
from .skills import SkillRegistry, BaseSkill
from .planning import TaskPlanner, PlanExecutor
from .planning.reasoner import StepReasoner

__all__ = [
    "TIRAgent",
    "Settings",
    "get_settings",
    "settings",
    "reload_settings",
    "ContextManager",
    "MemoryManager",
    "SkillRegistry",
    "BaseSkill",
    "TaskPlanner",
    "PlanExecutor",
    "StepReasoner",
]
