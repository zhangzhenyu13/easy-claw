"""
Planning模块数据模型

定义任务规划相关的数据结构和状态枚举。
"""

from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum


class StepStatus(Enum):
    """步骤执行状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class StepDecision(str, Enum):
    """步骤执行后的决策类型"""
    CONTINUE = "continue"   # 标记成功，继续下一步
    RETRY = "retry"         # 调整参数后重试
    SKIP = "skip"           # 跳过非关键步骤
    REPLAN = "replan"       # 触发完整重规划
    ABORT = "abort"         # 终止整个计划


@dataclass
class Step:
    """
    计划中的单个步骤
    
    Attributes:
        id: 步骤唯一标识符
        description: 步骤描述
        skill_name: 要调用的Skill名称
        params: 调用参数
        depends_on: 依赖的Step ID列表
        status: 当前执行状态
        result: 执行结果
        error: 错误信息
        retry_count: 当前重试次数
        max_retries: 最大重试次数
        context: 步骤上下文，存放前序步骤结果等
        tir_history: TIR循环尝试历史记录
    """
    id: str
    description: str
    skill_name: str          # 要调用的Skill名称
    params: dict = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)  # 依赖的Step ID列表
    status: StepStatus = StepStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 2
    context: dict = field(default_factory=dict)  # 存放前序步骤结果等上下文
    tir_history: list = field(default_factory=list)  # TIR循环尝试历史记录


@dataclass
class Plan:
    """
    任务执行计划
    
    Attributes:
        task_description: 任务描述
        steps: 步骤列表
        reasoning: 规划推理过程
        memory_context: 使用的记忆上下文
    """
    task_description: str
    steps: list[Step] = field(default_factory=list)
    reasoning: str = ""         # 规划推理过程
    memory_context: str = ""    # 使用的记忆上下文
    
    @property
    def step_results(self) -> dict:
        """映射 step_id -> result，便于后续步骤引用"""
        return {s.id: s.result for s in self.steps if s.status == StepStatus.COMPLETED and s.result}

    def get_ready_steps(self) -> list[Step]:
        """
        获取所有依赖已完成、状态为pending的步骤
        SKIPPED状态的步骤被视为"已完成"（允许下游步骤继续）
        
        Returns:
            可执行的步骤列表
        """
        resolved_ids = {s.id for s in self.steps if s.status in (StepStatus.COMPLETED, StepStatus.SKIPPED)}
        return [s for s in self.steps if s.status == StepStatus.PENDING 
                and all(dep in resolved_ids for dep in s.depends_on)]
    
    def is_complete(self) -> bool:
        """
        检查计划是否已完成
        
        Returns:
            所有步骤都已完成或跳过
        """
        return all(s.status in (StepStatus.COMPLETED, StepStatus.SKIPPED) for s in self.steps)
    
    def has_failed(self) -> bool:
        """
        检查计划是否有失败的步骤
        
        Returns:
            存在失败步骤
        """
        return any(s.status == StepStatus.FAILED for s in self.steps)
    
    def get_failed_steps(self) -> list[Step]:
        """获取所有失败的步骤"""
        return [s for s in self.steps if s.status == StepStatus.FAILED]


@dataclass
class ReasoningResult:
    """
    LLM推理结果
    
    Attributes:
        decision: 决策类型
        reasoning: LLM推理说明
        adjusted_params: RETRY时的调整参数
        observation_analysis: 观察分析 - CoT Step 1 的输出
        root_cause: 根因诊断 - CoT Step 2 的输出
        recovery_strategy: 恢复策略 - CoT Step 3 的输出
        confidence: 决策置信度 (0.0-1.0)
        lessons_learned: 本次尝试的经验教训
    """
    decision: StepDecision
    reasoning: str = ""              # LLM推理说明
    adjusted_params: Optional[dict] = None  # RETRY时的调整参数
    observation_analysis: str = ""   # 观察分析 - CoT Step 1 的输出
    root_cause: str = ""             # 根因诊断 - CoT Step 2 的输出
    recovery_strategy: str = ""      # 恢复策略 - CoT Step 3 的输出
    confidence: float = 0.5          # 决策置信度 (0.0-1.0)
    lessons_learned: str = ""        # 本次尝试的经验教训


@dataclass
class TaskResult:
    """
    任务执行结果
    
    Attributes:
        success: 是否成功
        plan: 执行的计划
        final_answer: 最终答案
        steps_completed: 完成的步骤数
        steps_failed: 失败的步骤数
    """
    success: bool
    plan: Plan
    final_answer: str = ""
    steps_completed: int = 0
    steps_failed: int = 0
