"""
任务规划器

基于LLM的任务规划系统，支持复杂任务分解和重新规划。
"""

import json
import logging
import re
from typing import Any, Optional

import openai

from .models import Plan, Step, StepStatus

logger = logging.getLogger("tir_agent.planning")

# 导入PromptManager，用于类型提示
try:
    from ..prompt_manager import PromptManager
except ImportError:
    PromptManager = None


class TaskPlanner:
    """
    任务规划器
    
    使用LLM生成任务执行计划，支持任务复杂度评估和重新规划。
    """
    
    def __init__(self, llm_config: dict, prompt_manager: Optional[Any] = None):
        """
        初始化任务规划器
        
        Args:
            llm_config: LLM配置字典，包含model, api_key, base_url等
            prompt_manager: PromptManager实例（可选）
        """
        self.llm_config = llm_config
        self.model = llm_config.get("model", "qwen-max-latest")
        self.api_key = llm_config.get("api_key") or llm_config.get("dashscope_api_key")
        self.base_url = llm_config.get("base_url") or llm_config.get("model_server")
        
        # 初始化OpenAI客户端
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        
        # 初始化PromptManager（如果未提供则创建默认实例）
        if prompt_manager is None and PromptManager is not None:
            from ..config import settings
            prompt_manager = PromptManager(
                prompts_dir=settings.prompts_dir,
                version=settings.prompt_version
            )
        self.prompt_manager = prompt_manager
        
        logger.info(f"TaskPlanner initialized with model: {self.model}")
    
    def plan(self, query: str, available_skills: list[dict], memory_context: str = "") -> Plan:
        """
        基于用户查询生成任务执行计划
        
        Args:
            query: 用户任务描述
            available_skills: 可用技能列表，每项包含name, description, parameters
            memory_context: 历史记忆上下文
            
        Returns:
            生成的执行计划
        """
        logger.info("开始规划任务: %s", query[:100])
        
        prompt = self._build_planning_prompt(query, available_skills, memory_context)
        
        try:
            response = self._call_llm(prompt)
            plan = self._parse_plan_response(response, query)
            logger.info("规划完成，共 %d 个步骤", len(plan.steps))
            for step in plan.steps:
                logger.debug("  步骤 [%s]: %s (skill: %s, 依赖: %s)", 
                           step.id, step.description, step.skill_name, step.depends_on or [])
            return plan
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            raise  # 让 agent.py 捕获并回退到直接调用
    
    def replan(self, original_plan: Plan, failed_step: Step, error: str) -> Plan:
        """
        基于失败信息重新规划
        
        Args:
            original_plan: 原始计划
            failed_step: 失败的步骤
            error: 错误信息
            
        Returns:
            重新规划后的计划
        """
        logger.warning("触发重新规划，失败步骤: %s, 错误: %s", failed_step.id, error[:100])
        
        # 标记失败步骤
        failed_step.status = StepStatus.FAILED
        failed_step.error = error
        
        # 构建重新规划prompt
        steps_info = []
        for step in original_plan.steps:
            status = step.status.value if step.status else "pending"
            steps_info.append(f"- {step.id}: {step.description} [{status}]")
        
        # 使用PromptManager获取prompt
        if self.prompt_manager:
            prompt = self.prompt_manager.get(
                "planning_replan",
                original_task=original_plan.task_description,
                steps_info="\n".join(steps_info),
                failed_step_id=failed_step.id,
                failed_step_description=failed_step.description,
                error=error
            )
        else:
            # Fallback: 使用硬编码prompt
            prompt = f"""你是一个任务规划助手。之前的计划执行失败，需要重新规划。

原始任务: {original_plan.task_description}

已执行步骤:
{chr(10).join(steps_info)}

失败步骤: {failed_step.id} - {failed_step.description}
错误信息: {error}

请重新规划剩余任务。你可以选择:
1. 跳过失败步骤继续执行
2. 调整参数重试失败步骤
3. 采用替代方案

请输出JSON格式:
{{
    "reasoning": "重新规划的推理过程",
    "steps": [
        {{
            "id": "step_id",
            "description": "步骤描述",
            "skill_name": "skill_name",
            "params": {{}},
            "depends_on": []
        }}
    ]
}}

注意:
- 只包含尚未完成的步骤（pending状态）
- 已完成的步骤不需要重复
- 失败的步骤可以选择跳过或重试
"""
        
        try:
            response = self._call_llm(prompt)
            new_plan = self._parse_plan_response(response, original_plan.task_description)
            # 保留已完成的步骤
            completed_steps = [s for s in original_plan.steps if s.status == StepStatus.COMPLETED]
            new_plan.steps = completed_steps + new_plan.steps
            new_plan.memory_context = original_plan.memory_context
            return new_plan
        except Exception as e:
            logger.error(f"Replanning failed: {e}")
            # 标记失败步骤为跳过，继续执行剩余步骤
            failed_step.status = StepStatus.SKIPPED
            return original_plan
    
    def assess_complexity(self, query: str) -> str:
        """
        评估任务复杂度
        
        Args:
            query: 用户任务描述
            
        Returns:
            "simple" 或 "complex"
            
        简单任务特征：
        - 单一明确指令
        - 不需要多步工具调用
        
        复杂任务特征：
        - 多步骤
        - 条件依赖
        - 需要多工具协作
        """
        # 使用PromptManager获取prompt
        if self.prompt_manager:
            prompt = self.prompt_manager.get("planning_complexity", query=query)
        else:
            # Fallback: 使用硬编码prompt
            prompt = f"""评估以下任务的复杂度，只回复 "simple" 或 "complex"。

任务: {query}

判断标准:
- simple: 单一指令，不需要多步工具调用
- complex: 多步骤、有条件依赖、需要多工具协作

只输出一个词: simple 或 complex"""
        
        try:
            response = self._call_llm(prompt).strip().lower()
            if "complex" in response:
                complexity = "complex"
            else:
                complexity = "simple"
            logger.info("任务复杂度评估: %s", complexity)
            return complexity
        except Exception as e:
            logger.error(f"Complexity assessment failed: {e}")
            # 默认返回complex以启用完整规划流程
            return "complex"
    
    def _call_llm(self, prompt: str) -> str:
        """
        内部LLM调用方法
        
        Args:
            prompt: 提示词
            
        Returns:
            LLM响应文本
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个任务规划助手，擅长将复杂任务分解为可执行的步骤。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                top_p=0.8,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise
    
    def _parse_plan_response(self, response: str, query: str) -> Plan:
        """
        解析LLM响应为Plan对象
        
        Args:
            response: LLM响应文本
            query: 原始任务描述
            
        Returns:
            解析后的Plan对象
        """
        # 健壮处理JSON解析，处理markdown代码块
        json_str = response
        
        # 尝试提取markdown代码块中的JSON
        code_block_pattern = r'```(?:json)?\s*([\s\S]*?)```'
        matches = re.findall(code_block_pattern, response)
        if matches:
            json_str = matches[0].strip()
        
        # 尝试找到JSON对象
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # 尝试提取花括号中的内容
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON from response: {e}")
                    raise ValueError(f"Invalid JSON response: {response[:200]}...")
            else:
                raise ValueError(f"No JSON found in response: {response[:200]}...")
        
        # 构建Plan对象
        reasoning = data.get("reasoning", "")
        steps_data = data.get("steps", [])
        
        steps = []
        for step_data in steps_data:
            step = Step(
                id=step_data.get("id", f"step_{len(steps) + 1}"),
                description=step_data.get("description", ""),
                skill_name=step_data.get("skill_name", ""),
                params=step_data.get("params", {}),
                depends_on=step_data.get("depends_on", []),
                status=StepStatus.PENDING
            )
            steps.append(step)
        
        return Plan(
            task_description=query,
            steps=steps,
            reasoning=reasoning
        )
    
    def _build_planning_prompt(self, query: str, available_skills: list[dict], memory_context: str) -> str:
        """
        构建规划prompt
        
        Args:
            query: 用户任务描述
            available_skills: 可用技能列表
            memory_context: 历史记忆上下文
            
        Returns:
            完整的prompt文本
        """
        # 格式化技能列表
        skills_desc = []
        for skill in available_skills:
            name = skill.get("name", "unknown")
            desc = skill.get("description", "")
            params = skill.get("parameters", {})
            skills_desc.append(f"- {name}: {desc}")
            if params:
                params_str = json.dumps(params, ensure_ascii=False)
                skills_desc.append(f"  参数: {params_str}")
        
        skills_text = "\n".join(skills_desc) if skills_desc else "无可用技能"
        
        # 构建记忆上下文部分
        memory_text = f"\n\n历史记忆上下文:\n{memory_context}" if memory_context else ""
        
        # 使用PromptManager获取prompt
        if self.prompt_manager:
            prompt = self.prompt_manager.get(
                "planning_plan",
                skills_text=skills_text,
                memory_text=memory_text,
                query=query
            )
        else:
            # Fallback: 使用硬编码prompt
            prompt = f"""你是一个任务规划助手。请将用户任务分解为可执行的步骤。

可用技能列表:
{skills_text}
{memory_text}

当前任务: {query}

请输出JSON格式计划:
{{
    "reasoning": "规划推理过程，解释为什么这样分解任务",
    "steps": [
        {{
            "id": "step_1",
            "description": "步骤描述",
            "skill_name": "使用的技能名称",
            "params": {{"参数名": "参数值"}},
            "depends_on": []  // 依赖的步骤ID列表
        }}
    ]
}}

规划要求:
1. 步骤ID使用 step_1, step_2 等格式
2. 每个步骤必须指定一个可用技能
3. 如果步骤有依赖，在depends_on中列出前置步骤ID
4. 参数值应从任务描述中提取，或使用占位符
5. 推理过程要清晰说明任务分解思路"""
        
        return prompt
