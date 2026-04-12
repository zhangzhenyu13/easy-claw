"""Step-level Reasoner for Tool-Integrated Reasoning.

每个步骤执行后，由 LLM 审查 observation 并决定下一步行动。
"""
import json
import logging
import re
from typing import Optional, Callable

from .models import Step, Plan, StepDecision, ReasoningResult, StepStatus

logger = logging.getLogger(__name__)


class StepReasoner:
    """步骤间 LLM 推理器，实现 Tool-Integrated Reasoning。"""
    
    def __init__(self, llm_caller: Callable):
        """
        Args:
            llm_caller: 调用LLM的函数，签名为 (messages: list[dict]) -> str
                       接收消息列表，返回LLM文本响应
        """
        self.llm_caller = llm_caller
    
    def reason(self, step: Step, observation: str, plan: Plan, is_error: bool = False, 
               tir_attempt: int = 0, max_tir_loops: int = 5) -> ReasoningResult:
        """审查步骤执行结果，决定下一步行动。
        
        Args:
            step: 当前执行的步骤
            observation: 工具执行结果或错误信息
            plan: 当前计划（包含所有步骤状态）
            is_error: observation 是否为错误信息
            tir_attempt: 当前 TIR 循环尝试次数（从0开始）
            max_tir_loops: 最大 TIR 循环次数
        """
        # 构建上下文
        completed_summary = self._build_completed_summary(plan)
        pending_steps = [s for s in plan.steps if s.status == StepStatus.PENDING]
        
        prompt = self._build_reasoning_prompt(
            step=step,
            observation=observation,
            is_error=is_error,
            completed_summary=completed_summary,
            pending_steps=pending_steps,
            tir_attempt=tir_attempt,
            max_tir_loops=max_tir_loops
        )
        
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.llm_caller(messages)
            result = self._parse_reasoning_response(response, is_error)
            logger.info(f"[Reasoner] Step '{step.id}' decision: {result.decision.value} - {result.reasoning}")
            return result
        except Exception as e:
            logger.error(f"[Reasoner] Failed to reason about step '{step.id}': {e}")
            # 推理失败时，成功的observation默认CONTINUE，错误默认ABORT
            if is_error:
                return ReasoningResult(decision=StepDecision.ABORT, reasoning=f"Reasoning failed: {e}")
            return ReasoningResult(decision=StepDecision.CONTINUE, reasoning=f"Reasoning failed, defaulting to continue: {e}")
    
    def inject_prior_results(self, step: Step, plan: Plan) -> dict:
        """将前序步骤结果注入当前步骤参数。
        
        支持占位符语法：$step_<id>.result
        例如: {"input_file": "$step_1.result"} -> {"input_file": "actual_result_from_step_1"}
        """
        if not step.params:
            return {}
        
        params = dict(step.params)
        step_results = plan.step_results
        
        for key, value in params.items():
            if isinstance(value, str):
                params[key] = self._replace_placeholders(value, step_results)
        
        # 同时合并 step.context 中的动态参数
        if step.context:
            for k, v in step.context.items():
                if k not in params:
                    params[k] = v
        
        return params
    
    def _replace_placeholders(self, text: str, step_results: dict) -> str:
        """替换文本中的步骤结果占位符。"""
        pattern = r'\$step_(\w+)\.result'
        
        def replacer(match):
            step_id = match.group(1)
            if step_id in step_results:
                return step_results[step_id]
            logger.warning(f"[Reasoner] Placeholder $step_{step_id}.result not found in completed results")
            return match.group(0)  # 保留原文
        
        return re.sub(pattern, replacer, text)
    
    def _build_completed_summary(self, plan: Plan) -> str:
        """构建已完成步骤的摘要。"""
        completed = [s for s in plan.steps if s.status in (StepStatus.COMPLETED, StepStatus.SKIPPED)]
        if not completed:
            return "No steps completed yet."
        
        lines = []
        for s in completed:
            status_label = "COMPLETED" if s.status == StepStatus.COMPLETED else "SKIPPED"
            result_preview = (s.result or "")[:200]
            lines.append(f"- [{status_label}] Step {s.id} ({s.skill_name}): {result_preview}")
        return "\n".join(lines)
    
    def _build_reasoning_prompt(self, step, observation, is_error, completed_summary, pending_steps, 
                                tir_attempt, max_tir_loops):
        obs_label = "ERROR" if is_error else "RESULT"
        obs_truncated = observation[:2000] if observation else "(empty)"
        
        pending_desc = "\n".join([f"- Step {s.id}: {s.description} (skill: {s.skill_name})" for s in pending_steps]) if pending_steps else "No pending steps."
        
        # TIR 循环信息
        current_attempt = tir_attempt + 1  # 转换为1-based
        tir_info = f"\nThis is TIR attempt {current_attempt}/{max_tir_loops} for this step."
        is_last_attempt = (tir_attempt >= max_tir_loops - 1)
        
        if is_last_attempt:
            tir_info += "\n⚠️ This is the LAST attempt. If this fails, the step will be marked as FAILED and may trigger REPLAN/ABORT."
            retry_guidance = "RETRY is still available but this is your final chance to fix the issue."
        else:
            remaining = max_tir_loops - current_attempt
            tir_info += f"\nYou have {remaining} more attempt(s) remaining after this one."
            retry_guidance = "When in doubt, prefer RETRY with adjusted parameters to maximize success chances."
        
        prompt = f"""You are a reasoning agent in a Tool-Integrated Reasoning (TIR) system. 
A plan step has just been executed. Analyze the observation and decide the next action.

## Current Step
- ID: {step.id}
- Description: {step.description}
- Skill: {step.skill_name}
- Parameters: {json.dumps(step.params or {}, ensure_ascii=False)}
{tir_info}

## Execution {obs_label}
{obs_truncated}

## Completed Steps Summary
{completed_summary}

## Remaining Pending Steps
{pending_desc}

## Your Decision
Analyze the observation and respond with a JSON object:

```json
{{
    "decision": "CONTINUE|RETRY|SKIP|REPLAN|ABORT",
    "reasoning": "Brief explanation of your decision",
    "adjusted_params": null
}}
```

Decision guidelines:
- **CONTINUE**: The step succeeded or produced useful output. Proceed to next steps.
- **RETRY**: The step failed but could succeed with adjusted parameters. Provide "adjusted_params" with the new parameters. {retry_guidance}
- **SKIP**: The step failed but is not critical for the overall task. Downstream steps can proceed without its result.
- **REPLAN**: The failure indicates the original plan is flawed and needs restructuring.
- **ABORT**: The failure is unrecoverable and the entire plan should be terminated.

Respond ONLY with the JSON object, no other text."""

        return prompt
    
    def _parse_reasoning_response(self, response: str, is_error: bool) -> ReasoningResult:
        """解析LLM推理响应。"""
        try:
            # 尝试从response中提取JSON
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)
            
            decision_str = data.get("decision", "").upper()
            decision_map = {
                "CONTINUE": StepDecision.CONTINUE,
                "RETRY": StepDecision.RETRY,
                "SKIP": StepDecision.SKIP,
                "REPLAN": StepDecision.REPLAN,
                "ABORT": StepDecision.ABORT,
            }
            
            decision = decision_map.get(decision_str)
            if decision is None:
                logger.warning(f"[Reasoner] Unknown decision '{decision_str}', defaulting based on error status")
                decision = StepDecision.ABORT if is_error else StepDecision.CONTINUE
            
            return ReasoningResult(
                decision=decision,
                reasoning=data.get("reasoning", ""),
                adjusted_params=data.get("adjusted_params")
            )
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"[Reasoner] Failed to parse response: {e}, raw: {response[:300]}")
            return ReasoningResult(
                decision=StepDecision.ABORT if is_error else StepDecision.CONTINUE,
                reasoning=f"Parse failed: {e}"
            )
