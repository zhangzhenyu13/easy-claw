"""Step-level Reasoner for Tool-Integrated Reasoning.

每个步骤执行后，由 LLM 审查 observation 并决定下一步行动。
"""
import json
import logging
import re
from typing import Optional, Callable, Any

from .models import Step, Plan, StepDecision, ReasoningResult, StepStatus

logger = logging.getLogger(__name__)

# 导入PromptManager，用于类型提示
try:
    from ..prompt_manager import PromptManager
except ImportError:
    PromptManager = None


class StepReasoner:
    """步骤间 LLM 推理器，实现 Tool-Integrated Reasoning。"""
    
    def __init__(self, llm_caller: Callable, prompt_manager: Optional[Any] = None):
        """
        Args:
            llm_caller: 调用LLM的函数，签名为 (messages: list[dict]) -> str
                       接收消息列表，返回LLM文本响应
            prompt_manager: PromptManager实例（可选）
        """
        self.llm_caller = llm_caller
        
        # 初始化PromptManager（如果未提供则创建默认实例）
        if prompt_manager is None and PromptManager is not None:
            from ..config import settings
            prompt_manager = PromptManager(
                prompts_dir=settings.prompts_dir,
                version=settings.prompt_version
            )
        self.prompt_manager = prompt_manager
    
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
    
    def _build_adjustment_history(self, step) -> str:
        """将步骤的 TIR 历史格式化为文本，供 Prompt 使用"""
        if not hasattr(step, 'tir_history') or not step.tir_history:
            return "No previous attempts for this step."
        
        lines = []
        for record in step.tir_history:
            attempt = record.get("attempt", "?")
            decision = record.get("decision", "?")
            reasoning = record.get("reasoning", "")
            confidence = record.get("confidence", 0)
            obs = record.get("observation_snippet", "")
            is_error = record.get("is_error", False)
            
            lines.append(f"Attempt {attempt}:")
            lines.append(f"  - Result: {'ERROR' if is_error else 'OK'} - {obs}")
            lines.append(f"  - Decision: {decision} (confidence: {confidence:.1%})")
            if reasoning:
                lines.append(f"  - Reasoning: {reasoning[:150]}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _truncate_observation(self, observation: str, is_error: bool, max_length: int = 2000) -> str:
        """智能截断观察结果"""
        if not observation or len(observation) <= max_length:
            return observation or "(empty)"
        
        if is_error:
            # 错误信息：保留末尾（traceback关键信息在最后）
            head_size = 500
            tail_size = max_length - head_size - 20  # 20 for separator
            return observation[:head_size] + "\n...(truncated)...\n" + observation[-tail_size:]
        else:
            # 正常结果：保留开头
            return observation[:max_length - 15] + "\n...(truncated)"
    
    def _build_reasoning_prompt(self, step, observation, is_error, completed_summary, pending_steps, 
                                tir_attempt, max_tir_loops):
        obs_label = "ERROR" if is_error else "RESULT"
        obs_truncated = self._truncate_observation(observation, is_error)
        
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
        
        # 构建调整历史
        adjustment_history = self._build_adjustment_history(step)
        
        # 使用PromptManager获取prompt
        if self.prompt_manager:
            prompt = self.prompt_manager.get(
                "reasoning_decision",
                step_id=step.id,
                step_description=step.description,
                step_skill_name=step.skill_name,
                step_params=json.dumps(step.params or {}, ensure_ascii=False),
                tir_info=tir_info,
                obs_label=obs_label,
                obs_truncated=obs_truncated,
                adjustment_history=adjustment_history,
                completed_summary=completed_summary,
                pending_desc=pending_desc,
                retry_guidance=retry_guidance
            )
        else:
            # Fallback: 使用硬编码prompt
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
        """解析 LLM 推理响应为 ReasoningResult"""
        from tir_agent.planning.models import ReasoningResult, StepDecision

        try:
            # Step 1: 清理 Markdown fence
            cleaned = response.strip()
            # 去除 ```json ... ``` 或 ``` ... ```
            cleaned = re.sub(r'^```(?:json)?\s*\n?', '', cleaned)
            cleaned = re.sub(r'\n?```\s*$', '', cleaned)
            cleaned = cleaned.strip()

            # Step 2: 尝试直接解析整个清理后的文本
            data = None
            try:
                data = json.loads(cleaned)
            except json.JSONDecodeError:
                pass

            # Step 3: 如果直接解析失败，使用栈匹配法提取最大的完整 JSON 对象
            if data is None:
                data = self._extract_json_by_bracket_matching(cleaned)

            # Step 4: 如果仍然失败，返回默认结果
            if data is None:
                logger.warning(f"[Reasoner] Failed to parse response, raw: {response[:200]}")
                return ReasoningResult(
                    decision=StepDecision.RETRY,
                    reasoning="Failed to parse LLM response",
                    confidence=0.3
                )

            # Step 5: 提取字段
            decision_str = data.get("decision", "").upper().strip()
            decision = self._map_decision(decision_str)

            return ReasoningResult(
                decision=decision,
                reasoning=data.get("reasoning", ""),
                adjusted_params=data.get("adjusted_params"),
                observation_analysis=data.get("observation_analysis", ""),
                root_cause=data.get("root_cause", ""),
                recovery_strategy=data.get("recovery_strategy", ""),
                confidence=float(data.get("confidence", 0.5)),
                lessons_learned=data.get("lessons_learned", ""),
            )

        except Exception as e:
            logger.error(f"[Reasoner] Error parsing response: {e}")
            return ReasoningResult(
                decision=StepDecision.RETRY,
                reasoning=f"Parse error: {str(e)}",
                confidence=0.3
            )

    def _extract_json_by_bracket_matching(self, text: str) -> dict:
        """使用栈匹配法从文本中提取最大的完整 JSON 对象"""
        # 找到第一个 { 的位置
        start = -1
        for i, ch in enumerate(text):
            if ch == '{':
                start = i
                break

        if start == -1:
            return None

        # 栈匹配，处理字符串内的花括号
        depth = 0
        in_string = False
        escape_next = False

        for i in range(start, len(text)):
            ch = text[i]

            if escape_next:
                escape_next = False
                continue

            if ch == '\\' and in_string:
                escape_next = True
                continue

            if ch == '"' and not escape_next:
                in_string = not in_string
                continue

            if in_string:
                continue

            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    # 找到了完整的 JSON 对象
                    json_str = text[start:i+1]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        return None

        return None

    def _map_decision(self, decision_str: str) -> 'StepDecision':
        """将决策字符串映射为 StepDecision 枚举"""
        from tir_agent.planning.models import StepDecision

        mapping = {
            "CONTINUE": StepDecision.CONTINUE,
            "RETRY": StepDecision.RETRY,
            "SKIP": StepDecision.SKIP,
            "REPLAN": StepDecision.REPLAN,
            "ABORT": StepDecision.ABORT,
        }

        result = mapping.get(decision_str)
        if result is None:
            logger.warning(f"[Reasoner] Unknown decision '{decision_str}', defaulting to RETRY")
            return StepDecision.RETRY
        return result

    def parse_react_output(self, text: str) -> dict:
        """
        解析 ReAct 格式输出

        Returns:
            - {"type": "action", "thought": "...", "action": "工具名", "action_input": {...}}
            - {"type": "final_answer", "thought": "...", "answer": "..."}
            - {"type": "error", "raw": "..."} 如果解析失败
        """
        text = text.strip()

        # 提取 Thought
        thought = ""
        thought_match = re.search(r'Thought:\s*(.*?)(?=\n(?:Action:|Final Answer:)|$)', text, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()

        # 检查是否是 Final Answer
        final_match = re.search(r'Final Answer:\s*(.*)', text, re.DOTALL)
        if final_match:
            return {
                "type": "final_answer",
                "thought": thought,
                "answer": final_match.group(1).strip()
            }

        # 检查是否是 Action
        action_match = re.search(r'Action:\s*(.*?)(?:\n|$)', text)
        if action_match:
            action_name = action_match.group(1).strip()

            # 提取 Action Input
            action_input = {}
            input_match = re.search(r'Action Input:\s*(.*)', text, re.DOTALL)
            if input_match:
                input_text = input_match.group(1).strip()
                # 清理 markdown fence
                input_text = re.sub(r'^```(?:json)?\s*\n?', '', input_text)
                input_text = re.sub(r'\n?```\s*$', '', input_text)
                input_text = input_text.strip()

                # 尝试解析 JSON
                try:
                    action_input = json.loads(input_text)
                except json.JSONDecodeError:
                    # 尝试栈匹配
                    extracted = self._extract_json_by_bracket_matching(input_text)
                    if extracted:
                        action_input = extracted
                    else:
                        # 作为纯文本参数
                        action_input = {"input": input_text}

            return {
                "type": "action",
                "thought": thought,
                "action": action_name,
                "action_input": action_input
            }

        # 如果既没有 Action 也没有 Final Answer
        # 可能 LLM 直接输出了答案文本
        if thought:
            return {
                "type": "final_answer",
                "thought": thought,
                "answer": thought
            }

        # 完全无法解析
        return {
            "type": "final_answer",
            "thought": "",
            "answer": text
        }
