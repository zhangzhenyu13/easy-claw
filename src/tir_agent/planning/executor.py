"""
计划执行器

负责执行规划好的任务步骤，管理步骤状态和依赖关系。
"""

import json
import logging
from typing import Callable, Optional

from .models import Plan, Step, TaskResult, StepStatus, StepDecision, ReasoningResult
from tir_agent.config import settings

logger = logging.getLogger("tir_agent.planning")

# TIR (Tool-Integrated Reasoning) 循环最大次数（默认值为5，可通过配置覆盖）
MAX_TIR_LOOPS = 5


class PlanExecutor:
    """
    计划执行器
    
    执行规划好的任务步骤，支持通过回调函数与Skill和Memory系统解耦。
    集成 StepReasoner 实现 Tool-Integrated Reasoning 循环。
    """
    
    def __init__(self, reasoner=None, max_tir_loops: int = None):
        """
        初始化计划执行器
        
        Args:
            reasoner: StepReasoner 实例，可选。用于步骤执行后的 LLM 推理决策
            max_tir_loops: 单步 TIR 循环最大次数，默认从 settings 读取
        """
        self._current_plan: Optional[Plan] = None
        self.reasoner = reasoner  # StepReasoner 实例，可选
        self.max_tir_loops = max_tir_loops if max_tir_loops is not None else getattr(settings, 'max_tir_loops', MAX_TIR_LOOPS)
        logger.info("PlanExecutor initialized (max_tir_loops=%d)", self.max_tir_loops)
    
    def execute(
        self, 
        plan: Plan, 
        skill_executor: Callable[[str, dict], str], 
        on_step_complete: Optional[Callable[[Step], None]] = None,
        on_replan: Optional[Callable[[Plan, Step, str], Optional[Plan]]] = None
    ) -> TaskResult:
        """
        执行计划
        
        Args:
            plan: 要执行的计划
            skill_executor: 技能执行回调函数，接收(skill_name, params)返回结果字符串
            on_step_complete: 步骤完成回调函数，接收(step)用于记忆写入等
            on_replan: 重新规划回调函数，接收(plan, failed_step, error)返回新Plan或None
            
        Returns:
            任务执行结果
        """
        self._current_plan = plan
        logger.info("开始执行计划，共 %d 个步骤", len(plan.steps))
        
        steps_completed = 0
        steps_failed = 0
        iteration_count = 0
        max_iterations = len(plan.steps) * 3  # 防止无限循环
        
        try:
            while iteration_count < max_iterations:
                iteration_count += 1
                
                # 检查是否已完成
                if plan.is_complete():
                    logger.info("Plan completed successfully")
                    break
                
                # 检查是否有失败
                if plan.has_failed():
                    logger.warning("Plan has failed steps")
                    break
                
                # 获取可执行的步骤
                ready_steps = plan.get_ready_steps()
                
                if not ready_steps:
                    # 没有就绪步骤但计划未完成，可能是死锁
                    pending_steps = [s for s in plan.steps if s.status == StepStatus.PENDING]
                    running_steps = [s for s in plan.steps if s.status == StepStatus.RUNNING]
                    failed_steps = [s for s in plan.steps if s.status == StepStatus.FAILED]
                    
                    # 如果所有步骤都已失败或被阻塞，提前退出
                    if not running_steps and (failed_steps or not pending_steps):
                        logger.error("Deadlock detected: no progress possible")
                        if pending_steps:
                            for step in pending_steps:
                                step.status = StepStatus.FAILED
                                step.error = "Dependency deadlock - unable to satisfy dependencies"
                                steps_failed += 1
                                if on_step_complete:
                                    try:
                                        on_step_complete(step)
                                    except Exception as e:
                                        logger.error(f"Step complete callback failed: {e}")
                        break
                    
                    # 还有运行的步骤，继续等待
                    continue
                
                # 执行就绪的步骤
                for step in ready_steps:
                    logger.info("执行步骤 [%s]: %s (skill: %s)", step.id, step.description, step.skill_name)
                    step.status = StepStatus.RUNNING
                                    
                    # TIR (Tool-Integrated Reasoning) 循环
                    # 每个步骤最多尝试 MAX_TIR_LOOPS 次
                    tir_completed = False
                    final_decision = None
                    final_reasoning_result = None
                    final_observation = None
                    final_is_error = False
                                    
                    for tir_attempt in range(self.max_tir_loops):
                        # 1. 参数注入（首次用原始参数，后续用 Reasoner 调整后的参数）
                        if self.reasoner:
                            actual_params = self.reasoner.inject_prior_results(step, plan)
                        else:
                            actual_params = step.params or {}
                                        
                        # 2. 执行 skill
                        is_error = False
                        observation = None
                        try:
                            observation = skill_executor(step.skill_name, actual_params)
                        except Exception as e:
                            observation = str(e)
                            is_error = True
                            logger.warning(f"[PlanExecutor] TIR attempt {tir_attempt + 1}/{self.max_tir_loops} for step '{step.id}' failed: {observation[:200]}")
                                        
                        # 3. 如果执行成功且有 reasoner，让 reasoner 判断结果质量
                        if self.reasoner:
                            reasoning_result = self.reasoner.reason(
                                step, observation, plan, 
                                is_error=is_error,
                                tir_attempt=tir_attempt,
                                max_tir_loops=self.max_tir_loops
                            )
                            decision = reasoning_result.decision
                        else:
                            # 无 reasoner 时的默认行为：成功则继续，失败则直接标记 FAILED
                            decision = StepDecision.CONTINUE if not is_error else StepDecision.ABORT
                                        
                        # 保存最终状态（用于 TIR 循环耗尽后的处理）
                        final_decision = decision
                        final_reasoning_result = reasoning_result if self.reasoner else None
                        final_observation = observation
                        final_is_error = is_error
                        
                        # 记录本次 TIR 尝试到历史
                        attempt_record = {
                            "attempt": tir_attempt + 1,
                            "params": actual_params.copy() if isinstance(actual_params, dict) else actual_params,
                            "observation_snippet": (observation[:200] + "...") if observation and len(observation) > 200 else (observation or ""),
                            "is_error": is_error,
                            "decision": reasoning_result.decision.value if reasoning_result else None,
                            "reasoning": reasoning_result.reasoning if reasoning_result else "",
                            "confidence": reasoning_result.confidence if reasoning_result else 0.5
                        }
                        step.tir_history.append(attempt_record)
                                        
                        # 4. 根据决策处理
                        if decision == StepDecision.CONTINUE:
                            # 成功！跳出 TIR 循环
                            step.result = observation
                            step.status = StepStatus.COMPLETED
                            step.retry_count = tir_attempt
                            steps_completed += 1
                            logger.info(f"[PlanExecutor] Step '{step.id}' completed after {tir_attempt + 1} TIR attempt(s)")
                            if on_step_complete:
                                try:
                                    on_step_complete(step)
                                except Exception as e:
                                    logger.error(f"Step complete callback failed: {e}")
                            tir_completed = True
                            break
                                            
                        elif decision == StepDecision.RETRY:
                            # 检查是否需要升级为 REPLAN（模式化失败或低置信度）
                            should_replan = False
                            
                            # 模式化失败检测
                            if self._detect_pattern_failure(step.tir_history):
                                logger.warning(f"[PlanExecutor] Pattern failure detected for step '{step.id}', upgrading to REPLAN")
                                should_replan = True
                                final_decision = StepDecision.REPLAN
                            
                            # 置信度驱动的早停：置信度 < 0.3 且已重试至少 2 次
                            if (not should_replan and reasoning_result and 
                                reasoning_result.confidence < 0.3 and tir_attempt >= 2):
                                logger.warning(f"[PlanExecutor] Low confidence ({reasoning_result.confidence:.2f}) after {tir_attempt + 1} attempts, "
                                               f"upgrading to REPLAN for step '{step.id}'")
                                should_replan = True
                                final_decision = StepDecision.REPLAN
                            
                            if should_replan:
                                # 立即跳出 TIR 循环，触发重规划
                                break
                            
                            # Reasoner 建议调整参数重试 → 继续 TIR 循环
                            step.retry_count = tir_attempt + 1
                            if reasoning_result and reasoning_result.adjusted_params:
                                step.params = reasoning_result.adjusted_params
                            logger.info(f"[PlanExecutor] TIR loop {tir_attempt + 1}/{self.max_tir_loops} for step '{step.id}': retrying with adjusted params")
                            continue  # 继续下一次 TIR 循环
                                            
                        elif decision == StepDecision.SKIP:
                            # 立即标记步骤为 SKIPPED，跳出循环
                            step.status = StepStatus.SKIPPED
                            step.result = reasoning_result.reasoning if reasoning_result else "Skipped by reasoner"
                            step.error = observation if is_error else None
                            logger.info(f"[PlanExecutor] Skipping step '{step.id}' as suggested by reasoner: {step.result}")
                            if on_step_complete:
                                try:
                                    on_step_complete(step)
                                except Exception as e:
                                    logger.error(f"Step complete callback failed: {e}")
                            tir_completed = True
                            break
                        
                        elif decision == StepDecision.REPLAN:
                            # 立即跳出循环，触发重规划
                            final_decision = StepDecision.REPLAN
                            logger.info(f"[PlanExecutor] REPLAN requested by reasoner for step '{step.id}'")
                            break
                        
                        elif decision == StepDecision.ABORT:
                            # 立即跳出循环，终止计划
                            final_decision = StepDecision.ABORT
                            logger.error(f"[PlanExecutor] ABORT requested by reasoner for step '{step.id}'")
                            break
                                    
                    # TIR 循环结束后，如果步骤仍未完成，处理最终决策
                    if not tir_completed:
                        step.status = StepStatus.FAILED
                        steps_failed += 1
                                        
                        if final_decision == StepDecision.SKIP:
                            step.status = StepStatus.SKIPPED
                            step.error = final_observation if final_is_error else None
                            step.result = final_reasoning_result.reasoning if final_reasoning_result else "Skipped by reasoner"
                            logger.info(f"[PlanExecutor] Skipping step '{step.id}' after TIR exhaustion: {step.result}")
                            if on_step_complete:
                                try:
                                    on_step_complete(step)
                                except Exception as e:
                                    logger.error(f"Step complete callback failed: {e}")
                                                    
                        elif final_decision == StepDecision.REPLAN:
                            step.error = final_observation if final_is_error else "Replan requested after TIR exhaustion"
                            if on_replan:
                                try:
                                    logger.info("步骤 [%s] 失败，触发重新规划", step.id)
                                    new_plan = on_replan(plan, step, step.error)
                                    if new_plan and isinstance(new_plan, Plan):
                                        plan = new_plan
                                        self._current_plan = plan
                                        logger.info("[PlanExecutor] Plan updated via replan")
                                        break  # 跳出 for step in ready_steps，重新开始主循环
                                    else:
                                        logger.warning("重新规划返回无效结果，继续执行原计划")
                                except Exception as replan_e:
                                    logger.error("重新规划失败: %s", str(replan_e))
                            else:
                                logger.warning("[PlanExecutor] Replan requested but no on_replan callback")
                            if on_step_complete:
                                try:
                                    on_step_complete(step)
                                except Exception as e:
                                    logger.error(f"Step complete callback failed: {e}")
                                                    
                        elif final_decision == StepDecision.ABORT:
                            step.error = final_observation if final_is_error else "Aborted by reasoner after TIR exhaustion"
                            # 标记所有未完成步骤为 FAILED
                            for s in plan.steps:
                                if s.status in (StepStatus.PENDING, StepStatus.RUNNING):
                                    s.status = StepStatus.FAILED
                                    s.error = "Aborted due to critical failure in earlier step"
                            logger.error(f"[PlanExecutor] Plan aborted at step '{step.id}' after TIR exhaustion")
                            # 跳出主循环
                            iteration_count = max_iterations
                            break
                        else:
                            # 默认：TIR耗尽，标记失败，尝试 replan
                            step.error = f"Failed after {self.max_tir_loops} TIR attempts. Last error: {final_observation}"
                            if on_replan:
                                try:
                                    logger.info("步骤 [%s] TIR 循环耗尽，触发重新规划", step.id)
                                    new_plan = on_replan(plan, step, step.error)
                                    if new_plan and isinstance(new_plan, Plan):
                                        plan = new_plan
                                        self._current_plan = plan
                                        logger.info("[PlanExecutor] Plan updated via replan")
                                        break  # 跳出 for step in ready_steps，重新开始主循环
                                    else:
                                        logger.warning("重新规划返回无效结果，继续执行原计划")
                                except Exception as replan_e:
                                    logger.error("重新规划失败: %s", str(replan_e))
                            if on_step_complete:
                                try:
                                    on_step_complete(step)
                                except Exception as e:
                                    logger.error(f"Step complete callback failed: {e}")
            
            # 生成最终答案
            final_answer = self._format_final_answer(plan)
            success = plan.is_complete() and not plan.has_failed()
            
            logger.info("计划执行完成，成功: %d, 失败: %d", steps_completed, steps_failed)
            
            return TaskResult(
                success=success,
                plan=plan,
                final_answer=final_answer,
                steps_completed=steps_completed,
                steps_failed=steps_failed
            )
            
        except Exception as e:
            logger.error(f"Plan execution failed with exception: {e}")
            return TaskResult(
                success=False,
                plan=plan,
                final_answer=f"执行失败: {str(e)}",
                steps_completed=steps_completed,
                steps_failed=steps_failed
            )
    
    def _detect_pattern_failure(self, tir_history: list) -> bool:
        """检测是否出现模式化失败（连续相同类型的错误）"""
        if len(tir_history) < 2:
            return False
        
        # 获取最近的错误记录
        recent_errors = [
            h for h in tir_history[-3:]
            if h.get("is_error", False)
        ]
        
        if len(recent_errors) < 2:
            return False
        
        # 提取错误摘要并比较
        error_snippets = [h.get("observation_snippet", "")[:50] for h in recent_errors]
        # 如果错误摘要高度相似（简化判断：前50字符相同），认为是模式化失败
        if len(set(error_snippets)) == 1:
            return True
        
        return False

    def _format_final_answer(self, plan: Plan) -> str:
        """
        从已完成步骤的结果中汇总最终答案
        
        Args:
            plan: 执行完成的计划
            
        Returns:
            汇总的最终答案
        """
        results = []
        failed_steps = []
        
        for step in plan.steps:
            if step.status == StepStatus.COMPLETED and step.result:
                results.append(f"【{step.description}】\n{step.result}")
            elif step.status == StepStatus.SKIPPED:
                results.append(f"【{step.description}】\n[已跳过]")
            elif step.status == StepStatus.FAILED:
                results.append(f"【{step.description}】\n[失败: {step.error or '未知错误'}]")
                failed_steps.append(step)
        
        if not results:
            return "计划执行完成，但没有产生结果。"
        
        # 构建最终答案
        final_answer = ""
        
        # 如果有失败步骤，在开头标注
        if failed_steps:
            final_answer += "⚠️ 任务部分失败\n"
            final_answer += f"共有 {len(failed_steps)} 个步骤执行失败:\n"
            for step in failed_steps:
                final_answer += f"  - [{step.id}] {step.description}: {step.error or '未知错误'}\n"
            final_answer += "\n"
        
        # 如果只有一个结果，直接返回
        if len(results) == 1:
            result_content = results[0].split("\n", 1)[1] if "\n" in results[0] else results[0]
            final_answer += result_content
            return final_answer
        
        # 多个结果时汇总
        final_answer += "任务执行结果汇总:\n\n"
        final_answer += "\n\n".join(results)
        
        return final_answer

    def execute_react(self, query: str, tools: dict, llm_caller,
                      memory_context: str = "", max_iterations: int = 10,
                      prompt_manager=None) -> str:
        """
        ReAct（Reasoning + Acting）模式执行循环
        
        Args:
            query: 用户问题
            tools: 可用工具字典 {name: skill_instance}，每个 skill 有 execute(**params) 方法
            llm_caller: LLM 调用函数，签名: llm_caller(messages: list) -> str
                        接受 messages 列表，返回 LLM 响应文本
            memory_context: 记忆上下文文本
            max_iterations: 最大迭代次数（默认10）
            prompt_manager: PromptManager 实例（可选）
        
        Returns:
            最终答案字符串
        """
        logger.info("=" * 50)
        logger.info("[ReAct] 开始 ReAct 执行循环")
        logger.info(f"[ReAct] 用户问题: {query[:100]}...")
        logger.info(f"[ReAct] 可用工具: {list(tools.keys())}")
        logger.info(f"[ReAct] 最大迭代次数: {max_iterations}")
        
        # 1. 构建工具描述
        tools_description = self._build_tools_description(tools)
        
        # 2. 构建系统提示
        if prompt_manager:
            system_prompt = prompt_manager.get(
                "react_system",
                tools_description=tools_description,
                memory_context=memory_context or "无相关历史记忆"
            )
        else:
            system_prompt = f"你是一个智能助手。可用工具:\n{tools_description}"
        
        # 3. 构建初始 messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        # 4. ReAct 历史记录（用于调试和日志）
        react_history = []
        
        # 5. 迭代循环
        for iteration in range(max_iterations):
            logger.info(f"[ReAct] --- 迭代 {iteration + 1}/{max_iterations} ---")
            
            # 调用 LLM
            try:
                response = llm_caller(messages)
            except Exception as e:
                logger.error(f"[ReAct] LLM 调用失败: {e}")
                return f"LLM 调用失败: {str(e)}"
            
            logger.info(f"[ReAct] LLM 响应: {response[:200]}...")
            
            # 解析 ReAct 输出
            parsed = self.reasoner.parse_react_output(response)
            
            # 记录历史
            react_history.append({
                "iteration": iteration + 1,
                "type": parsed["type"],
                "thought": parsed.get("thought", ""),
                "action": parsed.get("action", ""),
                "action_input": parsed.get("action_input", {}),
            })
            
            # 处理 Final Answer
            if parsed["type"] == "final_answer":
                answer = parsed.get("answer", "")
                logger.info("=" * 50)
                logger.info(f"[ReAct] 得到最终答案，共 {iteration + 1} 轮迭代")
                logger.info(f"[ReAct] 答案长度: {len(answer)} 字符")
                logger.info("=" * 50)
                return answer
            
            # 处理 Action
            if parsed["type"] == "action":
                action_name = parsed["action"]
                action_input = parsed.get("action_input", {})
                thought = parsed.get("thought", "")
                
                logger.info(f"[ReAct] Thought: {thought[:150]}")
                logger.info(f"[ReAct] Action: {action_name}")
                logger.info(f"[ReAct] Action Input: {json.dumps(action_input, ensure_ascii=False)[:200]}")
                
                # 执行工具
                observation = self._execute_tool(action_name, action_input, tools)
                
                logger.info(f"[ReAct] Observation: {observation[:200]}...")
                
                # 记录观察到历史
                react_history[-1]["observation"] = observation[:500]
                
                # 将本轮结果追加到 messages
                # Assistant 的输出（Thought + Action）
                messages.append({
                    "role": "assistant",
                    "content": response
                })
                # 工具执行结果作为新的 user 消息
                messages.append({
                    "role": "user",
                    "content": f"Observation: {observation}\n\n请继续你的推理。如果你已经有足够信息回答问题，请输出 Final Answer。"
                })
                
                continue
            
            # 未知类型，当作 final answer
            logger.warning(f"[ReAct] 未知输出类型: {parsed['type']}")
            return parsed.get("answer", response)
        
        # 超过最大迭代次数，强制要求输出最终答案
        logger.warning(f"[ReAct] 达到最大迭代次数 {max_iterations}，请求最终答案")
        messages.append({
            "role": "user",
            "content": "你已经进行了足够多的工具调用。请现在立即输出 Final Answer，总结你的所有发现并回答用户的问题。"
        })
        
        try:
            final_response = llm_caller(messages)
            final_parsed = self.reasoner.parse_react_output(final_response)
            return final_parsed.get("answer", final_response)
        except Exception as e:
            logger.error(f"[ReAct] 最终答案获取失败: {e}")
            # 汇总之前的观察作为答案
            observations = [h.get("observation", "") for h in react_history if h.get("observation")]
            if observations:
                return "基于工具调用结果的汇总:\n\n" + "\n\n".join(observations)
            return f"执行超时，无法得到最终答案: {str(e)}"

    def _build_tools_description(self, tools: dict) -> str:
        """构建工具描述文本"""
        lines = []
        for name, skill in tools.items():
            # 尝试获取技能的描述信息
            desc = ""
            params_desc = ""
            
            if hasattr(skill, 'metadata') and skill.metadata:
                desc = getattr(skill.metadata, 'description', '') or ''
                # 尝试获取参数描述
                params = getattr(skill.metadata, 'parameters', None)
                if params:
                    if isinstance(params, dict):
                        param_items = []
                        for pname, pinfo in params.items():
                            if isinstance(pinfo, dict):
                                ptype = pinfo.get('type', 'string')
                                pdesc = pinfo.get('description', '')
                                required = pinfo.get('required', False)
                                req_mark = " (必需)" if required else " (可选)"
                                param_items.append(f"    - {pname} ({ptype}){req_mark}: {pdesc}")
                            else:
                                param_items.append(f"    - {pname}: {pinfo}")
                        if param_items:
                            params_desc = "\n  参数:\n" + "\n".join(param_items)
                    elif isinstance(params, list):
                        param_items = [f"    - {p}" for p in params]
                        params_desc = "\n  参数:\n" + "\n".join(param_items)
            elif hasattr(skill, 'description'):
                desc = skill.description or ''
            
            lines.append(f"- **{name}**: {desc}{params_desc}")
        
        if not lines:
            return "（无可用工具）"
        
        return "\n".join(lines)

    def _execute_tool(self, action_name: str, action_input: dict, tools: dict) -> str:
        """执行指定工具并返回结果"""
        if action_name not in tools:
            available = ", ".join(tools.keys())
            return f"错误: 工具 '{action_name}' 不存在。可用工具: {available}"
        
        skill = tools[action_name]
        try:
            if isinstance(action_input, dict):
                result = skill.execute(**action_input)
            else:
                result = skill.execute(input=str(action_input))
            return str(result) if result else "(工具执行成功但无输出)"
        except Exception as e:
            error_msg = f"工具执行错误 [{action_name}]: {type(e).__name__}: {str(e)}"
            logger.error(f"[ReAct] {error_msg}")
            return error_msg

    def execute_planned_react(self, plan, tools, llm_caller,
                              prompt_manager=None, on_step_start=None,
                              on_step_complete=None, on_replan=None,
                              max_react_per_step=5, max_replan_count=3):
        """
        双层架构执行引擎：按照 Plan 的步骤顺序执行，每个步骤使用 ReAct 模式
        
        Args:
            plan: Plan 对象（包含 steps 列表）
            tools: 可用工具字典 {name: skill_instance}
            llm_caller: LLM 调用函数，签名: llm_caller(messages: list) -> str
            prompt_manager: PromptManager 实例
            on_step_start: 回调 (step, plan) -> None，步骤开始时调用
            on_step_complete: 回调 (step, plan) -> None，步骤完成/失败时调用
            on_replan: 回调 (old_plan, new_plan, failed_step) -> Plan，重规划时调用
                       返回新的 Plan 对象，如果返回 None 则终止执行
            max_react_per_step: 每个步骤的 ReAct 最大迭代次数
            max_replan_count: 最大重规划次数
        
        Returns:
            最终答案字符串
        """
        logger.info("=" * 50)
        logger.info("[Planning+ReAct] 开始双层架构执行")
        logger.info(f"[Planning+ReAct] 计划共 {len(plan.steps)} 个步骤")
        logger.info(f"[Planning+ReAct] 可用工具: {list(tools.keys())}")
        logger.info("=" * 50)
        
        replan_count = 0
        max_iterations = len(plan.steps) * 3  # 防止死循环
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            ready_steps = plan.get_ready_steps()
            
            if not ready_steps:
                # 检查是否所有步骤都已完成
                if plan.is_complete():
                    break
                # 检查死锁（有 pending 步骤但无 ready 步骤）
                pending = [s for s in plan.steps if s.status == StepStatus.PENDING]
                if pending:
                    logger.error("[Planning+ReAct] 检测到死锁")
                    break
                break
            
            for step in ready_steps:
                # 回调：步骤开始
                if on_step_start:
                    try:
                        on_step_start(step, plan)
                    except Exception as e:
                        logger.error(f"[Planning+ReAct] Step start callback failed: {e}")
                
                step.status = StepStatus.RUNNING
                logger.info(f"[Planning+ReAct] 执行步骤 [{step.id}]: {step.description}")
                
                # 构建前序步骤结果
                prior_results = self._build_prior_results(plan)
                
                # 用 ReAct 执行单个步骤
                success, result, error = self._execute_step_react(
                    step=step,
                    tools=tools,
                    llm_caller=llm_caller,
                    prompt_manager=prompt_manager,
                    prior_results=prior_results,
                    max_iterations=max_react_per_step
                )
                
                if success:
                    step.status = StepStatus.COMPLETED
                    step.result = result
                    logger.info(f"[Planning+ReAct] 步骤 [{step.id}] 执行成功")
                else:
                    step.status = StepStatus.FAILED
                    step.error = error
                    logger.warning(f"[Planning+ReAct] 步骤 [{step.id}] 执行失败: {error}")
                    
                    # 尝试重规划
                    if replan_count < max_replan_count and on_replan:
                        logger.info(f"[Planning+ReAct] 触发重规划 ({replan_count + 1}/{max_replan_count})")
                        try:
                            new_plan = on_replan(plan, None, step)
                            if new_plan:
                                plan = new_plan
                                replan_count += 1
                                break  # 跳出 for 循环，用新 plan 重新开始 while 循环
                            else:
                                logger.warning("[Planning+ReAct] 重规划失败，继续执行剩余步骤")
                        except Exception as replan_e:
                            logger.error(f"[Planning+ReAct] 重规划回调异常: {replan_e}")
                    else:
                        logger.warning("[Planning+ReAct] 已达最大重规划次数或无重规划回调")
                
                # 回调：步骤完成
                if on_step_complete:
                    try:
                        on_step_complete(step, plan)
                    except Exception as e:
                        logger.error(f"[Planning+ReAct] Step complete callback failed: {e}")
        
        # 汇总最终答案
        return self._format_planned_react_answer(plan)

    def _execute_step_react(self, step, tools, llm_caller, prompt_manager=None,
                            prior_results="", max_iterations=5):
        """
        用 ReAct 模式执行单个步骤
        
        Args:
            step: Step 对象
            tools: 可用工具字典 {name: skill_instance}
            llm_caller: LLM 调用函数
            prompt_manager: PromptManager 实例
            prior_results: 前序步骤结果摘要
            max_iterations: 最大迭代次数
        
        Returns:
            (success: bool, result: str, error: str)
        """
        logger.info(f"[StepReAct] 开始执行步骤 [{step.id}]")
        
        # 构建工具描述
        tools_description = self._build_tools_description(tools)
        
        # 安全获取 expected_output
        step_expected_output = getattr(step, 'expected_output', step.description)
        
        # 构建系统提示
        if prompt_manager:
            system_prompt = prompt_manager.get(
                "react_step",
                step_description=step.description,
                step_expected_output=step_expected_output,
                tools_description=tools_description,
                prior_results=prior_results,
                step_skill_hint=step.skill_name,
                memory_context="（无）"  # 后续可扩展
            )
        else:
            system_prompt = f"""你是一个智能助手，正在执行计划中的一个步骤。

步骤描述: {step.description}
期望输出: {step_expected_output}
推荐工具: {step.skill_name}

可用工具:
{tools_description}

前序步骤结果:
{prior_results}

请完成上述步骤描述的任务。使用 ReAct 模式：Thought + Action 或 Thought + Final Answer。"""
        
        # 构建 messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "请完成上述步骤"}
        ]
        
        # ReAct 小循环
        for iteration in range(max_iterations):
            logger.info(f"[StepReAct] 步骤 [{step.id}] 迭代 {iteration + 1}/{max_iterations}")
            
            try:
                response = llm_caller(messages)
            except Exception as e:
                logger.error(f"[StepReAct] LLM 调用失败: {e}")
                return False, "", f"LLM 调用失败: {str(e)}"
            
            logger.info(f"[StepReAct] LLM 响应: {response[:200]}...")
            
            # 解析 ReAct 输出
            parsed = self.reasoner.parse_react_output(response)
            
            # 处理 Final Answer
            if parsed["type"] == "final_answer":
                answer = parsed.get("answer", "")
                logger.info(f"[StepReAct] 步骤 [{step.id}] 完成，得到最终答案")
                return True, answer, ""
            
            # 处理 Action
            if parsed["type"] == "action":
                action_name = parsed["action"]
                action_input = parsed.get("action_input", {})
                thought = parsed.get("thought", "")
                
                logger.info(f"[StepReAct] Thought: {thought[:150]}")
                logger.info(f"[StepReAct] Action: {action_name}")
                logger.info(f"[StepReAct] Action Input: {json.dumps(action_input, ensure_ascii=False)[:200]}")
                
                # 执行工具
                observation = self._execute_tool(action_name, action_input, tools)
                
                logger.info(f"[StepReAct] Observation: {observation[:200]}...")
                
                # 将本轮结果追加到 messages
                messages.append({
                    "role": "assistant",
                    "content": response
                })
                messages.append({
                    "role": "user",
                    "content": f"Observation: {observation}\n\n请继续你的推理。如果你已经有足够信息完成这个步骤，请输出 Final Answer。"
                })
                
                continue
            
            # 未知类型，当作 final answer
            logger.warning(f"[StepReAct] 未知输出类型: {parsed['type']}")
            return True, parsed.get("answer", response), ""
        
        # 超过迭代次数，强制要求最终答案
        logger.warning(f"[StepReAct] 步骤 [{step.id}] 达到最大迭代次数 {max_iterations}，请求最终答案")
        messages.append({
            "role": "user",
            "content": "你已经进行了足够多的工具调用。请现在立即输出 Final Answer，总结这个步骤的执行结果。"
        })
        
        try:
            final_response = llm_caller(messages)
            final_parsed = self.reasoner.parse_react_output(final_response)
            answer = final_parsed.get("answer", final_response)
            return True, answer, ""
        except Exception as e:
            logger.error(f"[StepReAct] 最终答案获取失败: {e}")
            return False, "", f"ReAct 执行超时: {str(e)}"

    def _build_prior_results(self, plan):
        """构建已完成步骤的结果摘要"""
        lines = []
        for step in plan.steps:
            if step.status == StepStatus.COMPLETED and step.result:
                result_preview = step.result[:500] if len(step.result) > 500 else step.result
                lines.append(f"[{step.id}] {step.description}:")
                lines.append(f"  结果: {result_preview}")
                lines.append("")
        return "\n".join(lines) if lines else "（无已完成的前序步骤）"

    def _format_planned_react_answer(self, plan):
        """汇总所有步骤结果为最终答案"""
        completed = [s for s in plan.steps if s.status == StepStatus.COMPLETED]
        failed = [s for s in plan.steps if s.status == StepStatus.FAILED]
        
        if not completed:
            return "任务执行失败，没有步骤成功完成。"
        
        # 如果只有一个成功步骤，直接返回其结果
        if len(completed) == 1 and not failed:
            return completed[0].result
        
        # 多步骤：汇总
        lines = []
        for step in completed:
            lines.append(f"### {step.description}")
            lines.append(step.result or "（无输出）")
            lines.append("")
        
        if failed:
            lines.append("---")
            lines.append(f"注意：有 {len(failed)} 个步骤未能完成。")
        
        return "\n".join(lines)
