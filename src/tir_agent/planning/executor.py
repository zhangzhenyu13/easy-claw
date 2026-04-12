"""
计划执行器

负责执行规划好的任务步骤，管理步骤状态和依赖关系。
"""

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
                            # Reasoner 建议调整参数重试 → 继续 TIR 循环
                            step.retry_count = tir_attempt + 1
                            if reasoning_result and reasoning_result.adjusted_params:
                                step.params = reasoning_result.adjusted_params
                            logger.info(f"[PlanExecutor] TIR loop {tir_attempt + 1}/{self.max_tir_loops} for step '{step.id}': retrying with adjusted params")
                            continue  # 继续下一次 TIR 循环
                                            
                        elif decision in (StepDecision.SKIP, StepDecision.REPLAN, StepDecision.ABORT):
                            # Reasoner 认为无法通过重试解决 → 但如果还有 TIR 循环余量，降级为重试
                            if tir_attempt < self.max_tir_loops - 1:
                                # 还有重试机会，记录 Reasoner 的建议但仍然尝试
                                step.retry_count = tir_attempt + 1
                                if reasoning_result and reasoning_result.adjusted_params:
                                    step.params = reasoning_result.adjusted_params
                                logger.info(f"[PlanExecutor] TIR loop {tir_attempt + 1}/{self.max_tir_loops} for step '{step.id}': "
                                           f"Reasoner suggested {decision.value}, but retrying (attempts remaining)")
                                continue
                            else:
                                # TIR 循环已耗尽，执行 Reasoner 的最终决策
                                logger.warning(f"[PlanExecutor] TIR loops exhausted for step '{step.id}' after {self.max_tir_loops} attempts")
                                break  # 跳出 TIR 循环，执行最终决策
                                    
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
