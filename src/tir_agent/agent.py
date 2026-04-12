"""
TIR Agent核心模块
Tool-Integrated Reasoning Agent基于Qwen-Agent实现
集成Memory、Skills、Planning三个子系统
"""
import os
import json
import uuid
import logging
from typing import List, Dict, Any, Optional, Union, Generator

from .config import settings, Settings

# 导入三个子系统
from .memory import MemoryManager
from .skills import SkillRegistry, BaseSkill
from .planning import TaskPlanner, PlanExecutor
from .planning.reasoner import StepReasoner
from .context_manager import ContextManager
from .prompt_manager import PromptManager

# 配置日志
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def format_file_context(processed_files: List[dict]) -> str:
    """
    将处理后的文件信息格式化为上下文字符串
    
    Args:
        processed_files: 处理后的文件列表
        
    Returns:
        str: 格式化后的上下文字符串
    """
    context_parts = []
    
    for pf in processed_files:
        header = f"\n--- 文件: {pf['file_name']} (类型: {pf['file_type']}) ---\n"
        
        if pf.get('error'):
            context_parts.append(f"{header}[处理错误: {pf['error']}]")
            continue
        
        content = pf.get('content', '')
        if content:
            context_parts.append(f"{header}{content}")
        
        if pf.get('images'):
            img_count = len(pf['images'])
            context_parts.append(f"{header}[包含 {img_count} 张图片]")
    
    return "\n".join(context_parts)


class TIRAgent:
    """
    Tool-Integrated Reasoning Agent
    
    支持多文件输入、代码执行、VLM图像理解的智能Agent
    集成Memory记忆系统、Skills动态工具系统、Planning任务规划系统
    """
    
    def __init__(
        self,
        llm_config: Optional[Dict[str, Any]] = None,
        vlm_config: Optional[Dict[str, Any]] = None,
        system_message: Optional[str] = None,
        tools: Optional[List[str]] = None,
        memory_enabled: bool = True,
        planning_enabled: bool = True,
        skill_dirs: Optional[List[str]] = None,
        memory_compression_threshold: Optional[int] = None,
        complexity_threshold: Optional[str] = None,
        prompt_manager: Optional[PromptManager] = None,
        **kwargs
    ):
        """
        初始化TIR Agent
        
        Args:
            llm_config: LLM配置
            vlm_config: VLM配置
            system_message: 系统提示词
            tools: 工具列表
            memory_enabled: 是否启用记忆系统
            planning_enabled: 是否启用规划系统
            skill_dirs: 额外的Skill目录列表
            prompt_manager: PromptManager实例（可选）
        """
        # 1. 加载配置（合并传入参数和Settings）
        self.settings = settings
        self.llm_config = llm_config or self.settings.get_llm_config()
        self.vlm_config = vlm_config or self.settings.get_vlm_config()
        
        # 初始化PromptManager
        self.prompt_manager = prompt_manager or PromptManager(
            prompts_dir=self.settings.prompts_dir,
            version=self.settings.prompt_version
        )
        
        self.system_message = system_message or self.prompt_manager.get("system")
        self.tools = tools or ["code_interpreter", "vlm_analyzer"]
        
        # 保存构造参数，优先于settings
        self._compression_threshold = memory_compression_threshold or self.settings.memory_compression_threshold
        self._complexity_threshold = complexity_threshold or self.settings.complexity_threshold
        
        # 读取ReAct执行模式配置
        self._execution_mode = self.settings.execution_mode  # "auto", "react" 或 "planning"
        self._react_max_iterations = self.settings.react_max_iterations
        self._max_react_per_step = self.settings.max_react_per_step
        
        # 2. 初始化SkillRegistry，发现并加载builtin Skills + 自定义Skills
        self.skill_registry = SkillRegistry()
        self._init_skills(skill_dirs)
        
        # 3. 初始化MemoryManager（如果memory_enabled）
        self.memory_manager: Optional[MemoryManager] = None
        self._memory_enabled = memory_enabled and self.settings.memory_enabled
        if self._memory_enabled:
            self._init_memory()
        
        # 4. 初始化TaskPlanner（如果planning_enabled）
        self.planner: Optional[TaskPlanner] = None
        self.step_reasoner: Optional[StepReasoner] = None
        self._planning_enabled = planning_enabled and self.settings.planning_enabled
        if self._planning_enabled:
            self._init_planner()
        
        # 创建 PlanExecutor，如果启用了 planning 则传入 reasoner
        if self._planning_enabled and self.planner:
            self.step_reasoner = StepReasoner(
                llm_caller=self._llm_call_for_reasoner,
                prompt_manager=self.prompt_manager
            )
            self.plan_executor = PlanExecutor(
                reasoner=self.step_reasoner,
                max_tir_loops=self.settings.max_tir_loops
            )
            logger.info("PlanExecutor initialized with StepReasoner")
        else:
            self.step_reasoner = None
            self.plan_executor = PlanExecutor(max_tir_loops=self.settings.max_tir_loops) if self._planning_enabled else None
        
        # 5. 初始化后从 registry 获取核心 Skill 实例
        self.file_processor = self.skill_registry.get("file_processor")
        self.code_executor = self.skill_registry.get("code_executor")
        self.vlm_tool = self.skill_registry.get("vlm_analyzer")
        
        if not self.file_processor:
            logger.warning("file_processor skill 未加载")
        if not self.code_executor:
            logger.warning("code_executor skill 未加载")
        if not self.vlm_tool:
            logger.warning("vlm_analyzer skill 未加载")
        
        # 6. 初始化ContextManager
        try:
            self.context_manager = ContextManager(
                max_tokens=self.settings.max_context_tokens if hasattr(self.settings, 'max_context_tokens') else 30000,
                prompt_manager=self.prompt_manager
            )
            logger.info("ContextManager初始化完成")
        except Exception as e:
            logger.error(f"ContextManager初始化失败: {e}")
            self.context_manager = None
        
        # 7. 初始化Qwen-Agent（使用registry.to_qwen_tools()获取工具列表）
        self._qwen_agent = None
        self._init_qwen_agent()
        
        # 当前session_id（用于Memory记录）
        self._current_session_id: Optional[str] = None
        
        logger.info(f"TIRAgent初始化完成: memory={self._memory_enabled}, planning={self._planning_enabled}")
    
    def _get_builtin_skill_dirs(self) -> list[str]:
        """获取builtin Skills目录路径（已弃用，返回空列表）"""
        # builtin skills 已迁移到项目根目录的 skills/ 文件夹
        return []
    
    def _init_skills(self, extra_skill_dirs: Optional[List[str]] = None):
        """初始化Skills系统"""
        try:
            skill_dirs_to_scan = []
            
            # 从配置加载默认 skill 目录（相对路径基于 cwd 解析）
            if self.settings.skill_dirs:
                for d in self.settings.skill_dirs:
                    expanded = os.path.expanduser(d)
                    if not os.path.isabs(expanded):
                        expanded = os.path.join(os.getcwd(), expanded)
                    if os.path.isdir(expanded):
                        skill_dirs_to_scan.append(expanded)
            
            # 自定义 skill 目录
            if self.settings.custom_skill_dir:
                expanded = os.path.expanduser(self.settings.custom_skill_dir)
                if not os.path.isabs(expanded):
                    expanded = os.path.join(os.getcwd(), expanded)
                if os.path.isdir(expanded):
                    skill_dirs_to_scan.append(expanded)
            
            # 额外传入的目录
            if extra_skill_dirs:
                for dir_path in extra_skill_dirs:
                    expanded = os.path.expanduser(dir_path)
                    if not os.path.isabs(expanded):
                        expanded = os.path.join(os.getcwd(), expanded)
                    if os.path.isdir(expanded):
                        skill_dirs_to_scan.append(expanded)
            
            # 发现并注册Skills
            count = self.skill_registry.discover(skill_dirs_to_scan)
            logger.info(f"Skill系统初始化完成，共加载 {count} 个Skills")
            
        except Exception as e:
            logger.error(f"Skill系统初始化失败: {e}", exc_info=True)
    
    def _init_memory(self):
        """初始化Memory系统"""
        try:
            db_path = os.path.expanduser(self.settings.memory_db_path)
            self.memory_manager = MemoryManager(
                db_path=db_path,
                llm_config=self.llm_config,
                compression_threshold=self._compression_threshold,
                recall_top_k=self.settings.memory_recall_top_k
            )
            logger.info(f"Memory系统初始化完成，数据库: {db_path}")
        except Exception as e:
            logger.error(f"Memory系统初始化失败: {e}", exc_info=True)
            self.memory_manager = None
            self._memory_enabled = False
    
    def _init_planner(self):
        """初始化Planning系统"""
        try:
            self.planner = TaskPlanner(self.llm_config, prompt_manager=self.prompt_manager)
            logger.info("Planning系统初始化完成")
        except Exception as e:
            logger.error(f"Planning系统初始化失败: {e}", exc_info=True)
            self.planner = None
            self._planning_enabled = False
    
    def _init_qwen_agent(self):
        """初始化Qwen-Agent"""
        try:
            from qwen_agent.agents import Assistant
            from qwen_agent.tools.base import BaseTool, register_tool
            
            # 注册自定义代码执行工具（不依赖Docker）
            @register_tool('code_interpreter_tir')
            class CodeInterpreterTIR(BaseTool):
                description = 'Python代码执行工具，可以执行Python代码并返回结果。支持numpy、pandas、matplotlib等常用库。不依赖Docker，直接在本地安全执行。'
                parameters = {
                    'type': 'object',
                    'properties': {
                        'code': {
                            'type': 'string',
                            'description': '要执行的Python代码'
                        },
                        'language': {
                            'type': 'string',
                            'description': '编程语言，默认为python',
                            'enum': ['python', 'bash']
                        }
                    },
                    'required': ['code']
                }
                
                def __init__(self, code_executor, **kwargs):
                    super().__init__(**kwargs)
                    self.executor = code_executor
                
                def call(self, params: str, **kwargs) -> str:
                    try:
                        args = json.loads(params)
                    except json.JSONDecodeError:
                        args = {"code": params}
                    
                    code = args.get("code", "")
                    language = args.get("language", "python")
                    
                    result = self.executor.execute(code, language)
                    return self.executor.format_result(result)
            
            # 注册自定义VLM工具
            @register_tool('vlm_analyzer')
            class VLMAnalyzerTool(BaseTool):
                description = 'VLM图像分析工具，用于分析图片内容、OCR识别、图表数据提取等。'
                parameters = {
                    'type': 'object',
                    'properties': {
                        'action': {
                            'type': 'string',
                            'description': '操作类型：analyze(分析), ocr(OCR识别), describe(描述), extract_chart(提取图表数据)',
                            'enum': ['analyze', 'ocr', 'describe', 'extract_chart']
                        },
                        'image_path': {
                            'type': 'string',
                            'description': '图片路径或base64编码'
                        },
                        'prompt': {
                            'type': 'string',
                            'description': '分析提示词（可选，用于analyze操作）'
                        }
                    },
                    'required': ['action', 'image_path']
                }
                
                def __init__(self, vlm_tool, **kwargs):
                    super().__init__(**kwargs)
                    self.vlm = vlm_tool
                
                def call(self, params: str, **kwargs) -> str:
                    try:
                        args = json.loads(params)
                    except json.JSONDecodeError:
                        return json.dumps({"error": "参数解析失败"})
                    
                    action = args.get("action", "analyze")
                    image_path = args.get("image_path", "")
                    prompt = args.get("prompt", "请分析这张图片")
                    
                    try:
                        if action == "analyze":
                            result = self.vlm.analyze_image(image_path, prompt)
                        elif action == "ocr":
                            result = self.vlm.ocr_image(image_path)
                        elif action == "describe":
                            result = self.vlm.describe_image(image_path)
                        elif action == "extract_chart":
                            result = self.vlm.extract_chart_data(image_path)
                        else:
                            result = f"未知操作: {action}"
                        
                        return json.dumps({"result": result}, ensure_ascii=False)
                    except Exception as e:
                        return json.dumps({"error": str(e)}, ensure_ascii=False)
            
            # 创建工具实例
            code_tool_instance = CodeInterpreterTIR(code_executor=self.code_executor)
            vlm_tool_instance = VLMAnalyzerTool(vlm_tool=self.vlm_tool)
            
            # 构建工具列表
            tool_list = []
            for tool_name in self.tools:
                if tool_name == "code_interpreter":
                    tool_list.append(code_tool_instance)
                elif tool_name == "vlm_analyzer":
                    tool_list.append(vlm_tool_instance)
            
            # 添加Skills系统提供的工具（过滤掉已在tools列表中手动注册的工具，避免重复）
            manually_registered_tools = set(self.tools)
            skill_tools = self.skill_registry.to_qwen_tools()
            filtered_skill_tools = [
                tool for tool in skill_tools 
                if getattr(tool, 'name', None) not in manually_registered_tools
            ]
            tool_list.extend(filtered_skill_tools)
            logger.info(f"Qwen-Agent加载了 {len(filtered_skill_tools)} 个Skill工具（过滤了 {len(skill_tools) - len(filtered_skill_tools)} 个重复工具）")
            
            # 创建Assistant agent
            self._qwen_agent = Assistant(
                llm=self.llm_config,
                system_message=self.system_message,
                function_list=tool_list,
            )
            
            logger.info("Qwen-Agent初始化成功（使用自定义CodeExecutor和Skills系统）")
            
        except ImportError as e:
            logger.warning(f"Qwen-Agent导入失败: {e}，使用简化模式")
            self._qwen_agent = None
        except Exception as e:
            logger.error(f"Qwen-Agent初始化失败: {e}")
            self._qwen_agent = None
    
    def _llm_call_for_reasoner(self, messages: list) -> str:
        """适配 qwen-agent LLM 调用为 StepReasoner 所需的接口。
        
        StepReasoner 需要一个 (messages: list[dict]) -> str 的 callable。
        此方法将 qwen-agent 的 LLM 调用适配为该接口。
        
        Args:
            messages: 消息列表，每个消息是包含 role 和 content 的字典
            
        Returns:
            str: LLM 的文本响应
        """
        try:
            # 使用 OpenAI 兼容 API 直接调用
            import openai
            
            client = openai.OpenAI(
                api_key=self.llm_config.get("api_key") or self.llm_config.get("dashscope_api_key"),
                base_url=self.llm_config.get("base_url") or self.llm_config.get("model_server")
            )
            
            model = self.llm_config.get("model", "qwen-max-latest")
            
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                top_p=0.8,
            )
            
            content = response.choices[0].message.content
            return content if content else ""
            
        except Exception as e:
            logger.error(f"[Agent] LLM call for reasoner failed: {e}")
            return ""
    
    def _build_tools_dict(self) -> dict:
        """构建工具字典供 ReAct 模式使用"""
        tools = {}
        if self.skill_registry:
            for skill_info in self.skill_registry.list_skills():
                name = skill_info.get("name")
                if name:
                    skill = self.skill_registry.get(name)
                    if skill:
                        tools[name] = skill
        return tools
    
    def _llm_caller(self, messages: list) -> str:
        """封装 LLM 调用，供 ReAct 执行引擎使用"""
        try:
            # 使用 OpenAI 兼容 API 直接调用
            import openai
            
            client = openai.OpenAI(
                api_key=self.llm_config.get("api_key") or self.llm_config.get("dashscope_api_key"),
                base_url=self.llm_config.get("base_url") or self.llm_config.get("model_server")
            )
            
            model = self.llm_config.get("model", "qwen-max-latest")
            
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                top_p=0.8,
            )
            
            content = response.choices[0].message.content
            return content if content else ""
            
        except Exception as e:
            logger.error(f"[Agent] LLM caller failed: {e}")
            return ""
    
    def _execute_skill(self, skill_name: str, params: dict) -> str:
        """供PlanExecutor回调使用的Skill执行器"""
        # 记录Skill调用
        params_summary = str(params)[:200] if params else "{}"
        logger.info("调用 Skill: %s, 参数: %s", skill_name, params_summary)
        
        skill = self.skill_registry.get(skill_name)
        if skill is None:
            logger.warning("Skill 未找到: %s", skill_name)
            raise ValueError(f"Skill '{skill_name}' not found in registry")
        
        result = skill.execute(**params)
        logger.info("Skill [%s] 执行完成，结果长度: %d", skill_name, len(result) if result else 0)
        return result
    
    def _on_step_start(self, step, plan):
        """步骤开始回调 - 打印日志"""
        logger.info(f"[Planning] 开始执行步骤 [{step.id}]: {step.description}")
        logger.info(f"[Planning]   技能: {step.skill_name}, 依赖: {step.depends_on}")

    def _on_step_complete(self, step, plan):
        """步骤完成回调 - 打印日志并记录到Memory"""
        from tir_agent.planning.models import StepStatus
        if step.status == StepStatus.COMPLETED:
            result_preview = (step.result[:100] + "...") if step.result and len(step.result) > 100 else (step.result or "")
            logger.info(f"[Planning] 步骤 [{step.id}] 执行成功: {result_preview}")
        else:
            logger.warning(f"[Planning] 步骤 [{step.id}] 执行失败: {step.error}")
        
        # 记录到Memory
        if self.memory_manager and self._current_session_id:
            try:
                self.memory_manager.store.add_tool_call(
                    session_id=self._current_session_id,
                    tool_name=step.skill_name,
                    params=step.params or {},
                    result=step.result or step.error or "",
                    duration=0.0  # 实际duration由executor内部计算
                )
            except Exception as e:
                logger.error(f"Memory记录工具调用失败: {e}")

    def _on_replan(self, old_plan, new_plan_unused, failed_step):
        """重规划回调 - 调用 planner.replan() 并返回新 Plan"""
        logger.info(f"[Planning] 触发重规划，失败步骤: [{failed_step.id}] {failed_step.description}")
        if not self.planner:
            logger.warning("[Planning] Replan requested but no planner available")
            return None
        try:
            new_plan = self.planner.replan(old_plan, failed_step, failed_step.error or "执行失败")
            if new_plan and new_plan.steps:
                logger.info(f"[Planning] 重规划成功，新计划: {len(new_plan.steps)} 个步骤")
                return new_plan
            else:
                logger.warning("[Planning] 重规划返回空计划")
                return None
        except Exception as e:
            logger.error(f"[Planning] 重规划失败: {e}")
            return None

    def _format_plan_for_display(self, plan):
        """将 Plan 格式化为 Markdown 用于展示"""
        lines = ["**执行计划**\n"]
        for i, step in enumerate(plan.steps, 1):
            deps = f" (依赖: {', '.join(step.depends_on)})" if step.depends_on else ""
            lines.append(f"{i}. **{step.description}**{deps}")
            lines.append(f"   - 工具: `{step.skill_name}`")
        return "\n".join(lines)
    
    def process_files(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        处理上传的文件
        
        Args:
            file_paths: 文件路径列表
        
        Returns:
            dict: 处理结果
        """
        logger.info(f"📂 开始处理 {len(file_paths)} 个文件...")
        processed = self.file_processor.process_files(file_paths)
        
        # 提取所有图片
        all_images = []
        for pf in processed:
            if pf.get("images"):
                logger.info(f"   📷 文件 {pf['file_name']} 包含 {len(pf['images'])} 张图片")
                for img in pf["images"]:
                    all_images.append(img["path"])
                    logger.info(f"      - 图片路径: {img['path']}")
        
        logger.info(f"✅ 文件处理完成，共提取 {len(all_images)} 张图片")
        
        return {
            "processed_files": processed,
            "context": format_file_context(processed),
            "images": all_images,
        }
    
    def run(
        self,
        messages: List[Dict[str, Any]],
        files: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        **kwargs
    ) -> Generator[List[Dict[str, Any]], None, None]:
        """
        运行Agent
        
        Args:
            messages: 消息列表
            files: 文件路径列表
            session_id: 会话ID（可选，不提供时自动生成UUID）
            **kwargs: 其他参数
        
        Yields:
            响应消息列表
        """
        logger.info(f"🚀 TIRAgent.run() 开始执行")
        logger.info(f"   消息数: {len(messages)}, 文件数: {len(files) if files else 0}")
        
        # 生成或复用session_id
        self._current_session_id = session_id or str(uuid.uuid4())
        logger.info(f"   Session ID: {self._current_session_id}")
        
        # 1. 处理文件（复用现有FileProcessor逻辑）
        file_context = ""
        images_to_analyze = []
        
        if files:
            logger.info(f"   处理上传的文件...")
            file_result = self.process_files(files)
            file_context = file_result["context"]
            images_to_analyze = file_result["images"]
            logger.info(f"   文件上下文长度: {len(file_context)}, 图片数: {len(images_to_analyze)}")
        
        # 使用ContextManager截断文件上下文
        if file_context and self.context_manager:
            try:
                file_context = self.context_manager.truncate_text(
                    file_context,
                    self.context_manager.budget["file_context"]
                )
                logger.info(f"   文件上下文已截断至预算范围")
            except Exception as e:
                logger.warning(f"   文件上下文截断失败: {e}")
        
        # 构建增强的消息
        enhanced_messages = messages.copy()
        
        # 如果有文件上下文，添加到第一条用户消息
        if file_context and enhanced_messages:
            first_user_msg = None
            for i, msg in enumerate(enhanced_messages):
                if msg.get("role") == "user":
                    first_user_msg = i
                    break
            
            if first_user_msg is not None:
                original_content = enhanced_messages[first_user_msg].get("content", "")
                enhanced_messages[first_user_msg]["content"] = f"{file_context}\n\n用户问题: {original_content}"
                logger.info(f"   已添加文件上下文到用户消息")
        
        # 如果有图片需要分析，先用VLM预处理
        vlm_context = ""
        if images_to_analyze:
            logger.info(f"   🔍 使用VLM分析 {min(len(images_to_analyze), 5)} 张图片...")
            vlm_results = []
            for i, img_path in enumerate(images_to_analyze[:5]):  # 限制图片数量
                try:
                    logger.info(f"      分析图片 {i+1}/{min(len(images_to_analyze), 5)}: {img_path}")
                    result = self.vlm_tool.analyze_image(img_path)
                    vlm_results.append(f"图片分析结果 ({img_path}):\n{result}")
                    logger.info(f"      ✅ 图片分析完成")
                except Exception as e:
                    logger.error(f"      ❌ 图片分析失败: {e}")
                    vlm_results.append(f"图片分析失败 ({img_path}): {str(e)}")
            
            if vlm_results:
                vlm_context = "\n\n".join(vlm_results)
                # 使用ContextManager截断VLM结果
                if self.context_manager:
                    try:
                        vlm_context = self.context_manager.truncate_text(
                            vlm_context,
                            self.context_manager.budget["vlm_results"]
                        )
                        logger.info(f"   VLM结果已截断至预算范围")
                    except Exception as e:
                        logger.warning(f"   VLM结果截断失败: {e}")
                # 将VLM分析结果添加到用户消息中
                for i in range(len(enhanced_messages) - 1, -1, -1):
                    if enhanced_messages[i].get("role") == "user":
                        enhanced_messages[i]["content"] += f"\n\n---\n**VLM图片分析结果:**\n{vlm_context}"
                        break
                logger.info(f"   ✅ VLM分析完成，已添加到用户消息上下文")
        
        # 获取当前查询文本（用于Memory和Planning）
        query = ""
        for msg in reversed(enhanced_messages):
            if msg.get("role") == "user":
                query = msg.get("content", "")
                break
        
        # 2. 如果memory_enabled，获取记忆上下文
        memory_context = ""
        if self._memory_enabled and self.memory_manager:
            try:
                memory_context = self.memory_manager.get_context_for_agent(
                    self._current_session_id, query
                )
                if memory_context:
                    logger.info(f"   📚 获取到记忆上下文，长度: {len(memory_context)}")
                    # 使用ContextManager截断Memory上下文
                    if self.context_manager:
                        try:
                            memory_context = self.context_manager.truncate_text(
                                memory_context,
                                self.context_manager.budget["memory"]
                            )
                            logger.info(f"   Memory上下文已截断至预算范围")
                        except Exception as e:
                            logger.warning(f"   Memory上下文截断失败: {e}")
            except Exception as e:
                logger.error(f"   ⚠️ 获取记忆上下文失败: {e}")
        
        # 3. 如果planning_enabled且complexity_threshold != "never"，评估复杂度并可能进行规划
        use_planning = False
        complexity = "simple"
        if self._planning_enabled and self.planner and self._complexity_threshold != "never":
            try:
                complexity = self.planner.assess_complexity(query)
                
                # 如果是复杂任务或强制启用规划
                if complexity == "complex" or self._complexity_threshold == "always":
                    use_planning = True
            except Exception as e:
                logger.error(f"   ⚠️ 复杂度评估失败: {e}")
        
        # 记录复杂度判断结果
        logger.info("任务复杂度: %s, 使用%s模式", complexity, "规划" if use_planning else "直接")
        
        # 判断Memory注入方式：Planning路径只传给planner，直接路径注入消息
        inject_memory_to_messages = not use_planning
        
        # 如果不是planning路径，将memory_context注入到消息中
        if memory_context and inject_memory_to_messages and enhanced_messages:
            system_found = False
            for msg in enhanced_messages:
                if msg.get("role") == "system":
                    msg["content"] = f"{msg['content']}\n\n[历史记忆上下文]\n{memory_context}"
                    system_found = True
                    break
            if not system_found:
                # 添加到第一条用户消息
                enhanced_messages[0]["content"] = f"[历史记忆上下文]\n{memory_context}\n\n---\n\n{enhanced_messages[0]['content']}"
            logger.info(f"   已添加记忆上下文到消息")
        
        # 执行规划或直接调用Agent
        if use_planning:
            try:
                # 根据执行模式选择: auto(双层), react(纯ReAct), planning(纯Planning)
                if self._execution_mode == "auto":
                    # ===== 双层模式：Planning + ReAct 执行 =====
                    logger.info("   使用双层模式: Planning + ReAct")
                    
                    # 第一层：生成计划
                    available_skills = self.skill_registry.list_skills()
                    available_skills_text = "\n".join([
                        f"- {s.get('name', 'unknown')}: {s.get('description', '')}"
                        for s in available_skills
                    ])
                    plan = self.planner.plan(query, available_skills_text, memory_context)
                    
                    if plan and plan.steps:
                        # 打印计划日志
                        logger.info("=" * 50)
                        logger.info("[Planning] 生成执行计划，共 %d 个步骤", len(plan.steps))
                        for step in plan.steps:
                            deps = ", ".join(step.depends_on) if step.depends_on else "无"
                            logger.info(f"  [{step.id}] {step.description} (skill: {step.skill_name}, 依赖: {deps})")
                        logger.info("=" * 50)
                        
                        # yield 计划信息供 Streamlit 展示
                        plan_display = self._format_plan_for_display(plan)
                        yield [{"role": "assistant", "content": plan_display, "name": "planning"}]
                        
                        # 第二层：ReAct 执行
                        tools = self._build_tools_dict()
                        react_result = self.plan_executor.execute_planned_react(
                            plan=plan,
                            tools=tools,
                            llm_caller=self._llm_caller,
                            prompt_manager=self.prompt_manager,
                            on_step_start=self._on_step_start,
                            on_step_complete=self._on_step_complete,
                            on_replan=self._on_replan,
                            max_react_per_step=self._max_react_per_step,
                            max_replan_count=3
                        )
                        
                        response_content = react_result
                        yield [{"role": "assistant", "content": response_content}]
                        
                        # 完成日志
                        logger.info("=" * 50)
                        logger.info("✅ 任务执行完成")
                        logger.info(f"   Session ID: {self._current_session_id}")
                        logger.info(f"   执行模式: Planning + ReAct (auto)")
                        logger.info(f"   响应长度: {len(response_content)} 字符")
                        logger.info("=" * 50)
                        
                        # 记录到Memory
                        if self._memory_enabled and self.memory_manager:
                            try:
                                self.memory_manager.remember(
                                    self._current_session_id,
                                    enhanced_messages + [{"role": "assistant", "content": response_content}],
                                    []
                                )
                                self.memory_manager.compress_if_needed(self._current_session_id)
                            except Exception as e:
                                logger.error(f"   ⚠️ 记忆存储失败: {e}")
                        
                        return
                    else:
                        # Planning 失败，回退到纯 ReAct
                        logger.warning("[Planning] 计划生成失败，回退到 ReAct 模式")
                        # 继续执行下面的 ReAct 逻辑
                        
                if self._execution_mode == "react" or (self._execution_mode == "auto" and not (plan and plan.steps)):
                    # ===== 纯 ReAct 模式（调试用或auto回退）=====
                    logger.info("   🔄 使用 ReAct 模式执行任务...")
                    tools = self._build_tools_dict()
                    react_result = self.plan_executor.execute_react(
                        query=query,  # 用户原始问题
                        tools=tools,
                        llm_caller=self._llm_caller,
                        memory_context=memory_context,  # 记忆上下文
                        max_iterations=self._react_max_iterations,
                        prompt_manager=self.prompt_manager
                    )
                    
                    # yield 最终结果
                    yield [{"role": "assistant", "content": react_result}]
                    
                    # 任务完成日志
                    logger.info("=" * 50)
                    logger.info("✅ 任务执行完成")
                    logger.info(f"   Session ID: {self._current_session_id}")
                    logger.info(f"   执行模式: ReAct")
                    logger.info(f"   响应长度: {len(react_result)} 字符")
                    logger.info("=" * 50)
                    
                    # 记录到Memory
                    if self._memory_enabled and self.memory_manager:
                        try:
                            self.memory_manager.remember(
                                self._current_session_id,
                                enhanced_messages + [{"role": "assistant", "content": react_result}],
                                []
                            )
                            self.memory_manager.compress_if_needed(self._current_session_id)
                        except Exception as e:
                            logger.error(f"   ⚠️ 记忆存储失败: {e}")
                    
                    return
                    
                else:
                    # ===== 纯 Planning 模式（调试用）=====
                    logger.info("   📋 使用Planning系统执行任务...")
                    available_skills = self.skill_registry.list_skills()
                    plan = self.planner.plan(query, available_skills, memory_context)
                    logger.info(f"   📋 生成计划，共 {len(plan.steps)} 个步骤")
                    
                    # 执行计划
                    result = self.plan_executor.execute(
                        plan, 
                        skill_executor=self._execute_skill,
                        on_step_complete=lambda step: self._on_step_complete(step, plan),
                        on_replan=lambda plan, failed_step, error: self._on_replan(plan, None, failed_step)
                    )
                    
                    # 将plan执行结果格式化为响应
                    if result.success:
                        response_content = result.final_answer
                    else:
                        # 记录详细失败信息
                        failed_steps_info = []
                        for step in result.plan.steps:
                            if step.status.value == "failed":
                                failed_steps_info.append(f"  - [{step.id}] {step.skill_name}: {step.error or 'unknown'}")
                        if failed_steps_info:
                            logger.warning("Planning执行失败，失败步骤:\n%s", "\n".join(failed_steps_info))
                        
                        response_content = f"任务执行遇到问题。\n\n{result.final_answer}"
                        logger.warning("Planning执行失败，回退到直接Agent调用")
                        use_planning = False
                    
                    if use_planning:
                        yield [{"role": "assistant", "content": response_content}]

                        # 记录任务完成日志
                        logger.info("=" * 50)
                        logger.info("✅ 任务执行完成")
                        logger.info(f"   Session ID: {self._current_session_id}")
                        logger.info(f"   执行模式: Planning")
                        logger.info(f"   响应长度: {len(response_content)} 字符")
                        logger.info("=" * 50)

                        # 记录到Memory
                        if self._memory_enabled and self.memory_manager:
                            try:
                                self.memory_manager.remember(
                                    self._current_session_id,
                                    enhanced_messages + [{"role": "assistant", "content": response_content}],
                                    []
                                )
                                self.memory_manager.compress_if_needed(self._current_session_id)
                            except Exception as e:
                                logger.error(f"   ⚠️ 记忆存储失败: {e}")

                        return
                
            except Exception as e:
                logger.error(f"   ⚠️ Planning/ReAct 异常，回退到直接Agent调用: {e}")
                use_planning = False
        
        # 如果不使用planning（包括回退情况），执行直接调用
        if not use_planning:
            # 如果是从planning回退，需要补充memory注入
            if memory_context and not inject_memory_to_messages and enhanced_messages:
                system_found = False
                for msg in enhanced_messages:
                    if msg.get("role") == "system":
                        msg["content"] = f"{msg['content']}\n\n[历史记忆上下文]\n{memory_context}"
                        system_found = True
                        break
                if not system_found:
                    enhanced_messages[0]["content"] = f"[历史记忆上下文]\n{memory_context}\n\n---\n\n{enhanced_messages[0]['content']}"
                logger.info(f"   回退时补充记忆上下文到消息")
        
        # 直接调用Qwen-Agent（现有流程，但用Skills作为工具）
        tool_calls_recorded = []
        pending_tool_calls = {}  # 暂存等待结果的工具调用 {tool_call_id: {...}}
        
        if self._qwen_agent:
            try:
                for response in self._qwen_agent.run(messages=enhanced_messages, **kwargs):
                    # 记录工具调用（从response中提取）
                    for msg in response:
                        # 1. 捕获工具调用请求（role=assistant + tool_calls）
                        if msg.get("role") == "assistant" and msg.get("tool_calls"):
                            for tc in msg.get("tool_calls", []):
                                tc_id = tc.get("id", "")
                                fn = tc.get("function", {})
                                tool_name = fn.get("name", "unknown")
                                pending_tool_calls[tc_id] = {
                                    "tool_name": tool_name,
                                    "params": fn.get("arguments", {}),
                                    "result": "",
                                    "duration": 0
                                }
                        
                        # 2. 捕获工具执行结果（role=function 或 role=tool）
                        if msg.get("role") in ("function", "tool"):
                            tool_name = msg.get("name", "")
                            tool_result = msg.get("content", "")
                            
                            # 截断过长的结果再存入记录
                            if len(tool_result) > 2000:
                                stored_result = tool_result[:2000] + "... [已截断]"
                            else:
                                stored_result = tool_result
                            
                            # 尝试通过tool_call_id匹配，否则通过name匹配
                            tc_id = msg.get("tool_call_id", "")
                            if tc_id and tc_id in pending_tool_calls:
                                pending_tool_calls[tc_id]["result"] = stored_result
                                tool_calls_recorded.append(pending_tool_calls.pop(tc_id))
                            elif tool_name in pending_tool_calls:
                                pending_tool_calls[tool_name]["result"] = stored_result
                                tool_calls_recorded.append(pending_tool_calls.pop(tool_name))
                            else:
                                # 未找到匹配的调用请求，直接记录结果
                                tool_calls_recorded.append({
                                    "tool_name": tool_name,
                                    "params": {},
                                    "result": stored_result,
                                    "duration": 0
                                })
                    
                    yield response
                
                # 将未匹配结果的工具调用也记录下来
                for tc in pending_tool_calls.values():
                    tool_calls_recorded.append(tc)

                # 记录任务完成日志
                logger.info("=" * 50)
                logger.info("✅ 任务执行完成")
                logger.info(f"   Session ID: {self._current_session_id}")
                logger.info(f"   执行模式: 直接调用")
                logger.info("=" * 50)

                # 4. 如果memory_enabled，记录对话
                if self._memory_enabled and self.memory_manager:
                    try:
                        # 收集完整的对话
                        all_messages = enhanced_messages.copy()
                        # 这里简化处理，实际应该从response中获取assistant回复
                        self.memory_manager.remember(
                            self._current_session_id,
                            all_messages,
                            tool_calls_recorded
                        )
                        self.memory_manager.compress_if_needed(self._current_session_id)
                        logger.info("   💾 对话已记录到Memory")
                    except Exception as e:
                        logger.error(f"   ⚠️ 记忆存储失败: {e}")

                return
            except Exception as e:
                logger.error("=" * 50)
                logger.error("❌ 任务执行失败")
                logger.error(f"   Session ID: {self._current_session_id}")
                logger.error(f"   错误: {str(e)}")
                logger.error("=" * 50)
                logger.error(f"Qwen-Agent运行失败: {e}")

        # 降级模式：使用简化的处理逻辑
        yield from self._fallback_run(enhanced_messages)
    
    def _fallback_run(
        self,
        messages: List[Dict[str, Any]]
    ) -> Generator[List[Dict[str, Any]], None, None]:
        """
        降级模式运行（当Qwen-Agent不可用时）
        """
        # 使用OpenAI兼容API直接调用
        try:
            from openai import OpenAI
            
            client_kwargs = {}
            if self.llm_config.get("model_type") == "qwen_dashscope":
                client_kwargs = {
                    "api_key": self.llm_config.get("api_key") or os.getenv("DASHSCOPE_API_KEY"),
                    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
                }
            else:
                client_kwargs = {
                    "api_key": self.llm_config.get("api_key") or os.getenv("OPENAI_API_KEY"),
                    "base_url": self.llm_config.get("model_server")
                }
            
            client = OpenAI(**client_kwargs)
            
            # 构建消息，确保只有一条system消息在开头
            api_messages = [{"role": "system", "content": self.system_message}]
            
            # 只添加非system消息
            for msg in messages:
                if msg.get("role") != "system":
                    api_messages.append(msg)
            
            logger.info(f"   📤 发送API请求，消息数: {len(api_messages)}")
            
            response = client.chat.completions.create(
                model=self.llm_config.get("model", "qwen-max-latest"),
                messages=api_messages,
                temperature=self.llm_config.get("generate_cfg", {}).get("temperature", 0.7),
            )
            
            assistant_message = response.choices[0].message.content
            logger.info(f"   ✅ API响应成功，长度: {len(assistant_message)}")
            
            yield [{"role": "assistant", "content": assistant_message}]
            
        except Exception as e:
            logger.error(f"   ❌ API调用失败: {e}", exc_info=True)
            yield [{"role": "assistant", "content": f"处理请求时出错: {str(e)}"}]
    
    def chat(
        self,
        user_input: str,
        files: Optional[List[str]] = None,
        history: Optional[List[Dict[str, str]]] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """
        简单的聊天接口
        
        Args:
            user_input: 用户输入
            files: 文件路径列表
            history: 对话历史
            session_id: 会话ID（可选）
        
        Returns:
            str: 助手回复
        """
        messages = history or []
        messages.append({"role": "user", "content": user_input})
        
        final_response = None
        for response in self.run(messages, files, session_id=session_id):
            final_response = response
        
        if final_response:
            return final_response[-1].get("content", "")
        return "抱歉，无法处理您的请求。"
    
    def analyze_image_directly(self, image_path: str, prompt: str = None) -> str:
        """
        直接分析图片（不经过Agent）
        
        Args:
            image_path: 图片路径
            prompt: 分析提示
        
        Returns:
            str: 分析结果
        """
        if prompt:
            return self.vlm_tool.analyze_image(image_path, prompt)
        return self.vlm_tool.describe_image(image_path)
    
    def execute_code(self, code: str, language: str = "python") -> str:
        """
        直接执行代码
        
        Args:
            code: 代码
            language: 语言
        
        Returns:
            str: 执行结果
        """
        result = self.code_executor.execute(code, language)
        return self.code_executor.format_result(result)
    
    def cleanup(self):
        """清理资源"""
        self.file_processor.cleanup()
        if self.memory_manager:
            self.memory_manager.close()
