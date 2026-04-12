import logging
import json
from typing import Optional, Any

logger = logging.getLogger("tir_agent.context")

# 导入PromptManager，用于类型提示
try:
    from .prompt_manager import PromptManager
except ImportError:
    PromptManager = None


class ContextManager:
    """统一管理Agent上下文的token预算和截断"""
    
    # 默认token预算分配
    DEFAULT_BUDGET = {
        "system": 1000,        # 系统提示 - 固定
        "memory": 3000,        # Memory上下文 - 可压缩
        "file_context": 8000,  # 文件内容 - 可摘要
        "vlm_results": 3000,   # VLM结果 - 可截断
        "user_message": 5000,  # 用户消息 - 保留
        "reserve": 10000,      # 预留给LLM推理+工具输出
    }
    
    def __init__(self, max_tokens: int = 30000, budget: dict = None, prompt_manager: Optional[Any] = None):
        self.max_tokens = max_tokens
        self.budget = budget or self.DEFAULT_BUDGET.copy()
        
        # 初始化PromptManager（如果未提供则创建默认实例）
        if prompt_manager is None and PromptManager is not None:
            from .config import settings
            prompt_manager = PromptManager(
                prompts_dir=settings.prompts_dir,
                version=settings.prompt_version
            )
        self.prompt_manager = prompt_manager
        
        logger.info("ContextManager初始化，max_tokens=%d", max_tokens)
    
    def estimate_tokens(self, text: str) -> int:
        """估算文本的token数量
        
        使用简单启发式：
        - 中文字符：约1.5字/token
        - 英文/数字：约4字符/token
        - 混合文本：取加权平均
        """
        if not text:
            return 0
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other_chars = len(text) - chinese_chars
        tokens = int(chinese_chars / 1.5 + other_chars / 4)
        return max(tokens, 1)
    
    def truncate_text(self, text: str, max_tokens: int, preserve_ends: bool = True) -> str:
        """智能截断文本
        
        Args:
            text: 输入文本
            max_tokens: 最大token数
            preserve_ends: 是否保留首尾（True=保留首部+尾部，False=只保留首部）
        
        Returns:
            截断后的文本
        """
        if not text:
            return text
        
        current_tokens = self.estimate_tokens(text)
        if current_tokens <= max_tokens:
            return text
        
        # 估算每个token对应的字符数
        chars_per_token = len(text) / current_tokens if current_tokens > 0 else 4
        max_chars = int(max_tokens * chars_per_token)
        
        if preserve_ends:
            # 保留首60%和尾30%，中间省略
            head_chars = int(max_chars * 0.6)
            tail_chars = int(max_chars * 0.3)
            truncated = (
                text[:head_chars] 
                + f"\n\n... [内容已截断: 原文约{current_tokens}tokens，已压缩至{max_tokens}tokens] ...\n\n"
                + text[-tail_chars:]
            )
        else:
            truncated = text[:max_chars] + f"\n\n... [已截断: 原文约{current_tokens}tokens]"
        
        logger.info("文本截断: %d tokens -> %d tokens", current_tokens, max_tokens)
        return truncated
    
    def summarize_if_needed(self, text: str, max_tokens: int, llm_config: dict = None) -> str:
        """超过预算时尝试LLM摘要，失败则降级为截断
        
        Args:
            text: 输入文本
            max_tokens: 最大token数
            llm_config: LLM配置（包含model, model_server, api_key）。为None则直接截断
        
        Returns:
            摘要或截断后的文本
        """
        if not text:
            return text
        
        current_tokens = self.estimate_tokens(text)
        if current_tokens <= max_tokens:
            return text
        
        # 尝试LLM摘要
        if llm_config:
            try:
                summary = self._llm_summarize(text, max_tokens, llm_config)
                if summary and self.estimate_tokens(summary) <= max_tokens * 1.2:
                    logger.info("LLM摘要成功: %d tokens -> %d tokens", current_tokens, self.estimate_tokens(summary))
                    return summary
            except Exception as e:
                logger.warning("LLM摘要失败，降级为截断: %s", str(e))
        
        # 降级为截断
        return self.truncate_text(text, max_tokens, preserve_ends=True)
    
    def _llm_summarize(self, text: str, max_tokens: int, llm_config: dict) -> str:
        """调用LLM生成摘要"""
        from openai import OpenAI
        
        # 如果文本太长，先截断到一个合理长度再摘要
        input_limit = max_tokens * 5  # 输入最多是输出的5倍
        if self.estimate_tokens(text) > input_limit:
            text = self.truncate_text(text, input_limit, preserve_ends=True)
        
        model_server = llm_config.get("model_server", "")
        api_key = llm_config.get("api_key", "none")
        model = llm_config.get("model", "")
        
        # 构建base_url
        if "dashscope" in model_server:
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        elif model_server:
            base_url = model_server
        else:
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        
        client = OpenAI(api_key=api_key, base_url=base_url)
        
        # 使用PromptManager获取摘要prompt
        if self.prompt_manager:
            system_content = self.prompt_manager.get("summarization")
        else:
            system_content = "你是一个文本摘要专家。请提取以下内容的关键信息，生成简洁的摘要。保留重要的数据、结论和关键细节。"
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"请将以下内容压缩为不超过{max_tokens}个token的摘要：\n\n{text}"}
            ],
            max_tokens=int(max_tokens * 1.2),
            temperature=0.1
        )
        
        return response.choices[0].message.content
    
    def truncate_tool_output(self, output: str, max_chars: int = 4000) -> str:
        """截断工具输出
        
        Args:
            output: 工具输出文本
            max_chars: 最大字符数
        
        Returns:
            截断后的输出
        """
        if not output or len(output) <= max_chars:
            return output
        
        original_len = len(output)
        # 保留首部大部分 + 尾部少量
        head = int(max_chars * 0.8)
        tail = int(max_chars * 0.15)
        truncated = (
            output[:head]
            + f"\n\n... [工具输出已截断: 原文{original_len}字符，保留{max_chars}字符] ...\n\n"
            + output[-tail:]
        )
        logger.info("工具输出截断: %d chars -> %d chars", original_len, max_chars)
        return truncated
    
    def build_enhanced_messages(
        self,
        messages: list,
        file_context: str = None,
        vlm_context: str = None,
        memory_context: str = None,
        system_message: str = None
    ) -> list:
        """按token预算组装增强消息
        
        按优先级截断各层上下文：
        1. system_message (固定，不截断)
        2. user_message (高优先级，尽量保留)
        3. file_context (中优先级，可摘要)
        4. memory_context (中优先级，可截断)
        5. vlm_context (低优先级，可截断)
        
        Args:
            messages: 原始消息列表 [{"role": "user"/"assistant", "content": "..."}]
            file_context: 文件处理结果文本
            vlm_context: VLM分析结果文本
            memory_context: Memory检索结果文本
            system_message: 系统提示词
        
        Returns:
            增强后的消息列表
        """
        enhanced = []
        used_tokens = 0
        
        # 1. System message
        if system_message:
            sys_tokens = self.estimate_tokens(system_message)
            if sys_tokens > self.budget["system"]:
                system_message = self.truncate_text(system_message, self.budget["system"])
            enhanced.append({"role": "system", "content": system_message})
            used_tokens += self.estimate_tokens(system_message)
        
        # 2. 截断各上下文层
        if file_context:
            file_context = self.truncate_text(file_context, self.budget["file_context"])
            logger.debug("文件上下文: %d tokens", self.estimate_tokens(file_context))
        
        if memory_context:
            memory_context = self.truncate_text(memory_context, self.budget["memory"])
            logger.debug("Memory上下文: %d tokens", self.estimate_tokens(memory_context))
        
        if vlm_context:
            vlm_context = self.truncate_text(vlm_context, self.budget["vlm_results"])
            logger.debug("VLM上下文: %d tokens", self.estimate_tokens(vlm_context))
        
        # 3. 组装消息（将上下文注入到第一条用户消息前）
        context_parts = []
        if memory_context:
            context_parts.append(f"[历史记忆上下文]\n{memory_context}")
        if file_context:
            context_parts.append(f"[文件内容]\n{file_context}")
        if vlm_context:
            context_parts.append(f"[图片分析结果]\n{vlm_context}")
        
        context_prefix = "\n\n---\n\n".join(context_parts) if context_parts else ""
        
        # 4. 处理消息列表
        first_user_found = False
        for msg in messages:
            new_msg = dict(msg)
            if msg.get("role") == "user" and not first_user_found:
                first_user_found = True
                if context_prefix:
                    new_msg["content"] = f"{context_prefix}\n\n---\n\n用户问题: {msg['content']}"
                # 检查用户消息是否超长
                user_tokens = self.estimate_tokens(new_msg["content"])
                if user_tokens > self.budget["user_message"] + sum(
                    self.budget[k] for k in ["file_context", "memory", "vlm_results"]
                ):
                    new_msg["content"] = self.truncate_text(
                        new_msg["content"],
                        self.budget["user_message"] + sum(
                            self.budget[k] for k in ["file_context", "memory", "vlm_results"]
                        )
                    )
            enhanced.append(new_msg)
        
        total_tokens = sum(self.estimate_tokens(m.get("content", "")) for m in enhanced)
        logger.info("消息组装完成，总计约 %d tokens (预算: %d)", total_tokens, self.max_tokens)
        
        return enhanced
