"""
Memory Manager - High-level interface for memory operations.

提供记忆管理的高级接口，包括写入、检索、压缩和上下文组装功能。
"""

import json
import logging
from typing import Optional

from .store import MemoryStore
from .compressor import MemoryCompressor

logger = logging.getLogger("tir_agent.memory")


class MemoryManager:
    """
    High-level memory management interface.
    
    协调存储和压缩功能，提供统一的记忆管理接口。
    """
    
    def __init__(
        self,
        db_path: str,
        llm_config: dict,
        compression_threshold: int = 10,
        recall_top_k: int = 5
    ):
        """
        Initialize MemoryManager.
        
        Args:
            db_path: Path to SQLite database
            llm_config: LLM configuration for compression
            compression_threshold: Number of conversation rounds before triggering compression
            recall_top_k: Number of top relevant memories to retrieve
        """
        self.store = MemoryStore(db_path)
        self.compressor = MemoryCompressor(llm_config)
        self.compression_threshold = compression_threshold
        self.recall_top_k = recall_top_k
    
    def remember(
        self,
        session_id: str,
        messages: list[dict],
        tool_calls: Optional[list[dict]] = None
    ) -> None:
        """
        Write memories to storage.
        
        Args:
            session_id: Session identifier
            messages: List of message dicts with 'role' and 'content' keys
            tool_calls: Optional list of tool call dicts with 'tool_name', 'params', 'result', 'duration' keys
        """
        logger.info("存储记忆 [session=%s]: %d 条消息, %d 条工具调用", 
                   session_id, len(messages), len(tool_calls or []))
        try:
            # Store conversations
            for msg in messages:
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                if content:  # Only store non-empty content
                    self.store.add_conversation(session_id, role, content)
            
            # Store tool calls if provided
            if tool_calls:
                for tc in tool_calls:
                    tool_name = tc.get('tool_name', 'unknown')
                    params = tc.get('params', {})
                    result = tc.get('result', '')
                    duration = tc.get('duration', 0.0)
                    self.store.add_tool_call(session_id, tool_name, params, result, duration)
        except Exception as e:
            print(f"[MemoryManager] Error remembering: {e}")
    
    def recall(
        self,
        query: str,
        session_id: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> list[dict]:
        """
        Retrieve relevant memories.
        
        If session_id is provided, prioritize summaries from that session.
        Otherwise, use LLM to score and select most relevant summaries from all sessions.
        
        Args:
            query: Search query for relevance scoring
            session_id: Optional session ID to prioritize
            top_k: Number of results to return (defaults to self.recall_top_k)
            
        Returns:
            List of relevant memory summaries
        """
        logger.info("检索记忆: query=%s, session=%s", query[:80], session_id)
        
        if top_k is None:
            top_k = self.recall_top_k
        
        try:
            if session_id:
                # Get summaries from specific session
                summaries = self.store.get_summaries(session_id=session_id)
                return summaries[:top_k]
            else:
                # Get all summaries and use LLM to rank relevance
                all_summaries = self.store.get_summaries()
                if not all_summaries:
                    return []
                
                # Format summaries for LLM scoring
                summaries_text = "\n\n".join([
                    f"[{i}] Session: {s.get('session_id', 'unknown')}\n"
                    f"Type: {s.get('summary_type', 'unknown')}\n"
                    f"Content: {s.get('content', '')[:500]}"
                    for i, s in enumerate(all_summaries)
                ])
                
                prompt = f"""Given the following query and available memory summaries, identify the {top_k} most relevant summaries.

Query: {query}

Available Summaries:
{summaries_text}

Please respond with a JSON array of indices (0-based) of the most relevant summaries, ordered by relevance:
[0, 3, 1]  // example format

Only return the JSON array, nothing else."""

                try:
                    import openai
                    
                    client = openai.OpenAI(
                        api_key=self.compressor.api_key,
                        base_url=self.compressor.model_server
                    )
                    
                    response = client.chat.completions.create(
                        model=self.compressor.model,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that selects relevant information."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.3,
                        max_tokens=100
                    )
                    
                    response_text = response.choices[0].message.content or "[]"
                    
                    # Extract JSON array from response
                    json_str = response_text
                    if "```json" in response_text:
                        json_str = response_text.split("```json")[1].split("```")[0].strip()
                    elif "```" in response_text:
                        json_str = response_text.split("```")[1].split("```")[0].strip()
                    
                    indices = json.loads(json_str)
                    
                    # Return selected summaries
                    result = []
                    for idx in indices[:top_k]:
                        if 0 <= idx < len(all_summaries):
                            result.append(all_summaries[idx])
                    logger.info("检索到 %d 条相关记忆", len(result))
                    return result
                    
                except Exception as e:
                    logger.error("LLM 排序记忆失败: %s", e)
                    # Fallback: return most recent summaries
                    return sorted(
                        all_summaries,
                        key=lambda x: x.get('created_at', 0),
                        reverse=True
                    )[:top_k]
        except Exception as e:
            logger.error("检索记忆失败: %s", e)
            return []
    
    def compress_if_needed(self, session_id: str) -> bool:
        """
        Check if compression is needed and perform it if so.
        
        Args:
            session_id: Session identifier to check
            
        Returns:
            True if compression was performed, False otherwise
        """
        try:
            conversation_count = self.store.get_conversation_count(session_id)
            
            if conversation_count >= self.compression_threshold:
                logger.info("触发记忆压缩 [session=%s]: 对话轮次 %d 超过阈值 %d", 
                           session_id, conversation_count, self.compression_threshold)
                # Get conversations and tool calls for compression
                conversations = self.store.get_conversations(session_id, limit=100)
                tool_calls = self.store.get_tool_calls(session_id, limit=50)
                
                # Compress conversations
                if conversations:
                    conv_summary = self.compressor.compress_conversations(conversations)
                    conv_content = json.dumps(conv_summary, ensure_ascii=False)
                    # Estimate token count (rough approximation: 1 token ~ 4 chars)
                    token_count = len(conv_content) // 4
                    self.store.add_summary(
                        session_id,
                        "conversation",
                        conv_content,
                        token_count
                    )
                
                # Compress tool calls
                if tool_calls:
                    tool_summary = self.compressor.compress_tool_calls(tool_calls)
                    tool_content = json.dumps(tool_summary, ensure_ascii=False)
                    token_count = len(tool_content) // 4
                    self.store.add_summary(
                        session_id,
                        "tool_calls",
                        tool_content,
                        token_count
                    )
                
                return True
            
            logger.debug("记忆无需压缩 [session=%s]: 对话轮次 %d", session_id, conversation_count)
            return False
        except Exception as e:
            logger.error("记忆压缩失败: %s", e)
            return False
    
    def get_context_for_agent(self, session_id: str, current_query: str) -> str:
        """
        Assemble augmented context string for agent.
        
        Includes:
        1. Current session summaries
        2. Relevant historical memories
        3. Formatted as readable text
        
        Args:
            session_id: Current session identifier
            current_query: Current user query for relevance matching
            
        Returns:
            Formatted context string
        """
        logger.info("组装增强上下文 [session=%s], 查询: %s", session_id, current_query[:50])
        try:
            context_parts = []
            
            # 1. Get current session summaries
            session_summaries = self.store.get_summaries(session_id=session_id)
            if session_summaries:
                context_parts.append("=== Previous Session Summary ===")
                for summary in session_summaries:
                    summary_type = summary.get('summary_type', 'unknown')
                    content = summary.get('content', '')
                    try:
                        # Try to parse JSON content for better formatting
                        parsed = json.loads(content)
                        if isinstance(parsed, dict):
                            for key, value in parsed.items():
                                if value:
                                    context_parts.append(f"{key}: {value}")
                        else:
                            context_parts.append(content)
                    except json.JSONDecodeError:
                        context_parts.append(content)
                context_parts.append("")
            
            # 2. Get relevant memories from other sessions
            relevant_memories = self.recall(current_query, top_k=self.recall_top_k)
            # Filter out current session summaries
            other_summaries = [
                m for m in relevant_memories
                if m.get('session_id') != session_id
            ]
            
            if other_summaries:
                context_parts.append("=== Relevant Historical Context ===")
                for memory in other_summaries:
                    session = memory.get('session_id', 'unknown')
                    summary_type = memory.get('summary_type', 'unknown')
                    content = memory.get('content', '')
                    context_parts.append(f"[From session {session} - {summary_type}]")
                    try:
                        parsed = json.loads(content)
                        if isinstance(parsed, dict):
                            for key, value in parsed.items():
                                if value:
                                    context_parts.append(f"  {key}: {value}")
                        else:
                            context_parts.append(f"  {content}")
                    except json.JSONDecodeError:
                        context_parts.append(f"  {content}")
                context_parts.append("")
            
            # 3. Get recent conversation history from current session
            recent_conversations = self.store.get_conversations(session_id, limit=10)
            if recent_conversations:
                context_parts.append("=== Recent Conversation ===")
                for conv in recent_conversations:
                    role = conv.get('role', 'unknown')
                    content = conv.get('content', '')
                    context_parts.append(f"{role}: {content}")
                context_parts.append("")
            
            context = "\n".join(context_parts)
            logger.info("组装增强上下文 [session=%s], 上下文长度: %d", session_id, len(context))
            return context
        except Exception as e:
            logger.error("组装上下文失败: %s", e)
            return ""
    
    def close(self) -> None:
        """Close storage connection."""
        try:
            self.store.close()
            logger.info("MemoryManager 已关闭")
        except Exception as e:
            logger.error("关闭 MemoryManager 失败: %s", e)
