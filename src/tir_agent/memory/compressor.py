"""
Memory compressor using LLM for summarization.

使用LLM对历史对话和工具调用进行压缩摘要，提取关键信息。
"""

import json
from typing import Optional


class MemoryCompressor:
    """
    LLM-based memory compressor.
    
    使用大语言模型对历史记忆进行压缩，提取关键信息并丢弃冗余细节。
    """
    
    def __init__(self, llm_config: dict):
        """
        Initialize MemoryCompressor.
        
        Args:
            llm_config: LLM configuration dict with keys:
                - model: Model name (e.g., 'qwen-max')
                - model_server: API base URL
                - api_key: API key for authentication
        """
        self.llm_config = llm_config
        self.model = llm_config.get('model', 'qwen-max')
        self.model_server = llm_config.get('model_server', '')
        self.api_key = llm_config.get('api_key', '')
    
    def _call_llm(self, prompt: str) -> str:
        """
        Internal method to call LLM for summary generation.
        
        Args:
            prompt: The prompt to send to LLM
            
        Returns:
            Generated summary text
        """
        try:
            import openai
            
            client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.model_server
            )
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes conversations and extracts key information."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            return response.choices[0].message.content or ""
        except Exception as e:
            print(f"[MemoryCompressor] Error calling LLM: {e}")
            return ""
    
    def compress_conversations(self, conversations: list[dict]) -> dict:
        """
        Compress conversation history into a structured summary.
        
        Args:
            conversations: List of conversation records with keys:
                - role: Message role (user/assistant/system)
                - content: Message content
                - timestamp: Message timestamp
                
        Returns:
            Dict with keys:
                - task_summary: Overall summary of the conversation
                - key_findings: Key information extracted
                - tool_usage_patterns: Patterns in tool usage
                - errors_encountered: Any errors mentioned
        """
        if not conversations:
            return {
                "task_summary": "",
                "key_findings": [],
                "tool_usage_patterns": [],
                "errors_encountered": []
            }
        
        # Format conversations for the prompt
        conversation_text = "\n".join([
            f"[{conv.get('role', 'unknown')}]: {conv.get('content', '')}"
            for conv in conversations
        ])
        
        prompt = f"""Please analyze the following conversation history and provide a structured summary.

Conversation History:
{conversation_text}

Please provide a JSON response with the following structure:
{{
    "task_summary": "A concise summary of what the user was trying to accomplish",
    "key_findings": ["List of key information, decisions, or facts from the conversation"],
    "tool_usage_patterns": ["Patterns in how tools were used or requested"],
    "errors_encountered": ["Any errors, issues, or problems mentioned"]
}}

Focus on extracting actionable information and key context that would be useful for future interactions. Discard redundant or irrelevant details."""

        try:
            response = self._call_llm(prompt)
            
            # Try to parse as JSON
            # Extract JSON from response if wrapped in code blocks
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            
            result = json.loads(json_str)
            
            # Ensure all required keys exist
            return {
                "task_summary": result.get("task_summary", ""),
                "key_findings": result.get("key_findings", []),
                "tool_usage_patterns": result.get("tool_usage_patterns", []),
                "errors_encountered": result.get("errors_encountered", [])
            }
        except json.JSONDecodeError as e:
            print(f"[MemoryCompressor] Error parsing LLM response as JSON: {e}")
            # Return raw text as summary
            return {
                "task_summary": response if 'response' in locals() else "Failed to generate summary",
                "key_findings": [],
                "tool_usage_patterns": [],
                "errors_encountered": []
            }
        except Exception as e:
            print(f"[MemoryCompressor] Error compressing conversations: {e}")
            return {
                "task_summary": "",
                "key_findings": [],
                "tool_usage_patterns": [],
                "errors_encountered": []
            }
    
    def compress_tool_calls(self, tool_calls: list[dict]) -> dict:
        """
        Compress tool call records into a structured summary.
        
        Args:
            tool_calls: List of tool call records with keys:
                - tool_name: Name of the tool
                - params: Tool parameters
                - result: Tool execution result
                - duration: Execution duration
                - timestamp: Call timestamp
                
        Returns:
            Dict with keys:
                - tool_summary: Summary of tool usage
                - frequently_used_tools: List of most used tools
                - common_patterns: Common usage patterns
                - performance_notes: Notes on execution performance
        """
        if not tool_calls:
            return {
                "tool_summary": "",
                "frequently_used_tools": [],
                "common_patterns": [],
                "performance_notes": []
            }
        
        # Format tool calls for the prompt
        tool_calls_text = "\n".join([
            f"Tool: {tc.get('tool_name', 'unknown')}\n"
            f"Params: {json.dumps(tc.get('params', {}), ensure_ascii=False)}\n"
            f"Duration: {tc.get('duration', 0):.2f}s\n"
            f"Result: {str(tc.get('result', ''))[:200]}..."
            for tc in tool_calls
        ])
        
        prompt = f"""Please analyze the following tool call history and provide a structured summary.

Tool Call History:
{tool_calls_text}

Please provide a JSON response with the following structure:
{{
    "tool_summary": "Overall summary of tool usage patterns and purposes",
    "frequently_used_tools": ["List of most frequently used tools"],
    "common_patterns": ["Common patterns in tool usage (e.g., sequential calls, parameter preferences)"],
    "performance_notes": ["Notes on tool execution performance or issues"]
}}

Focus on identifying patterns that would help optimize future tool usage."""

        try:
            response = self._call_llm(prompt)
            
            # Try to parse as JSON
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            
            result = json.loads(json_str)
            
            # Ensure all required keys exist
            return {
                "tool_summary": result.get("tool_summary", ""),
                "frequently_used_tools": result.get("frequently_used_tools", []),
                "common_patterns": result.get("common_patterns", []),
                "performance_notes": result.get("performance_notes", [])
            }
        except json.JSONDecodeError as e:
            print(f"[MemoryCompressor] Error parsing LLM response as JSON: {e}")
            return {
                "tool_summary": response if 'response' in locals() else "Failed to generate summary",
                "frequently_used_tools": [],
                "common_patterns": [],
                "performance_notes": []
            }
        except Exception as e:
            print(f"[MemoryCompressor] Error compressing tool calls: {e}")
            return {
                "tool_summary": "",
                "frequently_used_tools": [],
                "common_patterns": [],
                "performance_notes": []
            }
