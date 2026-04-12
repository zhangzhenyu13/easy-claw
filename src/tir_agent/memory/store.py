"""
SQLite storage engine for memory system.

提供基于SQLite的记忆存储功能，包括对话记录、工具调用记录和摘要存储。
"""

import sqlite3
import json
import time
import os
import logging
from typing import Optional

logger = logging.getLogger("tir_agent.memory")


class MemoryStore:
    """
    SQLite-based memory storage.
    
    管理三张核心表：
    - conversations: 会话记录
    - tool_calls: 工具调用记录
    - summaries: 压缩摘要
    """
    
    def __init__(self, db_path: str):
        """
        Initialize MemoryStore with database path.
        
        Args:
            db_path: SQLite database file path. Parent directory will be created if not exists.
        """
        # Ensure parent directory exists
        parent_dir = os.path.dirname(db_path)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
        
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
    
    def _create_tables(self) -> None:
        """Create required tables if not exist."""
        try:
            cursor = self.conn.cursor()
            
            # Conversations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp REAL NOT NULL
                )
            """)
            
            # Create index on session_id for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversations_session 
                ON conversations(session_id)
            """)
            
            # Tool calls table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tool_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    params TEXT NOT NULL,
                    result TEXT,
                    duration REAL,
                    timestamp REAL NOT NULL
                )
            """)
            
            # Create index on session_id for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_tool_calls_session 
                ON tool_calls(session_id)
            """)
            
            # Summaries table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    summary_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    token_count INTEGER,
                    created_at REAL NOT NULL
                )
            """)
            
            # Create indexes on session_id and summary_type
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_summaries_session 
                ON summaries(session_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_summaries_type 
                ON summaries(summary_type)
            """)
            
            self.conn.commit()
        except sqlite3.Error as e:
            # Log error but don't crash
            print(f"[MemoryStore] Error creating tables: {e}")
    
    def add_conversation(self, session_id: str, role: str, content: str) -> None:
        """
        Insert a conversation record.
        
        Args:
            session_id: Session identifier
            role: Message role (e.g., 'user', 'assistant', 'system')
            content: Message content
        """
        logger.debug("写入对话记录 [session=%s, role=%s]", session_id, role)
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT INTO conversations (session_id, role, content, timestamp)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, role, content, time.time())
            )
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"[MemoryStore] Error adding conversation: {e}")
    
    def add_tool_call(
        self,
        session_id: str,
        tool_name: str,
        params: dict,
        result: str,
        duration: float
    ) -> None:
        """
        Insert a tool call record.
        
        Args:
            session_id: Session identifier
            tool_name: Name of the tool called
            params: Tool parameters as dict (will be JSON serialized)
            result: Tool execution result
            duration: Execution duration in seconds
        """
        logger.debug("写入工具调用记录 [session=%s, tool=%s]", session_id, tool_name)
        try:
            cursor = self.conn.cursor()
            
            # 确保参数类型安全
            if isinstance(params, dict):
                params_json = json.dumps(params, ensure_ascii=False)
            elif isinstance(params, str):
                params_json = params
            else:
                params_json = json.dumps(params, ensure_ascii=False) if params else "{}"
            
            if isinstance(result, dict):
                result = json.dumps(result, ensure_ascii=False)
            elif result is None:
                result = ""
            else:
                result = str(result)
            
            duration = float(duration) if duration is not None else 0.0
            
            cursor.execute(
                """
                INSERT INTO tool_calls (session_id, tool_name, params, result, duration, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, tool_name, params_json, result, duration, time.time())
            )
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"[MemoryStore] Error adding tool call: {e}")
    
    def add_summary(
        self,
        session_id: str,
        summary_type: str,
        content: str,
        token_count: int
    ) -> None:
        """
        Insert a summary record.
        
        Args:
            session_id: Session identifier
            summary_type: Type of summary (e.g., 'conversation', 'tool_calls')
            content: Summary content
            token_count: Estimated token count of the summary
        """
        logger.debug("写入摘要 [session=%s, type=%s]", session_id, summary_type)
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT INTO summaries (session_id, summary_type, content, token_count, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, summary_type, content, token_count, time.time())
            )
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"[MemoryStore] Error adding summary: {e}")
    
    def get_conversations(self, session_id: str, limit: int = 50) -> list[dict]:
        """
        Get conversation records for a session.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of records to return
            
        Returns:
            List of conversation records as dicts
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                SELECT id, session_id, role, content, timestamp
                FROM conversations
                WHERE session_id = ?
                ORDER BY timestamp ASC
                LIMIT ?
                """,
                (session_id, limit)
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error as e:
            print(f"[MemoryStore] Error getting conversations: {e}")
            return []
    
    def get_tool_calls(self, session_id: str, limit: int = 20) -> list[dict]:
        """
        Get tool call records for a session.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of records to return
            
        Returns:
            List of tool call records as dicts
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                SELECT id, session_id, tool_name, params, result, duration, timestamp
                FROM tool_calls
                WHERE session_id = ?
                ORDER BY timestamp ASC
                LIMIT ?
                """,
                (session_id, limit)
            )
            rows = cursor.fetchall()
            result = []
            for row in rows:
                row_dict = dict(row)
                # Deserialize params JSON
                try:
                    row_dict['params'] = json.loads(row_dict['params'])
                except json.JSONDecodeError:
                    row_dict['params'] = {}
                result.append(row_dict)
            return result
        except sqlite3.Error as e:
            print(f"[MemoryStore] Error getting tool calls: {e}")
            return []
    
    def get_summaries(
        self,
        session_id: Optional[str] = None,
        summary_type: Optional[str] = None
    ) -> list[dict]:
        """
        Get summary records.
        
        Args:
            session_id: Filter by session ID (optional)
            summary_type: Filter by summary type (optional)
            
        Returns:
            List of summary records as dicts
        """
        try:
            cursor = self.conn.cursor()
            query = """
                SELECT id, session_id, summary_type, content, token_count, created_at
                FROM summaries
                WHERE 1=1
            """
            params = []
            
            if session_id:
                query += " AND session_id = ?"
                params.append(session_id)
            
            if summary_type:
                query += " AND summary_type = ?"
                params.append(summary_type)
            
            query += " ORDER BY created_at DESC"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error as e:
            print(f"[MemoryStore] Error getting summaries: {e}")
            return []
    
    def get_recent_sessions(self, limit: int = 10) -> list[str]:
        """
        Get list of recent session IDs.
        
        Args:
            limit: Maximum number of sessions to return
            
        Returns:
            List of session IDs ordered by most recent activity
        """
        try:
            cursor = self.conn.cursor()
            # Get distinct session_ids ordered by latest conversation timestamp
            cursor.execute(
                """
                SELECT session_id, MAX(timestamp) as last_activity
                FROM conversations
                GROUP BY session_id
                ORDER BY last_activity DESC
                LIMIT ?
                """,
                (limit,)
            )
            rows = cursor.fetchall()
            return [row['session_id'] for row in rows]
        except sqlite3.Error as e:
            print(f"[MemoryStore] Error getting recent sessions: {e}")
            return []
    
    def get_conversation_count(self, session_id: str) -> int:
        """
        Get the number of conversation rounds for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Number of conversation records
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                SELECT COUNT(*) as count
                FROM conversations
                WHERE session_id = ?
                """,
                (session_id,)
            )
            row = cursor.fetchone()
            return row['count'] if row else 0
        except sqlite3.Error as e:
            print(f"[MemoryStore] Error getting conversation count: {e}")
            return 0
    
    def close(self) -> None:
        """Close the database connection."""
        try:
            if self.conn:
                self.conn.close()
        except sqlite3.Error as e:
            print(f"[MemoryStore] Error closing connection: {e}")
