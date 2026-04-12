"""
代码执行器Skill实现
内联了CodeExecutor的核心逻辑
"""
import os
import sys
import json
import subprocess
import tempfile
import traceback
import io
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Dict, List, Optional

from tir_agent.skills.base import BaseSkill, SkillMetadata


class CodeExecutorSkill(BaseSkill):
    """代码执行器Skill - 直接内联代码执行逻辑"""

    SUPPORTED_LANGUAGES = ["python", "bash", "shell"]

    def __init__(self, metadata: SkillMetadata, config: dict = None):
        super().__init__(metadata, config)
        # 从配置获取超时时间，默认60秒
        self.timeout = config.get("timeout", 60) if config else 60
        self.temp_dir = config.get("temp_dir") if config else tempfile.gettempdir()
        self.allowed_modules = config.get("allowed_modules") if config else None
        self._execution_globals = {}

    def execute(
        self,
        code: str,
        language: str = "python",
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        执行代码 - 内联自原CodeExecutor

        Args:
            code: 要执行的代码
            language: 编程语言
            context: 执行上下文（变量环境）

        Returns:
            dict: 执行结果，包含stdout、stderr、result、success等字段
        """
        language = language.lower()

        if language not in self.SUPPORTED_LANGUAGES:
            return {
                "success": False,
                "error": f"不支持的语言: {language}",
                "supported_languages": self.SUPPORTED_LANGUAGES,
            }

        if language == "python":
            return self._execute_python(code, context)
        elif language in ["bash", "shell"]:
            return self._execute_bash(code)

    def _execute_python(
        self,
        code: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """执行Python代码"""
        # 准备执行环境
        globals_env = {
            "__builtins__": __builtins__,
            "__name__": "__main__",
        }

        # 添加常用模块
        common_imports = {
            "math": __import__("math"),
            "json": __import__("json"),
            "re": __import__("re"),
            "datetime": __import__("datetime"),
            "collections": __import__("collections"),
            "itertools": __import__("itertools"),
            "functools": __import__("functools"),
            "os": __import__("os"),
            "sys": __import__("sys"),
        }

        # 尝试导入可选模块
        optional_modules = ["numpy", "pandas", "matplotlib", "PIL"]
        for mod_name in optional_modules:
            try:
                common_imports[mod_name] = __import__(mod_name)
            except ImportError:
                pass

        globals_env.update(common_imports)

        # 添加上下文变量
        if context:
            globals_env.update(context)

        # 捕获输出
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        result = {
            "success": True,
            "stdout": "",
            "stderr": "",
            "result": None,
            "variables": {},
        }

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # 执行代码
                exec(code, globals_env)

                # 如果代码最后一个表达式有值，尝试获取
                lines = code.strip().split("\n")
                if lines:
                    last_line = lines[-1].strip()
                    # 如果最后一行是表达式（不是赋值或语句）
                    if last_line and not any(
                        last_line.startswith(kw)
                        for kw in ["import", "from", "def", "class", "if", "for", "while", "with", "try", "return", "print"]
                    ) and "=" not in last_line:
                        try:
                            result["result"] = eval(last_line, globals_env)
                        except:
                            pass

            result["stdout"] = stdout_capture.getvalue()
            result["stderr"] = stderr_capture.getvalue()

            # 收集新创建的变量（排除内置和模块）
            exclude_vars = set(dir(__builtins__)) | set(common_imports.keys()) | {"__builtins__", "__name__"}
            for key, value in globals_env.items():
                if not key.startswith("_") and key not in exclude_vars:
                    try:
                        # 只保存可序列化的变量
                        json.dumps({key: str(value)})
                        result["variables"][key] = str(value)[:1000]  # 限制长度
                    except:
                        pass

        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
            result["traceback"] = traceback.format_exc()
            result["stdout"] = stdout_capture.getvalue()
            result["stderr"] = stderr_capture.getvalue()

        return result

    def _execute_bash(self, code: str) -> Dict[str, Any]:
        """执行Bash命令"""
        result = {
            "success": True,
            "stdout": "",
            "stderr": "",
            "return_code": 0,
        }

        try:
            process = subprocess.run(
                code,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self.temp_dir,
            )

            result["stdout"] = process.stdout
            result["stderr"] = process.stderr
            result["return_code"] = process.returncode
            result["success"] = process.returncode == 0

        except subprocess.TimeoutExpired:
            result["success"] = False
            result["error"] = f"命令执行超时（{self.timeout}秒）"
        except Exception as e:
            result["success"] = False
            result["error"] = str(e)

        return result

    def format_result(self, result: Dict[str, Any]) -> str:
        """格式化执行结果为可读字符串 - 内联自原CodeExecutor"""
        parts = []

        if result.get("success"):
            parts.append("✅ 执行成功")
        else:
            parts.append("❌ 执行失败")
            if result.get("error"):
                parts.append(f"错误: {result['error']}")
            if result.get("traceback"):
                parts.append(f"```\n{result['traceback']}\n```")

        if result.get("stdout"):
            parts.append(f"**输出:**\n```\n{result['stdout']}\n```")

        if result.get("stderr"):
            parts.append(f"**错误输出:**\n```\n{result['stderr']}\n```")

        if result.get("result") is not None:
            parts.append(f"**返回值:** {result['result']}")

        if result.get("variables"):
            vars_str = "\n".join(f"- {k}: {v}" for k, v in result["variables"].items())
            parts.append(f"**新变量:**\n{vars_str}")

        return "\n\n".join(parts)

    def execute_skill(self, **params) -> str:
        """
        Skill接口方法 - 执行代码

        Args:
            code: 要执行的代码
            language: 编程语言，默认python

        Returns:
            str: 执行结果字符串
        """
        try:
            # 验证参数
            validated_params = self.validate_params(params)
            code = validated_params.get("code", "")
            language = validated_params.get("language", "python")

            # 执行代码
            result = self.execute(code, language)

            # 格式化结果
            return self.format_result(result)

        except ValueError as e:
            return f"参数错误: {str(e)}"
        except Exception as e:
            return f"代码执行失败: {str(e)}"
