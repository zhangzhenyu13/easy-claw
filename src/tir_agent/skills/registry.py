"""
Skill注册表

管理所有已加载的Skill，提供注册、查询和转换为qwen-agent工具的功能。
"""
import logging
from typing import Optional

from .base import BaseSkill
from .loader import SkillLoader

logger = logging.getLogger("tir_agent.skills")

# 延迟导入qwen-agent（避免循环导入）
_qwen_base_tool = None
_qwen_register_tool = None


def _get_qwen_tools():
    """延迟获取qwen-agent工具基类"""
    global _qwen_base_tool, _qwen_register_tool
    if _qwen_base_tool is None:
        try:
            from qwen_agent.tools.base import BaseTool, register_tool
            _qwen_base_tool = BaseTool
            _qwen_register_tool = register_tool
        except ImportError:
            logger.warning("qwen_agent未安装，无法转换为qwen-agent工具")
            return None, None
    return _qwen_base_tool, _qwen_register_tool


class SkillToolWrapper:
    """
    将BaseSkill包装为qwen-agent的BaseTool格式

    这个类会被动态创建子类，设置正确的name/description/parameters类属性
    """
    _skill_instance: BaseSkill = None

    def call(self, params: str, **kwargs) -> str:
        """
        调用Skill执行

        Args:
            params: JSON格式的参数字符串或字典

        Returns:
            str: 执行结果
        """
        import json

        try:
            # 解析参数
            if isinstance(params, str):
                param_dict = json.loads(params)
            else:
                param_dict = params

            # 验证参数
            validated = self._skill_instance.validate_params(param_dict)

            # 执行Skill
            result = self._skill_instance.execute(**validated)
            return result

        except json.JSONDecodeError as e:
            logger.error(f"参数JSON解析失败: {e}")
            return json.dumps({"error": f"参数解析失败: {str(e)}"}, ensure_ascii=False)
        except ValueError as e:
            logger.error(f"参数验证失败: {e}")
            return json.dumps({"error": str(e)}, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Skill执行失败: {e}", exc_info=True)
            return json.dumps({"error": f"执行失败: {str(e)}"}, ensure_ascii=False)


class SkillRegistry:
    """Skill注册表，管理所有已加载的Skill"""

    def __init__(self):
        self._skills: dict[str, BaseSkill] = {}
        self._loader = SkillLoader()

    def discover(self, skill_dirs: list[str]) -> int:
        """
        扫描多个目录，注册所有发现的Skills

        Args:
            skill_dirs: Skill目录列表

        Returns:
            成功加载的Skill数量
        """
        logger.info("扫描 Skill 目录: %s", skill_dirs)
        
        total_loaded = 0
        for skill_dir in skill_dirs:
            try:
                skills = self._loader.discover_skills(skill_dir)
                for skill in skills:
                    self.register(skill)
                    total_loaded += 1
            except Exception as e:
                logger.error(f"扫描目录失败 {skill_dir}: {e}", exc_info=True)
                # 单个目录失败不影响其他目录
                continue
        
        names = [name for name in self._skills.keys()]
        logger.info("发现并注册 %d 个 Skills: %s", total_loaded, names)
        return total_loaded

    def register(self, skill: BaseSkill) -> None:
        """
        注册单个Skill

        Args:
            skill: Skill实例
        """
        name = skill.name
        if name in self._skills:
            logger.warning(f"Skill '{name}' 已存在，将被覆盖")

        self._skills[name] = skill
        logger.info("注册 Skill: %s (v%s)", skill.name, skill.metadata.version)

    def unregister(self, name: str) -> bool:
        """
        注销Skill

        Args:
            name: Skill名称

        Returns:
            是否成功注销
        """
        if name in self._skills:
            del self._skills[name]
            logger.info("注销 Skill: %s", name)
            return True
        logger.warning(f"尝试注销不存在的Skill: {name}")
        return False

    def get(self, name: str) -> Optional[BaseSkill]:
        """
        按名称获取Skill

        Args:
            name: Skill名称

        Returns:
            Skill实例，不存在返回None
        """
        return self._skills.get(name)

    def list_skills(self) -> list[dict]:
        """
        列出所有Skills的基本信息

        Returns:
            包含名称、描述、标签等信息的字典列表
        """
        return [
            {
                "name": skill.name,
                "display_name": skill.metadata.display_name,
                "description": skill.description,
                "version": skill.metadata.version,
                "tags": skill.metadata.tags,
                "enabled": skill.enabled
            }
            for skill in self._skills.values()
        ]

    def get_enabled_skills(self) -> list[BaseSkill]:
        """
        获取所有enabled的Skills

        Returns:
            启用的Skill列表
        """
        return [skill for skill in self._skills.values() if skill.enabled]

    def _convert_param_type(self, yaml_type: str) -> str:
        """
        将YAML参数类型转换为JSON Schema类型

        Args:
            yaml_type: YAML中定义的类型

        Returns:
            JSON Schema类型
        """
        type_mapping = {
            'string': 'string',
            'str': 'string',
            'integer': 'integer',
            'int': 'integer',
            'number': 'number',
            'float': 'number',
            'boolean': 'boolean',
            'bool': 'boolean',
            'array': 'array',
            'list': 'array',
            'object': 'object',
            'dict': 'object',
        }
        return type_mapping.get(yaml_type.lower(), 'string')

    def _build_json_schema(self, skill: BaseSkill) -> dict:
        """
        从SkillMetadata构建JSON Schema

        Args:
            skill: Skill实例

        Returns:
            JSON Schema字典
        """
        properties = {}
        required = []

        for param in skill.metadata.parameters:
            name = param.get('name')
            if not name:
                continue

            prop = {
                'type': self._convert_param_type(param.get('type', 'string')),
                'description': param.get('description', '')
            }

            # 添加enum（如果有）
            if 'enum' in param:
                prop['enum'] = param['enum']

            # 添加default（如果有）
            if 'default' in param:
                prop['default'] = param['default']

            properties[name] = prop

            # 标记required
            if param.get('required', False):
                required.append(name)

        schema = {
            'type': 'object',
            'properties': properties
        }

        if required:
            schema['required'] = required

        return schema

    def to_qwen_tools(self, max_tool_output_chars: int = 4000) -> list:
        """
        将所有enabled Skills转换为qwen-agent的BaseTool实例列表

        Args:
            max_tool_output_chars: 工具输出最大字符数，超过将被截断

        Returns:
            qwen-agent BaseTool实例列表
        """
        BaseTool, register_tool = _get_qwen_tools()
        if BaseTool is None:
            logger.error("无法加载qwen-agent工具基类")
            return []

        tools = []
        enabled_skills = self.get_enabled_skills()

        for skill in enabled_skills:
            try:
                tool = self._create_qwen_tool(skill, BaseTool, max_tool_output_chars)
                if tool:
                    tools.append(tool)
            except Exception as e:
                logger.error(f"转换Skill '{skill.name}' 为qwen-tool失败: {e}", exc_info=True)
                # 单个Skill转换失败不影响其他
                continue

        logger.debug("转换 %d 个 Skills 为 qwen-agent 工具", len(tools))
        logger.info(f"成功转换 {len(tools)} 个Skills为qwen-agent工具")
        return tools

    def _create_qwen_tool(self, skill: BaseSkill, BaseTool, max_tool_output_chars: int = 4000):
        """
        为单个Skill创建qwen-agent工具类

        Args:
            skill: Skill实例
            BaseTool: qwen-agent的BaseTool基类
            max_tool_output_chars: 工具输出最大字符数，超过将被截断

        Returns:
            qwen-agent工具实例
        """
        import time

        # 构建类名（确保合法）
        class_name = f"Skill_{skill.name.replace('-', '_').replace('.', '_')}"

        # 构建JSON Schema
        parameters = self._build_json_schema(skill)

        # 创建动态类
        def make_call(skill_instance, max_output_chars):
            """创建绑定特定skill实例的call方法"""
            def call(self, params: str, **kwargs) -> str:
                import json
                try:
                    if isinstance(params, str):
                        param_dict = json.loads(params)
                    else:
                        param_dict = params
                    validated = skill_instance.validate_params(param_dict)
                    
                    # 执行Skill并记录时间
                    start_time = time.time()
                    result = skill_instance.execute(**validated)
                    duration = time.time() - start_time
                    
                    # 保存执行时间供外部获取
                    self._last_execution_duration = duration
                    
                    # 记录执行时间
                    logger.info("Skill [%s] 执行完成，耗时: %.2fs", skill_instance.name, duration)
                    
                    # 截断过长输出
                    if isinstance(result, str) and len(result) > max_output_chars:
                        original_len = len(result)
                        # 保留首80% + 尾15%
                        head = int(max_output_chars * 0.8)
                        tail = int(max_output_chars * 0.15)
                        result = (
                            result[:head]
                            + f"\n\n... [工具输出已截断: 原文{original_len}字符，保留{max_output_chars}字符] ...\n\n"
                            + result[-tail:]
                        )
                        logger.info("Skill [%s] 输出截断: %d -> %d 字符", skill_instance.name, original_len, len(result))
                    
                    return result
                except json.JSONDecodeError as e:
                    return json.dumps({"error": f"参数解析失败: {str(e)}"}, ensure_ascii=False)
                except ValueError as e:
                    return json.dumps({"error": str(e)}, ensure_ascii=False)
                except Exception as e:
                    return json.dumps({"error": f"执行失败: {str(e)}"}, ensure_ascii=False)
            return call

        # 定义__init__方法
        def make_init(skill_instance):
            def __init__(self, **kwargs):
                super(tool_class, self).__init__(**kwargs)
                self._skill_instance = skill_instance
            return __init__

        # 创建新类
        tool_class = type(
            class_name,
            (BaseTool,),
            {
                'name': skill.name,
                'description': skill.description,
                'parameters': parameters,
                '__init__': make_init(skill),
                'call': make_call(skill, max_tool_output_chars),
            }
        )

        # 实例化
        tool_instance = tool_class()
        logger.debug(f"创建qwen-agent工具: {skill.name}")
        return tool_instance
