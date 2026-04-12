"""
Skill加载器

提供从YAML文件和Python模块动态加载Skill的功能。
"""
import os
import logging
import importlib.util
from pathlib import Path
from typing import Optional

import yaml

from .base import BaseSkill, SkillMetadata

logger = logging.getLogger("tir_agent.skills")


class SkillLoader:
    """Skill加载器，从目录加载YAML+Python实现的Skill"""

    def load_skill_from_dir(self, skill_dir: str) -> Optional[BaseSkill]:
        """
        从目录加载单个Skill

        Args:
            skill_dir: Skill目录路径，包含skill.yaml和实现文件

        Returns:
            BaseSkill实例，加载失败返回None
        """
        skill_dir = Path(skill_dir)
        yaml_path = skill_dir / "skill.yaml"
        
        logger.debug("加载 Skill: %s (from %s)", skill_dir.name, skill_dir)

        if not yaml_path.exists():
            logger.warning(f"Skill目录 {skill_dir} 中未找到 skill.yaml")
            return None

        try:
            # 1. 读取并解析YAML
            with open(yaml_path, 'r', encoding='utf-8') as f:
                yaml_content = yaml.safe_load(f)

            if not yaml_content:
                logger.warning(f"Skill YAML文件为空: {yaml_path}")
                return None

            # 2. 解析为SkillMetadata
            metadata = self._parse_metadata(yaml_content)

            # 3. 根据entry_point动态导入Python模块
            entry_point = metadata.entry_point
            module_path, class_name = self._parse_entry_point(entry_point)

            # 解析模块文件路径
            if os.path.isabs(module_path):
                full_module_path = module_path
            else:
                full_module_path = skill_dir / module_path

            if not os.path.exists(full_module_path):
                logger.error(f"Skill模块文件不存在: {full_module_path}")
                return None

            # 动态导入模块
            skill_class = self._import_skill_class(full_module_path, class_name)
            if skill_class is None:
                return None

            # 4. 实例化Skill类
            # 获取config（如果有）
            config = yaml_content.get('config', {})

            skill_instance = skill_class(metadata=metadata, config=config)
            logger.info(f"成功加载Skill: {metadata.name} v{metadata.version}")
            return skill_instance

        except yaml.YAMLError as e:
            logger.error("加载 Skill 失败 [%s]: %s", skill_dir, e)
            return None
        except Exception as e:
            logger.error("加载 Skill 失败 [%s]: %s", skill_dir, e)
            return None

    def discover_skills(self, base_dir: str) -> list[BaseSkill]:
        """
        扫描base_dir下所有包含skill.yaml的子目录，逐个加载

        Args:
            base_dir: 基础目录路径

        Returns:
            成功加载的Skill列表
        """
        base_dir = Path(base_dir)
        skills = []

        if not base_dir.exists() or not base_dir.is_dir():
            logger.warning(f"Skill扫描目录不存在: {base_dir}")
            return skills

        # 遍历所有子目录
        for item in base_dir.iterdir():
            if item.is_dir():
                yaml_path = item / "skill.yaml"
                if yaml_path.exists():
                    skill = self.load_skill_from_dir(str(item))
                    if skill:
                        skills.append(skill)

        logger.info(f"在 {base_dir} 中发现 {len(skills)} 个Skills")
        return skills

    def _parse_metadata(self, yaml_content: dict) -> SkillMetadata:
        """解析YAML内容为SkillMetadata"""
        # 提取必需字段
        name = yaml_content.get('name')
        if not name:
            raise ValueError("Skill YAML缺少必需的'name'字段")

        display_name = yaml_content.get('display_name', name)
        version = yaml_content.get('version', '1.0.0')
        description = yaml_content.get('description', '')
        entry_point = yaml_content.get('entry_point')
        if not entry_point:
            raise ValueError("Skill YAML缺少必需的'entry_point'字段")

        # 解析parameters
        parameters = yaml_content.get('parameters', [])
        if not isinstance(parameters, list):
            parameters = []

        # 解析returns
        returns = yaml_content.get('returns', {'type': 'string', 'description': '执行结果'})
        if not isinstance(returns, dict):
            returns = {'type': 'string', 'description': str(returns)}

        # 解析tags
        tags = yaml_content.get('tags', [])
        if not isinstance(tags, list):
            tags = []

        # 解析enabled
        enabled = yaml_content.get('enabled', True)

        return SkillMetadata(
            name=name,
            display_name=display_name,
            version=version,
            description=description,
            parameters=parameters,
            returns=returns,
            entry_point=entry_point,
            tags=tags,
            enabled=enabled
        )

    def _parse_entry_point(self, entry_point: str) -> tuple[str, str]:
        """
        解析entry_point字符串

        Args:
            entry_point: 格式如 "executor.py:CodeExecutorSkill" 或 "module.py:ClassName"

        Returns:
            (module_path, class_name) 元组
        """
        if ':' in entry_point:
            module_path, class_name = entry_point.rsplit(':', 1)
        else:
            # 默认使用Skill名称作为类名
            module_path = entry_point
            class_name = "Skill"

        return module_path.strip(), class_name.strip()

    def _import_skill_class(self, module_path: str, class_name: str):
        """
        动态导入模块并获取Skill类

        Args:
            module_path: Python模块文件路径
            class_name: 类名

        Returns:
            Skill类，失败返回None
        """
        try:
            # 使用importlib动态导入
            module_name = Path(module_path).stem
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec is None or spec.loader is None:
                logger.error(f"无法创建模块spec: {module_path}")
                return None

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # 获取类
            if not hasattr(module, class_name):
                logger.error(f"模块 {module_path} 中未找到类: {class_name}")
                return None

            skill_class = getattr(module, class_name)

            # 验证类是否继承自BaseSkill
            if not issubclass(skill_class, BaseSkill):
                logger.error(f"类 {class_name} 必须继承自 BaseSkill")
                return None

            return skill_class

        except Exception as e:
            logger.error(f"导入模块失败 {module_path}: {e}", exc_info=True)
            return None
