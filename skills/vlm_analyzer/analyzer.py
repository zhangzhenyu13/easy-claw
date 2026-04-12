"""
VLM图像分析器Skill实现
内联了VLMTool的核心逻辑
"""
import base64
import json
import os
from typing import List, Optional, Union, Dict, Any
import logging

from tir_agent.skills.base import BaseSkill, SkillMetadata
from tir_agent.config import settings

# 配置日志
logger = logging.getLogger(__name__)


class VLMAnalyzerSkill(BaseSkill):
    """VLM图像分析器Skill - 直接内联VLM调用逻辑"""

    def __init__(self, metadata: SkillMetadata, config: dict = None):
        super().__init__(metadata, config)
        # 从配置获取VLM配置
        vlm_config = settings.get_vlm_config()
        self.model = vlm_config.get("model", "qwen-vl-max")
        self.api_key = vlm_config.get("api_key") or os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = vlm_config.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.extra_kwargs = config or {}
        self._client = None

    @property
    def client(self):
        """延迟加载OpenAI客户端"""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
            except ImportError:
                raise ImportError("请安装openai库: pip install openai")
        return self._client

    def encode_image_to_base64(self, image_path: str) -> str:
        """将图片编码为base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def get_image_url(self, image_source: str) -> str:
        """
        获取图片URL（支持文件路径、base64、URL）

        Args:
            image_source: 图片源（文件路径、base64字符串或URL）

        Returns:
            str: 可用于API调用的图片URL
        """
        # 如果是URL，直接返回
        if image_source.startswith(("http://", "https://")):
            return image_source

        # 如果是base64编码的图片数据
        if image_source.startswith("data:image"):
            return image_source

        # 如果是base64字符串（不带前缀）
        if len(image_source) > 200 and "/" not in image_source and "\\" not in image_source:
            return f"data:image/jpeg;base64,{image_source}"

        # 否则当作文件路径处理
        if os.path.exists(image_source):
            base64_data = self.encode_image_to_base64(image_source)
            ext = os.path.splitext(image_source)[1].lower()
            mime_type = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".gif": "image/gif",
                ".webp": "image/webp",
            }.get(ext, "image/jpeg")
            return f"data:{mime_type};base64,{base64_data}"

        # 假设是base64
        return f"data:image/jpeg;base64,{image_source}"

    def analyze_image(
        self,
        image: Union[str, List[str]],
        prompt: str = "请详细描述这张图片的内容。",
        **kwargs
    ) -> str:
        """
        分析单张或多张图片 - 内联自原VLMTool

        Args:
            image: 图片源（文件路径、base64、URL或列表）
            prompt: 分析提示词

        Returns:
            str: 分析结果
        """
        images = [image] if isinstance(image, str) else image
        logger.info(f"🖼️ VLM分析图片: {len(images)} 张, 模型: {self.model}")

        # 构建消息内容
        content = []

        # 添加图片
        for i, img in enumerate(images):
            logger.info(f"   处理图片 {i+1}: {img[:100] if len(img) > 100 else img}...")
            img_url = self.get_image_url(img)
            content.append({
                "type": "image_url",
                "image_url": {"url": img_url}
            })

        # 添加文本提示
        content.append({
            "type": "text",
            "text": prompt
        })
        logger.info(f"   提示词: {prompt[:100]}...")

        messages = [{"role": "user", "content": content}]

        try:
            logger.info(f"   调用VLM API...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **kwargs
            )
            result = response.choices[0].message.content
            logger.info(f"   ✅ VLM调用成功，返回内容长度: {len(result)}")
            return result
        except Exception as e:
            logger.error(f"   ❌ VLM调用失败: {e}", exc_info=True)
            return f"[图像分析失败: {str(e)}]"

    def ocr_image(
        self,
        image: Union[str, List[str]],
        **kwargs
    ) -> str:
        """
        对图片进行OCR识别 - 内联自原VLMTool

        Args:
            image: 图片源

        Returns:
            str: 识别结果
        """
        prompt = """请识别图片中的所有文字内容，并按照原始格式输出。
如果图片中有表格，请用Markdown表格格式输出。
如果图片中有公式，请用LaTeX格式输出。
只输出识别到的文字内容，不要添加额外的解释。"""

        return self.analyze_image(image, prompt, **kwargs)

    def describe_image(
        self,
        image: Union[str, List[str]],
        detail_level: str = "medium",
        **kwargs
    ) -> str:
        """
        描述图片内容 - 内联自原VLMTool

        Args:
            image: 图片源
            detail_level: 详细程度 (brief/medium/detailed)

        Returns:
            str: 描述内容
        """
        level_prompts = {
            "brief": "请用一句话简要描述这张图片的主要内容。",
            "medium": "请详细描述这张图片的内容，包括主要元素、场景和氛围。",
            "detailed": """请非常详细地描述这张图片，包括：
1. 图片的整体场景和主题
2. 主要物体和人物
3. 颜色、光线和构图
4. 图片传达的情感或信息
5. 任何值得注意的细节"""
        }

        prompt = level_prompts.get(detail_level, level_prompts["medium"])
        return self.analyze_image(image, prompt, **kwargs)

    def answer_question(
        self,
        image: Union[str, List[str]],
        question: str,
        **kwargs
    ) -> str:
        """
        基于图片回答问题 - 内联自原VLMTool

        Args:
            image: 图片源
            question: 问题

        Returns:
            str: 回答
        """
        return self.analyze_image(image, question, **kwargs)

    def compare_images(
        self,
        images: List[str],
        aspect: str = "overall",
        **kwargs
    ) -> str:
        """
        比较多张图片 - 内联自原VLMTool

        Args:
            images: 图片列表
            aspect: 比较方面 (overall/similarity/difference)

        Returns:
            str: 比较结果
        """
        aspect_prompts = {
            "overall": "请比较这些图片，描述它们之间的相同点和不同点。",
            "similarity": "请详细描述这些图片之间的相似之处。",
            "difference": "请详细描述这些图片之间的不同之处。"
        }

        prompt = aspect_prompts.get(aspect, aspect_prompts["overall"])
        return self.analyze_image(images, prompt, **kwargs)

    def extract_chart_data(
        self,
        image: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        从图表图片中提取数据 - 内联自原VLMTool

        Args:
            image: 图表图片

        Returns:
            dict: 提取的数据
        """
        prompt = """请分析这个图表，提取其中的数据，并以JSON格式返回。
返回格式：
{
    "chart_type": "图表类型(bar/line/pie/scatter等)",
    "title": "图表标题",
    "x_label": "X轴标签",
    "y_label": "Y轴标签",
    "data": [
        {"label": "标签", "value": 数值}
    ],
    "summary": "图表内容简述"
}
只返回JSON，不要添加其他内容。"""

        result = self.analyze_image(image, prompt, **kwargs)

        try:
            # 尝试解析JSON
            # 去除可能的markdown代码块标记
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0]
            elif "```" in result:
                result = result.split("```")[1].split("```")[0]

            return json.loads(result.strip())
        except json.JSONDecodeError:
            return {
                "chart_type": "unknown",
                "raw_response": result,
                "error": "无法解析为JSON"
            }

    def execute(self, **params) -> str:
        """
        Skill接口方法 - 执行图像分析

        Args:
            action: 分析动作类型
            image_path: 图片路径（单张或多张，多张用逗号分隔）
            prompt: 自定义提示词
            question: 问题
            detail_level: 描述详细程度
            aspect: 比较方面

        Returns:
            str: 分析结果
        """
        try:
            # 验证参数
            validated_params = self.validate_params(params)
            action = validated_params.get("action")
            image_path = validated_params.get("image_path", "")

            # 处理多张图片路径
            if "," in image_path:
                images = [p.strip() for p in image_path.split(",")]
            else:
                images = image_path

            # 根据动作类型调用不同方法
            if action == "analyze":
                prompt = validated_params.get("prompt", "请详细描述这张图片的内容。")
                return self.analyze_image(images, prompt)

            elif action == "ocr":
                return self.ocr_image(images)

            elif action == "describe":
                detail_level = validated_params.get("detail_level", "medium")
                return self.describe_image(images, detail_level)

            elif action == "extract_chart":
                # extract_chart_data返回dict，需要转换为字符串
                result = self.extract_chart_data(images)
                return json.dumps(result, ensure_ascii=False, indent=2)

            elif action == "answer_question":
                question = validated_params.get("question") or validated_params.get("prompt", "")
                if not question:
                    return "错误: answer_question动作需要提供question或prompt参数"
                return self.answer_question(images, question)

            elif action == "compare_images":
                if not isinstance(images, list) or len(images) < 2:
                    return "错误: compare_images动作需要提供至少2张图片路径（用逗号分隔）"
                aspect = validated_params.get("aspect", "overall")
                return self.compare_images(images, aspect)

            else:
                return f"错误: 不支持的动作类型 '{action}'"

        except ValueError as e:
            return f"参数错误: {str(e)}"
        except Exception as e:
            return f"图像分析失败: {str(e)}"
