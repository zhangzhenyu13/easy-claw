"""
TIR Agent Streamlit前端界面
支持多文件上传和对话
"""
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional
import streamlit as st
from PIL import Image

# 添加src目录到路径
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from tir_agent import TIRAgent, settings, reload_settings


# 页面配置
st.set_page_config(
    page_title="TIR Agent - 工具集成推理助手",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 自定义CSS
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .user-message {
        background-color: #f0f2f6;
    }
    .assistant-message {
        background-color: #e8f4ea;
    }
    .file-preview {
        max-height: 200px;
        overflow-y: auto;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .image-preview {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }
    .image-preview img {
        max-width: 150px;
        max-height: 150px;
        object-fit: contain;
        border-radius: 5px;
        border: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """初始化session state"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "agent" not in st.session_state:
        st.session_state.agent = None
    
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    
    if "file_paths" not in st.session_state:
        st.session_state.file_paths = []
    
    if "processing" not in st.session_state:
        st.session_state.processing = False
    
    # session_id管理
    if "session_id" not in st.session_state:
        import uuid
        st.session_state.session_id = str(uuid.uuid4())


def get_agent(
    memory_enabled: bool = True,
    planning_enabled: bool = True,
    compression_threshold: int = 10,
    complexity_mode: str = "auto"
) -> TIRAgent:
    """获取或创建Agent实例"""
    # 检查配置是否变更
    current_config = {
        "model_name": os.getenv("MODEL_NAME", ""),
        "model_type": os.getenv("MODEL_TYPE", ""),
        "vlm_model": os.getenv("VLM_MODEL_NAME", ""),
        "memory_enabled": memory_enabled,
        "planning_enabled": planning_enabled,
        "compression_threshold": compression_threshold,
        "complexity_mode": complexity_mode,
    }
    
    if "config" not in st.session_state:
        st.session_state.config = None
    
    config_changed = st.session_state.config != current_config
    
    if st.session_state.agent is None or config_changed:
        with st.spinner("正在初始化Agent..."):
            try:
                # 清理旧的agent
                if st.session_state.agent:
                    st.session_state.agent.cleanup()
                
                # 构建LLM和VLM配置
                llm_config = {
                    "model": os.getenv("MODEL_NAME", "qwen-max-latest"),
                    "model_server": "https://dashscope.aliyuncs.com/compatible-mode/v1" 
                        if os.getenv("MODEL_TYPE") == "qwen_dashscope" else os.getenv("OPENAI_API_BASE"),
                    "api_key": os.getenv("DASHSCOPE_API_KEY") if os.getenv("MODEL_TYPE") == "qwen_dashscope" 
                        else os.getenv("OPENAI_API_KEY"),
                }
                
                vlm_config = {
                    "model": os.getenv("VLM_MODEL_NAME", os.getenv("MODEL_NAME", "qwen-vl-max")),
                    "api_key": os.getenv("VLM_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY"),
                    "base_url": os.getenv("VLM_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
                }
                
                # 设置环境变量以影响Settings
                os.environ["MEMORY_ENABLED"] = str(memory_enabled).lower()
                os.environ["PLANNING_ENABLED"] = str(planning_enabled).lower()
                os.environ["MEMORY_COMPRESSION_THRESHOLD"] = str(compression_threshold)
                os.environ["COMPLEXITY_THRESHOLD"] = complexity_mode
                
                # 重新加载settings确保使用最新环境变量
                reload_settings()
                
                st.session_state.agent = TIRAgent(
                    llm_config=llm_config,
                    vlm_config=vlm_config,
                    memory_enabled=memory_enabled,
                    planning_enabled=planning_enabled,
                    memory_compression_threshold=compression_threshold,
                    complexity_threshold=complexity_mode,
                )
                st.session_state.config = current_config
                st.success(f"✅ Agent已初始化 (LLM: {current_config['model_name']}, Memory: {'开' if memory_enabled else '关'}, Planning: {'开' if planning_enabled else '关'})")
            except Exception as e:
                st.error(f"Agent初始化失败: {e}")
                return None
    return st.session_state.agent


def save_uploaded_file(uploaded_file) -> str:
    """保存上传的文件到临时目录"""
    temp_dir = tempfile.mkdtemp(prefix="tir_upload_")
    file_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path


def get_file_type(file_path: str) -> str:
    """根据文件扩展名获取文件类型"""
    ext = Path(file_path).suffix.lower()
    image_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'}
    
    if ext in image_exts:
        return "image"
    elif ext == '.pdf':
        return "pdf"
    elif ext in {'.docx', '.doc'}:
        return "word"
    elif ext in {'.xlsx', '.xls'}:
        return "excel"
    elif ext in {'.pptx', '.ppt'}:
        return "ppt"
    elif ext in {'.txt', '.md'}:
        return "text"
    else:
        return "unknown"


def display_file_preview(file_paths: List[str]):
    """显示文件预览"""
    st.markdown("### 📁 已上传文件")
    
    cols = st.columns(min(len(file_paths), 4))
    
    for i, file_path in enumerate(file_paths):
        with cols[i % 4]:
            file_type = get_file_type(file_path)
            file_name = Path(file_path).name
            
            if file_type == "image":
                try:
                    img = Image.open(file_path)
                    st.image(img, caption=file_name, use_container_width=True)
                except:
                    st.write(f"🖼️ {file_name}")
            else:
                emoji = {
                    "pdf": "📄",
                    "word": "📝",
                    "excel": "📊",
                    "ppt": "📽️",
                    "text": "📃",
                }.get(file_type, "📎")
                st.write(f"{emoji} {file_name}")


def display_chat_message(message: dict):
    """显示聊天消息"""
    role = message.get("role", "user")
    content = message.get("content", "")
    
    with st.chat_message(role):
        # 检查是否包含代码块
        if "```" in content:
            # 简单的代码高亮
            st.markdown(content)
        else:
            st.markdown(content)


def main():
    """主函数"""
    init_session_state()
    
    # 加载.env配置作为默认值
    from dotenv import load_dotenv
    load_dotenv()
    
    # 侧边栏
    with st.sidebar:
        st.title("🤖 TIR Agent")
        st.markdown("**工具集成推理助手**")
        st.markdown("---")
        
        # === LLM模型配置 ===
        st.subheader("🧠 LLM模型配置")
        st.caption("用于文本对话和推理")
        
        # 从环境变量获取默认值
        default_llm_type = os.getenv("MODEL_TYPE", "qwen_dashscope")
        llm_type_index = 0 if default_llm_type == "qwen_dashscope" else 1
        
        llm_type = st.selectbox(
            "LLM服务类型",
            options=["qwen_dashscope", "openai_compatible"],
            index=llm_type_index,
            help="选择主语言模型的服务类型"
        )
        
        if llm_type == "qwen_dashscope":
            llm_api_key = st.text_input(
                "DashScope API Key",
                type="password",
                value=os.getenv("DASHSCOPE_API_KEY", ""),
                help="输入您的DashScope API密钥"
            )
            if llm_api_key:
                os.environ["DASHSCOPE_API_KEY"] = llm_api_key
            
            # 支持预设+自定义的模型选择
            qwen_models = [
                "qwen3.6-plus",
                "qwen-max-latest",
                "qwen-max",
                "qwen-plus",
                "qwen-turbo",
                "qwen-long",
                "qwen2.5-72b-instruct",
                "qwen2.5-32b-instruct",
                "qwen2.5-14b-instruct",
                "qwen2.5-7b-instruct",
            ]
            
            default_model = os.getenv("MODEL_NAME", "qwen3.6-plus")
            
            # 使用columns实现预设+自定义
            col1, col2 = st.columns([3, 1])
            with col1:
                use_custom = st.checkbox("自定义模型", value=(default_model not in qwen_models))
            
            if use_custom:
                llm_model = st.text_input(
                    "模型名称",
                    value=default_model,
                    help="输入自定义模型名称"
                )
            else:
                # 找到默认模型的索引
                default_index = qwen_models.index(default_model) if default_model in qwen_models else 0
                llm_model = st.selectbox(
                    "LLM模型",
                    options=qwen_models,
                    index=default_index,
                    help="选择模型版本\n- qwen3.6-plus: 最新推荐\n- qwen-max-latest: 旗舰版\n- qwen-plus: 性价比高\n- qwen-turbo: 快速响应"
                )
        else:
            llm_api_key = st.text_input(
                "OpenAI API Key",
                type="password",
                value=os.getenv("OPENAI_API_KEY", ""),
                help="OpenAI或兼容服务的API密钥"
            )
            llm_base_url = st.text_input(
                "API Base URL",
                value=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
                help="API服务地址"
            )
            
            # OpenAI兼容模型列表
            openai_models = [
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4-turbo",
                "gpt-3.5-turbo",
                "deepseek-chat",
                "deepseek-coder",
                "claude-3-opus",
                "claude-3-sonnet",
            ]
            
            default_model = os.getenv("MODEL_NAME", "gpt-4o")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                use_custom = st.checkbox("自定义模型", value=(default_model not in openai_models))
            
            if use_custom:
                llm_model = st.text_input(
                    "模型名称",
                    value=default_model,
                    help="如: gpt-4o, deepseek-chat, claude-3-opus 等"
                )
            else:
                default_index = openai_models.index(default_model) if default_model in openai_models else 0
                llm_model = st.selectbox(
                    "LLM模型",
                    options=openai_models,
                    index=default_index,
                    help="选择模型版本"
                )
            
            if llm_api_key:
                os.environ["OPENAI_API_KEY"] = llm_api_key
            if llm_base_url:
                os.environ["OPENAI_API_BASE"] = llm_base_url
        
        os.environ["MODEL_NAME"] = llm_model
        os.environ["MODEL_TYPE"] = llm_type
        
        st.markdown("---")
        
        # === VLM模型配置 ===
        st.subheader("👁️ VLM模型配置")
        st.caption("用于图像理解和分析")
        
        # 检查是否有单独的VLM配置
        has_vlm_config = bool(os.getenv("VLM_API_KEY") or os.getenv("VLM_MODEL_NAME"))
        vlm_source_index = 1 if has_vlm_config else 0
        
        vlm_source = st.radio(
            "VLM配置来源",
            options=["使用LLM相同配置", "单独配置VLM"],
            index=vlm_source_index,
            horizontal=True
        )
        
        if vlm_source == "单独配置VLM":
            default_vlm_type = "qwen_vl_dashscope" if not os.getenv("VLM_API_BASE") else "openai_compatible"
            vlm_type_index = 0 if default_vlm_type == "qwen_vl_dashscope" else 1
            
            vlm_type = st.selectbox(
                "VLM服务类型",
                options=["qwen_vl_dashscope", "openai_compatible"],
                index=vlm_type_index
            )
            
            if vlm_type == "qwen_vl_dashscope":
                vlm_api_key = st.text_input(
                    "VLM DashScope API Key",
                    type="password",
                    value=os.getenv("VLM_API_KEY", os.getenv("DASHSCOPE_API_KEY", "")),
                    help="VLM专用的DashScope API密钥（留空则使用LLM的密钥）"
                )
                
                # VLM模型预设+自定义
                vlm_models = [
                    "qwen-vl-max",
                    "qwen-vl-max-latest",
                    "qwen-vl-plus",
                    "qwen-vl-ocr",
                    "qwen2.5-vl-72b-instruct",
                    "qwen2.5-vl-32b-instruct",
                ]
                
                default_vlm_model = os.getenv("VLM_MODEL_NAME", "qwen-vl-max")
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    use_custom_vlm = st.checkbox("自定义VLM模型", value=(default_vlm_model not in vlm_models), key="custom_vlm_qwen")
                
                if use_custom_vlm:
                    vlm_model = st.text_input(
                        "VLM模型名称",
                        value=default_vlm_model,
                        help="输入自定义VLM模型名称"
                    )
                else:
                    default_index = vlm_models.index(default_vlm_model) if default_vlm_model in vlm_models else 0
                    vlm_model = st.selectbox(
                        "VLM模型",
                        options=vlm_models,
                        index=default_index,
                        help="选择VLM模型版本\n- qwen-vl-max: 最强视觉理解\n- qwen-vl-plus: 平衡性能\n- qwen-vl-ocr: 专注OCR"
                    )
                
                if vlm_api_key:
                    os.environ["VLM_API_KEY"] = vlm_api_key
            else:
                vlm_api_key = st.text_input(
                    "VLM API Key",
                    type="password",
                    value=os.getenv("VLM_API_KEY", os.getenv("OPENAI_API_KEY", "")),
                )
                vlm_base_url = st.text_input(
                    "VLM API Base URL",
                    value=os.getenv("VLM_API_BASE", "https://api.openai.com/v1"),
                )
                
                vlm_models = ["gpt-4o", "gpt-4o-mini", "gpt-4-vision-preview", "claude-3-opus"]
                default_vlm_model = os.getenv("VLM_MODEL_NAME", "gpt-4o")
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    use_custom_vlm = st.checkbox("自定义VLM模型", value=(default_vlm_model not in vlm_models), key="custom_vlm_openai")
                
                if use_custom_vlm:
                    vlm_model = st.text_input(
                        "VLM模型名称",
                        value=default_vlm_model,
                        help="如: gpt-4o, gpt-4-vision-preview 等"
                    )
                else:
                    default_index = vlm_models.index(default_vlm_model) if default_vlm_model in vlm_models else 0
                    vlm_model = st.selectbox(
                        "VLM模型",
                        options=vlm_models,
                        index=default_index,
                    )
                
                if vlm_api_key:
                    os.environ["VLM_API_KEY"] = vlm_api_key
                if vlm_base_url:
                    os.environ["VLM_API_BASE"] = vlm_base_url
            
            os.environ["VLM_MODEL_NAME"] = vlm_model
        else:
            vlm_model = llm_model
        
        st.markdown("---")
        
        # === Memory配置面板 ===
        st.subheader("🧠 记忆系统")
        memory_enabled = st.checkbox("启用记忆", value=True, help="开启后Agent会记住历史对话和工具调用")
        compression_threshold = 10
        if memory_enabled:
            compression_threshold = st.slider("压缩阈值（对话轮次）", 5, 30, 10, help="对话超过此轮次自动压缩")
        
        st.markdown("---")
        
        # === Skills管理面板 ===
        st.subheader("🛠️ 技能管理")
        st.caption("已加载的Skills将在Agent初始化后显示")
        # Skills列表将在Agent初始化后动态展示
        
        st.markdown("---")
        
        # === Planning配置面板 ===
        st.subheader("📋 任务规划")
        planning_enabled = st.checkbox("启用规划", value=True, help="对复杂任务自动分解规划")
        complexity_mode = "auto"
        if planning_enabled:
            complexity_mode = st.selectbox("规划模式", ["auto", "always", "never"], help="auto=自动判断复杂度")
        
        st.markdown("---")
        
        # === 当前配置摘要 ===
        with st.expander("📋 当前配置摘要", expanded=False):
            st.markdown(f"""
            **LLM配置:**
            - 服务类型: `{llm_type}`
            - 模型: `{llm_model}`
            - API Key: `{'已配置 ✅' if (llm_type == 'qwen_dashscope' and os.getenv('DASHSCOPE_API_KEY')) or (llm_type != 'qwen_dashscope' and os.getenv('OPENAI_API_KEY')) else '未配置 ⚠️'}`
            
            **VLM配置:**
            - 来源: `{vlm_source}`
            {'- 模型: `' + vlm_model + '`' if vlm_source == '单独配置VLM' else '- 使用LLM相同配置'}
            
            **Memory配置:**
            - 启用: `{'是' if memory_enabled else '否'}`
            - 压缩阈值: `{compression_threshold}` 轮
            
            **Planning配置:**
            - 启用: `{'是' if planning_enabled else '否'}`
            - 规划模式: `{complexity_mode}`
            """)
        
        st.markdown("---")
        
        # 功能说明
        st.subheader("📖 功能说明")
        st.markdown("""
        **支持的功能：**
        - 📄 多文件上传（图片、PDF、Word、Excel、PPT）
        - 💬 智能对话问答
        - 🖼️ 图像理解和分析
        - 📊 数据处理和计算
        - 💻 Python代码执行
        
        **支持的文件格式：**
        - 图片: JPG, PNG, GIF, WebP
        - 文档: PDF, DOCX, XLSX, PPTX
        - 文本: TXT, MD
        """)
        
        st.markdown("---")
        
        # 清除对话
        if st.button("🗑️ 清除对话历史", use_container_width=True):
            st.session_state.messages = []
            st.session_state.file_paths = []
            st.session_state.uploaded_files = []
            st.rerun()
    
    # 主内容区
    st.title("💬 智能对话")
    
    # 文件上传区
    with st.expander("📎 上传文件", expanded=False):
        uploaded_files = st.file_uploader(
            "选择文件（支持多选）",
            type=["jpg", "jpeg", "png", "gif", "webp", "pdf", "docx", "doc", "xlsx", "xls", "pptx", "ppt", "txt", "md"],
            accept_multiple_files=True,
            help="支持图片、PDF、Office文档和文本文件"
        )
        
        if uploaded_files:
            # 保存上传的文件
            new_file_paths = []
            for uploaded_file in uploaded_files:
                file_path = save_uploaded_file(uploaded_file)
                new_file_paths.append(file_path)
            
            st.session_state.file_paths = new_file_paths
            
            # 显示文件预览
            display_file_preview(new_file_paths)
            
            if st.button("🗑️ 清除文件"):
                st.session_state.file_paths = []
                st.session_state.uploaded_files = []
                st.rerun()
    
    # 显示历史消息
    for message in st.session_state.messages:
        display_chat_message(message)
    
    # 聊天输入
    if prompt := st.chat_input("输入您的问题..."):
        # 获取Agent（传入配置参数）
        agent = get_agent(
            memory_enabled=memory_enabled,
            planning_enabled=planning_enabled,
            compression_threshold=compression_threshold,
            complexity_mode=complexity_mode
        )
        if agent is None:
            st.error("Agent未初始化，请检查配置")
            return
        
        # 在侧边栏显示已加载的Skills
        try:
            if hasattr(agent, 'skill_registry'):
                skills = agent.skill_registry.list_skills()
                with st.sidebar:
                    st.markdown("---")
                    st.subheader("🛠️ 已加载Skills")
                    if skills:
                        for skill in skills:
                            desc = skill['description'][:50] + "..." if len(skill['description']) > 50 else skill['description']
                            st.caption(f"**{skill['display_name']}**: {desc}")
                    else:
                        st.caption("暂无Skills加载")
        except Exception:
            pass  # Skills展示失败不影响主功能
        
        # 添加用户消息
        user_message = {"role": "user", "content": prompt}
        st.session_state.messages.append(user_message)
        
        # 显示用户消息
        with st.chat_message("user"):
            st.markdown(prompt)
            
            # 如果有文件，显示文件信息
            if st.session_state.file_paths:
                st.info(f"📎 已附加 {len(st.session_state.file_paths)} 个文件")
        
        # 生成助手回复
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                # 准备消息
                messages = st.session_state.messages.copy()
                
                # 调用Agent
                file_paths = st.session_state.file_paths if st.session_state.file_paths else None
                
                # 流式输出，传入session_id用于Memory记录
                for response in agent.run(
                    messages=messages, 
                    files=file_paths,
                    session_id=st.session_state.session_id
                ):
                    if response:
                        last_msg = response[-1] if response else {}
                        content = last_msg.get("content", "")
                        
                        if content:
                            full_response = content
                            message_placeholder.markdown(full_response + "▌")
                
                # 最终输出
                message_placeholder.markdown(full_response)
                
                # 添加助手消息到历史
                assistant_message = {"role": "assistant", "content": full_response}
                st.session_state.messages.append(assistant_message)
                
                # 清除已处理的文件
                st.session_state.file_paths = []
                
            except Exception as e:
                error_msg = f"抱歉，处理您的请求时出现错误: {str(e)}"
                message_placeholder.markdown(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })


if __name__ == "__main__":
    main()
