#!/usr/bin/env python3
"""
TIR Agent 命令行接口
"""
import argparse
import os
import sys
from pathlib import Path

# 添加src目录到路径
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


def run_streamlit(args):
    """运行Streamlit应用"""
    import subprocess
    
    port = args.port or 8501
    host = args.host or "localhost"
    
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(Path(__file__).parent / "app.py"),
        "--server.port", str(port),
        "--server.address", host,
    ]
    
    if args.theme:
        cmd.extend(["--theme.base", args.theme])
    
    print(f"🚀 启动TIR Agent Streamlit应用...")
    print(f"📍 访问地址: http://{host}:{port}")
    
    subprocess.run(cmd)


def run_cli(args):
    """运行命令行交互"""
    from tir_agent import TIRAgent
    import uuid
    
    print("🤖 TIR Agent - 工具集成推理助手")
    print("输入 'quit' 或 'exit' 退出，输入 'clear' 清除历史")
    print("-" * 50)
    
    # 处理CLI参数
    memory_enabled = not args.no_memory
    planning_enabled = not args.no_planning
    skill_dirs = [args.skill_dir] if args.skill_dir else None
    
    # 从环境变量读取配置（支持动态配置）
    compression_threshold = int(os.getenv("MEMORY_COMPRESSION_THRESHOLD", "10"))
    complexity_threshold = os.getenv("COMPLEXITY_THRESHOLD", "auto")
    
    # 设置环境变量
    os.environ["MEMORY_ENABLED"] = str(memory_enabled).lower()
    os.environ["PLANNING_ENABLED"] = str(planning_enabled).lower()
    
    # 显示启动配置
    print(f"\n⚙️  配置信息:")
    print(f"   记忆系统: {'启用 ✅' if memory_enabled else '禁用 ❌'}")
    print(f"   任务规划: {'启用 ✅' if planning_enabled else '禁用 ❌'}")
    print(f"   压缩阈值: {compression_threshold}")
    print(f"   复杂度策略: {complexity_threshold}")
    if skill_dirs:
        print(f"   额外Skill目录: {skill_dirs}")
    print("-" * 50)
    
    # 构建配置
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
    
    # 初始化Agent
    agent = TIRAgent(
        llm_config=llm_config,
        vlm_config=vlm_config,
        memory_enabled=memory_enabled,
        planning_enabled=planning_enabled,
        skill_dirs=skill_dirs,
        memory_compression_threshold=compression_threshold,
        complexity_threshold=complexity_threshold,
    )
    
    # 生成session_id
    session_id = str(uuid.uuid4())
    print(f"\n📍 Session ID: {session_id}")
    messages = []
    
    while True:
        try:
            user_input = input("\n👤 你: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ["quit", "exit"]:
                print("👋 再见！")
                break
            
            if user_input.lower() == "clear":
                messages = []
                print("✅ 对话历史已清除")
                continue
            
            # 处理文件路径（如果有）
            files = None
            if user_input.startswith("@file:"):
                parts = user_input.split(" ", 1)
                file_path = parts[0].replace("@file:", "")
                user_input = parts[1] if len(parts) > 1 else "请分析这个文件"
                files = [file_path]
            
            messages.append({"role": "user", "content": user_input})
            
            print("\n🤖 助手: ", end="")
            
            response_text = ""
            for response in agent.run(messages=messages, files=files, session_id=session_id):
                if response:
                    last_msg = response[-1] if response else {}
                    content = last_msg.get("content", "")
                    if content and content != response_text:
                        print(content[len(response_text):], end="", flush=True)
                        response_text = content
            
            print()
            print("\n✅ 回复完成")

            messages.append({"role": "assistant", "content": response_text})
            
        except KeyboardInterrupt:
            print("\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="TIR Agent - 工具集成推理助手",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # Streamlit命令
    web_parser = subparsers.add_parser("web", help="启动Web界面")
    web_parser.add_argument("--port", type=int, default=8501, help="端口号")
    web_parser.add_argument("--host", type=str, default="localhost", help="主机地址")
    web_parser.add_argument("--theme", type=str, choices=["light", "dark"], help="主题")
    web_parser.set_defaults(func=run_streamlit)
    
    # CLI命令
    cli_parser = subparsers.add_parser("cli", help="启动命令行交互")
    cli_parser.add_argument("--memory", action="store_true", default=True, help="启用记忆系统")
    cli_parser.add_argument("--no-memory", action="store_true", help="禁用记忆系统")
    cli_parser.add_argument("--planning", action="store_true", default=True, help="启用任务规划")
    cli_parser.add_argument("--no-planning", action="store_true", help="禁用任务规划")
    cli_parser.add_argument("--skill-dir", type=str, default=None, help="额外的Skill目录路径")
    cli_parser.set_defaults(func=run_cli)
    
    args = parser.parse_args()
    
    if args.command is None:
        # 默认启动Web界面
        args.func = run_streamlit
        args.port = 8501
        args.host = "localhost"
        args.theme = None
    
    args.func(args)


if __name__ == "__main__":
    main()
