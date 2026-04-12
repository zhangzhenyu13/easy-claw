"""
TIR Agent 使用示例
"""
import sys
from pathlib import Path

# 添加src目录到路径
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


def example_basic_chat():
    """基本对话示例"""
    from tir_agent import TIRAgent
    
    agent = TIRAgent()
    
    # 简单对话
    response = agent.chat("你好，请介绍一下你自己")
    print(f"回复: {response}")


def example_file_analysis():
    """文件分析示例"""
    from tir_agent import TIRAgent
    
    agent = TIRAgent()
    
    # 分析PDF文件
    response = agent.chat(
        "请总结这个PDF文件的主要内容",
        files=["./examples/sample.pdf"]
    )
    print(f"分析结果: {response}")


def example_image_analysis():
    """图像分析示例"""
    from tir_agent import TIRAgent
    
    agent = TIRAgent()
    
    # 分析图片
    response = agent.chat(
        "请详细描述这张图片的内容",
        files=["./examples/image.jpg"]
    )
    print(f"图像描述: {response}")


def example_code_execution():
    """代码执行示例"""
    from tir_agent import TIRAgent
    
    agent = TIRAgent()
    
    # 直接执行代码
    code = """
import numpy as np

# 创建数组
arr = np.array([1, 2, 3, 4, 5])

# 计算统计量
print(f"平均值: {np.mean(arr)}")
print(f"标准差: {np.std(arr)}")
print(f"总和: {np.sum(arr)}")
"""
    
    result = agent.execute_code(code)
    print(f"执行结果:\n{result}")


def example_streaming_chat():
    """流式对话示例"""
    from tir_agent import TIRAgent
    
    agent = TIRAgent()
    
    messages = [{"role": "user", "content": "请解释什么是机器学习"}]
    
    print("助手: ", end="")
    for response in agent.run(messages):
        if response:
            last_msg = response[-1] if response else {}
            content = last_msg.get("content", "")
            if content:
                print(content, end="", flush=True)
    print()


def example_vlm_direct():
    """直接使用VLM分析器示例（通过TIRAgent）"""
    from tir_agent import TIRAgent
    
    agent = TIRAgent()
    
    # 使用agent分析图片
    result = agent.analyze_image_directly("./examples/document.png")
    print(f"图像分析结果: {result}")


if __name__ == "__main__":
    print("TIR Agent 使用示例")
    print("=" * 50)
    
    # 运行基本对话示例
    print("\n1. 基本对话示例:")
    try:
        example_basic_chat()
    except Exception as e:
        print(f"示例运行失败: {e}")
    
    print("\n" + "=" * 50)
    print("更多示例请参考源代码")
