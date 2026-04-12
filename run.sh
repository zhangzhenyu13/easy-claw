#!/bin/bash

# TIR Agent 启动脚本

set -e

# 检查Python版本
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "🐍 Python版本: $PYTHON_VERSION"

# 检查.env文件
if [ ! -f ".env" ]; then
    echo "⚠️  未找到.env文件，正在从.env.example创建..."
    cp .env.example .env
    echo "📝 请编辑.env文件配置您的API密钥"
fi

# 安装依赖
echo "📦 正在安装依赖..."
pip install -e .

# 启动应用
echo "🚀 启动TIR Agent..."
python main.py web "$@"
