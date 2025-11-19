#!/usr/bin/env python3
"""
AI数据集蒸馏工厂 - 一键启动脚本
自动激活虚拟环境并启动Streamlit应用
创建时间: 2024年12月19日
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def get_venv_path():
    """获取虚拟环境路径"""
    current_dir = Path(__file__).parent
    venv_path = current_dir / "venv"
    
    if not venv_path.exists():
        print("❌ 虚拟环境不存在，请先创建虚拟环境！")
        print("执行: python -m venv venv")
        return None
    
    return venv_path

def get_activate_script_path(venv_path):
    """获取激活脚本路径（跨平台）"""
    system = platform.system()
    
    if system == "Windows":
        return venv_path / "Scripts" / "activate.bat"
    else:
        return venv_path / "bin" / "activate"

def check_streamlit_installed():
    """检查streamlit是否已安装"""
    try:
        import streamlit
        return True
    except ImportError:
        return False

def install_dependencies():
    """安装依赖库"""
    print("📦 正在安装依赖库...")
    try:
        # 使用国内源安装
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt",
            "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"
        ], check=True)
        print("✅ 依赖库安装完成")
        return True
    except subprocess.CalledProcessError:
        print("❌ 依赖库安装失败，请检查网络连接")
        return False

def activate_venv_and_run():
    """激活虚拟环境并运行Streamlit应用"""
    venv_path = get_venv_path()
    if not venv_path:
        return False
    
    # 检查依赖
    if not check_streamlit_installed():
        if not install_dependencies():
            return False
    
    print(f"🚀 正在启动AI数据集蒸馏工厂...")
    print(f"📁 项目路径: {os.path.dirname(os.path.abspath(__file__))}")
    print(f"🐍 Python版本: {sys.version}")
    print(f"🌐 虚拟环境: {venv_path}")
    print("=" * 50)
    
    try:
        # 直接运行streamlit命令，使用配置文件
        env = os.environ.copy()
        env["STREAMLIT_SERVER_HEADLESS"] = "true"  # 禁用浏览器自动打开
        
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py"
        ], env=env)
        return True
        
    except KeyboardInterrupt:
        print("\n👋 应用已停止")
        return True
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("🏭 AI数据集蒸馏工厂 - 一键启动工具")
    print("=" * 60)
    
    # 检查Python版本
    if sys.version_info < (3, 7):
        print("❌ Python版本过低，需要Python 3.7或更高版本")
        return False
    
    # 检查app.py是否存在
    if not os.path.exists("app.py"):
        print("❌ 未找到app.py文件，请确保在正确的目录中")
        return False
    
    # 显示自定义配置选项
    print("\n🛠️  自定义配置选项:")
    print("如果需要自定义模型配置，请在启动后选择 'Custom (OpenAI-Compatible)' 选项")
    print("支持的自定义服务商:")
    print("  • DeepSeek: https://api.deepseek.com/v1")
    print("  • Groq: https://api.groq.com/openai/v1") 
    print("  • Moonshot: https://api.moonshot.cn/v1")
    print("  • 本地vLLM: http://localhost:8000/v1")
    print("  • 本地Ollama: http://localhost:11434/v1")
    print("=" * 60)
    
    # 激活虚拟环境并运行
    return activate_venv_and_run()

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)