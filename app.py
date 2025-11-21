import streamlit as st
import os
import json
import pandas as pd
import time
import requests
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
import google.generativeai as genai

# 1. 加载环境变量
load_dotenv()

# ================== 后端逻辑：多模型适配器 ==================

class PollinationsAIClient:
    """Pollinations AI 客户端 - 支持文本和图像生成"""
    def __init__(self, model_name="openai", model_type="text"):
        self.base_url = "https://text.pollinations.ai" if model_type == "text" else "https://image.pollinations.ai"
        self.model_name = model_name
        self.model_type = model_type
    
    def generate_text(self, prompt, system_prompt=None, **kwargs):
        """生成文本 - 使用Pollinations AI文本API"""
        import requests
        import urllib.parse
        
        # 构建完整的提示词
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n用户任务：{prompt}"
        
        # URL编码提示词
        encoded_prompt = urllib.parse.quote(full_prompt)
        
        # 构建请求URL
        url = f"{self.base_url}/prompt/{encoded_prompt}"
        
        # 添加参数
        params = {"model": self.model_name}
        if kwargs.get("seed"):
            params["seed"] = kwargs["seed"]
        if kwargs.get("private"):
            params["private"] = "true"
        
        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            return response.text
        except Exception as e:
            return f"Error: {str(e)}"
    
    def generate_image(self, prompt, **kwargs):
        """生成图像 - 使用Pollinations AI图像API"""
        import requests
        import urllib.parse
        
        # URL编码提示词
        encoded_prompt = urllib.parse.quote(prompt)
        
        # 构建请求URL
        url = f"{self.base_url}/prompt/{encoded_prompt}"
        
        # 添加参数
        params = {
            "model": kwargs.get("model", "flux"),
            "width": kwargs.get("width", 1024),
            "height": kwargs.get("height", 1024),
            "seed": kwargs.get("seed"),
            "nologo": "true" if kwargs.get("nologo", False) else "false",
            "private": "true" if kwargs.get("private", True) else "false",
            "enhance": "true" if kwargs.get("enhance", False) else "false"
        }
        
        # 移除None值
        params = {k: v for k, v in params.items() if v is not None}
        
        try:
            response = requests.get(url, params=params, timeout=300)
            response.raise_for_status()
            return response.content  # 返回图像二进制数据
        except Exception as e:
            return f"Error: {str(e)}"

class LLMClient:
    def __init__(self, provider, api_key, base_url=None, model_name=None):
        self.provider = provider
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name

    def generate(self, system_prompt, user_prompt):
        """统一的生成接口，屏蔽不同厂商 SDK 的差异"""
        try:
            if self.provider == "OpenAI":
                client = OpenAI(api_key=self.api_key, base_url=self.base_url)
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    response_format={"type": "json_object"} # 尝试强制JSON模式
                )
                return response.choices[0].message.content

            elif self.provider == "Custom":
                if not self.base_url or not self.model_name:
                    return f"Error: Custom provider requires both base_url and model_name to be configured"
                client = OpenAI(api_key=self.api_key, base_url=self.base_url)
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    response_format={"type": "json_object"} # 尝试强制JSON模式
                )
                return response.choices[0].message.content

            elif self.provider == "Anthropic":
                client = anthropic.Anthropic(api_key=self.api_key)
                message = client.messages.create(
                    model=self.model_name,
                    max_tokens=4096,
                    temperature=0.7,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}]
                )
                return message.content[0].text

            elif self.provider == "Google":
                genai.configure(api_key=self.api_key)
                model = genai.GenerativeModel(
                    self.model_name,
                    generation_config={"response_mime_type": "application/json"}
                )
                # Gemini system prompt 需要在实例化时配置或拼接到 user prompt，这里简化处理
                full_prompt = f"System Instruction:\n{system_prompt}\n\nUser Task:\n{user_prompt}"
                response = model.generate_content(full_prompt)
                return response.text

            elif self.provider == "Pollinations":
                # 使用Pollinations AI生成文本
                client = PollinationsAIClient(model_name=self.model_name, model_type="text")
                # 构建额外参数
                kwargs = {}
                if hasattr(self, 'seed') and self.seed:
                    kwargs['seed'] = self.seed
                if hasattr(self, 'private') and self.private:
                    kwargs['private'] = self.private
                return client.generate_text(user_prompt, system_prompt, **kwargs)

        except Exception as e:
            return f"Error: {str(e)}"

def clean_json_text(text):
    """清洗 JSON 字符串，移除 Markdown 标记"""
    import re
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*$', '', text)
    return text.strip()

# ================== 前端界面 (Streamlit) ==================

st.set_page_config(page_title="AI 数据集蒸馏工厂", layout="wide", page_icon="🏭")

# 自定义CSS来调整密码输入框样式
st.markdown("""
<style>
/* 密码输入框样式调整 */
.stTextInput > div {
    position: relative !important;
}

.stTextInput > div > div {
    position: relative !important;
}

.stTextInput input[type="password"] {
    right: 0px !important;
    position: relative !important;
}

/* 调整小眼睛按钮位置，继续往右移动 */
.stTextInput > div > div > button[title*="password"] {
    right: -12px !important;
    position: relative !important;
}

/* 确保与下拉选择框的箭头垂直对齐 */
.stSelectbox > div > div {
    position: relative;
}

/* 调整下拉箭头位置，与密码框小眼睛图标垂直对齐 */
.stSelectbox > div > div > div:last-child {
    right: 0px;
}
</style>
""", unsafe_allow_html=True)

# 获取OpenAI模型列表的函数
def get_openai_models():
    """从OpenAI API获取最新的模型列表"""
    try:
        # 使用OpenAI客户端获取模型列表
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
        models_response = client.models.list()
        
        # 筛选出聊天模型
        chat_models = []
        for model in models_response.data:
            model_id = model.id
            # 筛选出适合聊天的模型
            if any(keyword in model_id for keyword in ["gpt", "chat"]):
                chat_models.append({
                    'display': model_id,
                    'value': model_id,
                    'description': f"OpenAI {model_id}"
                })
        
        # 按名称排序
        chat_models.sort(key=lambda x: x['display'])
        return chat_models
        
    except Exception as e:
        st.warning(f"获取OpenAI在线模型列表失败: {str(e)}，使用本地备份数据")
        return get_local_openai_models()

# 本地备份的OpenAI模型数据
def get_local_openai_models():
    """本地备份的OpenAI模型数据"""
    return [
        {'display': 'gpt-4o', 'value': 'gpt-4o', 'description': 'OpenAI GPT-4o'},
        {'display': 'gpt-4o-mini', 'value': 'gpt-4o-mini', 'description': 'OpenAI GPT-4o Mini'},
        {'display': 'gpt-4-turbo', 'value': 'gpt-4-turbo', 'description': 'OpenAI GPT-4 Turbo'},
        {'display': 'gpt-3.5-turbo', 'value': 'gpt-3.5-turbo', 'description': 'OpenAI GPT-3.5 Turbo'}
    ]

# 获取Anthropic模型列表的函数
def get_anthropic_models():
    """从Anthropic API获取最新的模型列表"""
    try:
        # 使用Anthropic客户端获取模型列表
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
        models_response = client.models.list()
        
        # 筛选出适合对话的模型
        chat_models = []
        for model in models_response.data:
            model_id = model.id
            # 筛选出Claude系列模型
            if "claude" in model_id:
                chat_models.append({
                    'display': model_id,
                    'value': model_id,
                    'description': f"Anthropic {model_id}"
                })
        
        # 按名称排序
        chat_models.sort(key=lambda x: x['display'])
        return chat_models
        
    except Exception as e:
        st.warning(f"获取Anthropic在线模型列表失败: {str(e)}，使用本地备份数据")
        return get_local_anthropic_models()

# 本地备份的Anthropic模型数据
def get_local_anthropic_models():
    """本地备份的Anthropic模型数据"""
    return [
        {'display': 'claude-3-5-sonnet-20240620', 'value': 'claude-3-5-sonnet-20240620', 'description': 'Anthropic Claude 3.5 Sonnet'},
        {'display': 'claude-3-opus-20240229', 'value': 'claude-3-opus-20240229', 'description': 'Anthropic Claude 3 Opus'},
        {'display': 'claude-3-sonnet-20240229', 'value': 'claude-3-sonnet-20240229', 'description': 'Anthropic Claude 3 Sonnet'},
        {'display': 'claude-3-haiku-20240307', 'value': 'claude-3-haiku-20240307', 'description': 'Anthropic Claude 3 Haiku'}
    ]

# 获取Google Gemini模型列表的函数
def get_google_models():
    """获取Google Gemini模型列表 - Google API不提供模型列表接口，使用本地配置"""
    try:
        # Google Generative AI没有提供获取模型列表的API
        # 使用预定义的模型列表
        return [
            {'display': 'gemini-1.5-pro', 'value': 'gemini-1.5-pro', 'description': 'Google Gemini 1.5 Pro'},
            {'display': 'gemini-1.5-flash', 'value': 'gemini-1.5-flash', 'description': 'Google Gemini 1.5 Flash'},
            {'display': 'gemini-2.0-flash-exp', 'value': 'gemini-2.0-flash-exp', 'description': 'Google Gemini 2.0 Flash Experimental'},
            {'display': 'gemini-2.0-flash-thinking-exp-1219', 'value': 'gemini-2.0-flash-thinking-exp-1219', 'description': 'Google Gemini 2.0 Flash Thinking'}
        ]
    except Exception as e:
        st.warning(f"获取Google模型列表失败: {str(e)}，使用本地备份数据")
        return get_local_google_models()

# 本地备份的Google模型数据
def get_local_google_models():
    """本地备份的Google模型数据"""
    return [
        {'display': 'gemini-1.5-pro', 'value': 'gemini-1.5-pro', 'description': 'Google Gemini 1.5 Pro'},
        {'display': 'gemini-1.5-flash', 'value': 'gemini-1.5-flash', 'description': 'Google Gemini 1.5 Flash'},
        {'display': 'gemini-2.0-flash-exp', 'value': 'gemini-2.0-flash-exp', 'description': 'Google Gemini 2.0 Flash Experimental'},
        {'display': 'gemini-2.0-flash-thinking-exp-1219', 'value': 'gemini-2.0-flash-thinking-exp-1219', 'description': 'Google Gemini 2.0 Flash Thinking'}
    ]

# 获取Pollinations AI模型列表的函数
def get_pollinations_models():
    """从Pollinations AI API获取最新的模型列表"""
    try:
        response = requests.get("https://text.pollinations.ai/models", timeout=5)
        if response.status_code == 200:
            models_data = response.json()
            # 构建模型选择列表，显示完整描述，使用name作为实际值
            model_options = []
            for model in models_data:
                display_name = model['description']
                model_options.append({
                    'display': display_name,
                    'value': model['name'],
                    'description': model.get('description', model['name'])
                })
            return model_options
        else:
            st.warning("无法从在线API获取模型列表，使用本地备份数据")
            return get_local_pollinations_models()
    except Exception as e:
        st.warning(f"获取在线模型列表失败: {str(e)}，使用本地备份数据")
        return get_local_pollinations_models()

# 本地备份的模型数据
def get_local_pollinations_models():
    """本地备份的模型数据"""
    return [
        {'display': 'DeepSeek V3.1', 'value': 'deepseek', 'description': 'DeepSeek V3.1'},
        {'display': 'Gemini 2.5 Flash Lite', 'value': 'gemini', 'description': 'Gemini 2.5 Flash Lite'},
        {'display': 'Gemini 2.5 Flash Lite with Google Search', 'value': 'gemini-search', 'description': 'Gemini 2.5 Flash Lite with Google Search'},
        {'display': 'Mistral Small 3.2 24B', 'value': 'mistral', 'description': 'Mistral Small 3.2 24B'},
        {'display': 'OpenAI GPT', 'value': 'openai', 'description': 'OpenAI GPT'},
        {'display': 'Llama 3.2 3B', 'value': 'llama', 'description': 'Llama 3.2 3B'},
        {'display': 'LlamaGuard 7B', 'value': 'llamaguard', 'description': 'LlamaGuard 7B'},
        {'display': 'Cohere Command', 'value': 'command', 'description': 'Cohere Command'},
        {'display': 'Unity', 'value': 'unity', 'description': 'Unity'}
    ]

st.title("🏭 高质量数据集蒸馏工厂")
st.markdown("利用强大的大模型（Teacher Model）生成用于微调（SFT）的高质量指令数据集。")

# --- 侧边栏：模型配置 ---
with st.sidebar:
    st.header("⚙️ 模型设置")
    
    provider = st.selectbox(
        "选择模型服务商",
        ["Pollinations", "OpenAI", "Anthropic", "Google", "Custom"],
        index=0
    )

    api_key = ""
    base_url = None
    model_name = ""

    # 动态显示配置项，优先读取 .env
    if provider == "OpenAI":
        api_key = st.text_input("API Key", value=os.getenv("OPENAI_API_KEY", ""), type="password", placeholder="sk-xxxxxxxxxxxxxxxx...")
        
        # 初始化session state用于跟踪模型加载状态
        if "openai_loading" not in st.session_state:
            st.session_state.openai_loading = False
        if "openai_models" not in st.session_state:
            st.session_state.openai_models = []
        
        # 只有在切换到OpenAI且有API密钥时才触发自动获取模型列表
        if api_key and ("last_provider" not in st.session_state or st.session_state.last_provider != "OpenAI"):
            st.session_state.last_provider = "OpenAI"
            st.session_state.openai_loading = True
            
            # 使用spinner显示加载动画
            with st.spinner("🔍 正在获取OpenAI模型列表..."):
                try:
                    openai_models = get_openai_models()
                    st.session_state.openai_models = openai_models
                    st.session_state.openai_loading = False
                except Exception as e:
                    st.error(f"获取OpenAI模型列表失败: {str(e)}")
                    st.session_state.openai_models = []
                    st.session_state.openai_loading = False
        
        # 获取OpenAI模型列表
        if api_key:
            try:
                if st.session_state.openai_loading:
                    # 如果正在加载，显示基础模型选项
                    display_names = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
                    selected_display = st.selectbox("选择模型", display_names, help="选择用于文本生成的OpenAI模型")
                    model_name = selected_display
                elif st.session_state.openai_models:
                    # 使用已加载的模型列表
                    openai_models = st.session_state.openai_models
                    display_names = [model['display'] for model in openai_models]
                    selected_display = st.selectbox("选择模型", display_names, help="选择用于文本生成的OpenAI模型")
                    model_name = next(model['value'] for model in openai_models if model['display'] == selected_display)
                else:
                    # 实时获取模型列表
                    openai_models = get_openai_models()
                    display_names = [model['display'] for model in openai_models]
                    selected_display = st.selectbox("选择模型", display_names, help="选择用于文本生成的OpenAI模型")
                    model_name = next(model['value'] for model in openai_models if model['display'] == selected_display)
            except Exception as e:
                st.error(f"获取OpenAI模型列表失败: {str(e)}")
                model_name = st.selectbox("选择模型", ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"])
        else:
            model_name = st.selectbox("选择模型", ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"])
    
    elif provider == "Anthropic":
        api_key = st.text_input("API Key", value=os.getenv("ANTHROPIC_API_KEY", ""), type="password", placeholder="sk-ant-xxxxxxxxxxxxx...")
        
        # 初始化session state用于跟踪模型加载状态
        if "anthropic_loading" not in st.session_state:
            st.session_state.anthropic_loading = False
        if "anthropic_models" not in st.session_state:
            st.session_state.anthropic_models = []
        
        # 只有在切换到Anthropic且有API密钥时才触发自动获取模型列表
        if api_key and ("last_provider" not in st.session_state or st.session_state.last_provider != "Anthropic"):
            st.session_state.last_provider = "Anthropic"
            st.session_state.anthropic_loading = True
            
            # 使用spinner显示加载动画
            with st.spinner("🔍 正在获取Anthropic模型列表..."):
                try:
                    anthropic_models = get_anthropic_models()
                    st.session_state.anthropic_models = anthropic_models
                    st.session_state.anthropic_loading = False
                except Exception as e:
                    st.error(f"获取Anthropic模型列表失败: {str(e)}")
                    st.session_state.anthropic_models = []
                    st.session_state.anthropic_loading = False
        
        # 获取Anthropic模型列表
        if api_key:
            try:
                if st.session_state.anthropic_loading:
                    # 如果正在加载，显示基础模型选项
                    display_names = ["claude-3-5-sonnet-20240620", "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
                    selected_display = st.selectbox("选择模型", display_names, help="选择用于文本生成的Anthropic模型")
                    model_name = selected_display
                elif st.session_state.anthropic_models:
                    # 使用已加载的模型列表
                    anthropic_models = st.session_state.anthropic_models
                    display_names = [model['display'] for model in anthropic_models]
                    selected_display = st.selectbox("选择模型", display_names, help="选择用于文本生成的Anthropic模型")
                    model_name = next(model['value'] for model in anthropic_models if model['display'] == selected_display)
                else:
                    # 实时获取模型列表
                    anthropic_models = get_anthropic_models()
                    display_names = [model['display'] for model in anthropic_models]
                    selected_display = st.selectbox("选择模型", display_names, help="选择用于文本生成的Anthropic模型")
                    model_name = next(model['value'] for model in anthropic_models if model['display'] == selected_display)
            except Exception as e:
                st.error(f"获取Anthropic模型列表失败: {str(e)}")
                model_name = st.selectbox("选择模型", ["claude-3-5-sonnet-20240620", "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"])
        else:
            model_name = st.selectbox("选择模型", ["claude-3-5-sonnet-20240620", "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"])
        
    elif provider == "Google":
        api_key = st.text_input("API Key", value=os.getenv("GOOGLE_API_KEY", ""), type="password", placeholder="AIxxxxxxxxxxxxxxxx...")
        
        # 初始化session state用于跟踪模型加载状态
        if "google_loading" not in st.session_state:
            st.session_state.google_loading = False
        if "google_models" not in st.session_state:
            st.session_state.google_models = []
        
        # 只有在切换到Google时才触发自动获取模型列表
        if "last_provider" not in st.session_state or st.session_state.last_provider != "Google":
            st.session_state.last_provider = "Google"
            st.session_state.google_loading = True
            
            # 使用spinner显示加载动画
            with st.spinner("🔍 正在获取Google模型列表..."):
                try:
                    google_models = get_google_models()
                    st.session_state.google_models = google_models
                    st.session_state.google_loading = False
                except Exception as e:
                    st.error(f"获取Google模型列表失败: {str(e)}")
                    st.session_state.google_models = []
                    st.session_state.google_loading = False
        
        # 获取Google模型列表
        if st.session_state.google_loading:
            # 如果正在加载，显示基础模型选项
            google_models = [
                {'display': 'Gemini 2.5 Pro', 'value': 'gemini-2.5-pro', 'description': 'Gemini 2.5 Pro'},
                {'display': 'Gemini 2.5 Flash', 'value': 'gemini-2.5-flash', 'description': 'Gemini 2.5 Flash'},
                {'display': 'Gemini 2.0 Flash', 'value': 'gemini-2.0-flash', 'description': 'Gemini 2.0 Flash'},
                {'display': 'Gemini 1.5 Pro', 'value': 'gemini-1.5-pro', 'description': 'Gemini 1.5 Pro'}
            ]
        elif st.session_state.google_models:
            # 使用已加载的模型列表
            google_models = st.session_state.google_models
        else:
            # 默认获取模型列表
            google_models = get_google_models()
        
        # 提取显示名称用于选择框
        display_names = [model['display'] for model in google_models]
        selected_display = st.selectbox("选择模型", display_names, help="选择用于文本生成的Google Gemini模型")
        
        # 根据选择的显示名称找到对应的实际模型值
        model_name = next(model['value'] for model in google_models if model['display'] == selected_display)
        
    elif provider == "Pollinations":
        st.info("🌸 Pollinations AI - 免费无需注册的AI生成平台")
        api_key = "pollinations"  # Pollinations AI不需要API密钥
        
        # 初始化session state用于跟踪模型加载状态
        if "pollinations_loading" not in st.session_state:
            st.session_state.pollinations_loading = False
        if "pollinations_models" not in st.session_state:
            st.session_state.pollinations_models = []
        
        # 只有在切换到Pollinations时才触发自动获取模型列表
        if "last_provider" not in st.session_state or st.session_state.last_provider != "Pollinations":
            st.session_state.last_provider = "Pollinations"
            st.session_state.pollinations_loading = True
            
            # 使用spinner显示加载动画
            with st.spinner("🌸 正在获取Pollinations模型列表..."):
                try:
                    model_options = get_pollinations_models()
                    st.session_state.pollinations_models = model_options
                    st.session_state.pollinations_loading = False
                except Exception as e:
                    st.error(f"获取模型列表失败: {str(e)}")
                    st.session_state.pollinations_models = get_local_pollinations_models()
                    st.session_state.pollinations_loading = False
        
        # 如果正在加载，显示加载状态
        if st.session_state.pollinations_loading:
            st.info("🔄 正在获取模型列表...")
            # 使用本地备份数据作为临时选项
            model_options = get_local_pollinations_models()
        elif st.session_state.pollinations_models:
            # 使用已加载的模型列表
            model_options = st.session_state.pollinations_models
        else:
            # 默认获取模型列表
            model_options = get_pollinations_models()
        
        # 提取显示名称和实际值用于选择框
        display_names = [model['display'] for model in model_options]
        selected_display = st.selectbox("选择文本模型", display_names, help="选择用于文本生成的模型")
        
        # 根据选择的显示名称找到对应的实际模型值
        selected_model = next(model['value'] for model in model_options if model['display'] == selected_display)
        model_name = selected_model
        
        # 高级参数配置
        with st.expander("🔧 高级参数配置"):
            st.write("**文本生成参数**:")
            pollinations_seed = st.number_input("随机种子 (文本)", min_value=0, max_value=999999, value=0, help="0表示随机")
            pollinations_private = st.checkbox("私有模式", value=True, help="生成的内容不显示在公共流中")
        
    elif provider == "Custom":
        st.info("完全自定义模型服务商配置")
        
        # 服务商名称输入
        custom_provider_name = st.text_input("服务商名称", value=os.getenv("CUSTOM_PROVIDER_NAME", ""), placeholder="如：DeepSeek、Groq、Moonshot、Ollama等")
        
        # 基础配置
        base_url = st.text_input("Base URL", value=os.getenv("CUSTOM_BASE_URL", ""), placeholder="https://api.example.com/v1")
        api_key = st.text_input("API Key", value=os.getenv("CUSTOM_API_KEY", ""), type="password", placeholder="sk-xxxxxxxxxxxxxxxx...")
        
        # 完全自定义模型名称
        model_name = st.text_input("模型名称", value=os.getenv("CUSTOM_MODEL_NAME", ""), placeholder="输入完整的模型名称，如：deepseek-chat、llama3-70b、gpt-3.5-turbo等")

    if not api_key and provider != "Pollinations":
        st.warning("⚠️ 请在 .env 文件中配置密钥或在上方输入")

# --- 主界面逻辑 ---

# 初始化 Session State (用于保存生成过程中的数据)
if "topics" not in st.session_state:
    st.session_state.topics = []
if "generated_data" not in st.session_state:
    st.session_state.generated_data = []

# 区域 1: 领域定义
st.subheader("1. 定义目标领域与任务")
col1, col2 = st.columns([3, 1])
with col1:
    target_domain = st.text_input("请输入你想复刻的领域能力", placeholder="例如：Python安全代码审计、医疗问诊对话、初中数学几何推理")
with col2:
    num_topics = st.number_input("生成主题数量", min_value=1, max_value=50, value=5)

if st.button("🚀 生成任务分类树 (Taxonomy)"):
    if not api_key and provider != "Pollinations":
        st.error("请先配置 API Key")
    else:
        client = LLMClient(provider, api_key, base_url, model_name)
        # 传递Pollinations AI的高级参数
        if provider == "Pollinations":
            client.seed = pollinations_seed if pollinations_seed > 0 else None
            client.private = pollinations_private
        
        with st.spinner(f"正在让 {model_name} 分析领域知识..."):
            system_prompt = "你是一位专家级数据架构师。请根据用户输入的领域，拆解出具体的细分任务场景。"
            user_prompt = f"""
            领域: {target_domain}
            请生成 {num_topics} 个具体的、高难度的细分任务。
            要求：输出严格的 JSON 格式，包含 'topics' 列表。
            
            JSON 示例：
            {{ "topics": ["任务A", "任务B", "任务C"] }}
            """
            
            raw_resp = client.generate(system_prompt, user_prompt)
            
            try:
                cleaned_resp = clean_json_text(raw_resp)
                data = json.loads(cleaned_resp)
                st.session_state.topics = data.get("topics", [])
                st.success(f"成功生成 {len(st.session_state.topics)} 个任务主题！")
            except Exception as e:
                st.error(f"解析失败: {e}")
                st.text(raw_resp)

# 显示已生成的主题
if st.session_state.topics:
    st.info(f"当前待生成主题：{', '.join(st.session_state.topics)}")

    st.divider()

    # 区域 2: 数据生成
    st.subheader("2. 批量生产高质量数据")
    
    samples_per_topic = st.slider("每个主题生成的数据量", 1, 20, 3)
    
    if st.button("🔥 开始蒸馏数据"):
        client = LLMClient(provider, api_key, base_url, model_name)
        # 传递Pollinations AI的高级参数
        if provider == "Pollinations":
            client.seed = pollinations_seed if pollinations_seed > 0 else None
            client.private = pollinations_private
        
        st.session_state.generated_data = [] # 清空旧数据
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_steps = len(st.session_state.topics)
        
        for i, topic in enumerate(st.session_state.topics):
            status_text.text(f"正在生成主题 ({i+1}/{total_steps}): {topic} ...")
            
            system_prompt = """
            你是一个用于构建高质量指令微调数据集的AI。
            严格要求：
            1. Output必须包含 "Thought" (思维链) 和 "Answer"。
            2. 格式必须是合法的 JSON 列表。
            """
            
            user_prompt = f"""
            主题：{topic}
            请生成 {samples_per_topic} 条复杂的指令微调数据。
            
            输出格式示例：
            {{
                "samples": [
                    {{
                        "instruction": "用户指令...",
                        "input": "上下文（可选）...",
                        "output": "Thought: ... Answer: ..."
                    }}
                ]
            }}
            """
            
            raw_resp = client.generate(system_prompt, user_prompt)
            
            try:
                cleaned_resp = clean_json_text(raw_resp)
                batch_data = json.loads(cleaned_resp).get("samples", [])
                
                for item in batch_data:
                    item['category'] = topic # 添加元数据
                    st.session_state.generated_data.append(item)
                    
            except Exception as e:
                st.warning(f"主题 {topic} 生成失败，跳过。")
            
            # 更新进度
            progress_bar.progress((i + 1) / total_steps)
            time.sleep(0.5) # 避免速率限制

        status_text.text("✅ 所有数据生成完毕！")
        
    # 区域 3: 结果展示与下载
    if st.session_state.generated_data:
        st.subheader("3. 数据集预览与导出")
        
        df = pd.DataFrame(st.session_state.generated_data)
        st.dataframe(df, use_container_width=True)
        
        # 转换为 JSONL 格式供下载
        jsonl_data = df.to_json(orient="records", lines=True, force_ascii=False)
        
        st.download_button(
            label="💾 下载 JSONL 格式数据集 (可直接用于训练)",
            data=jsonl_data,
            file_name=f"dataset_{target_domain}.jsonl",
            mime="application/json"
        )
        
        # CSV 下载选项
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="💾 下载 CSV 格式 (Excel查看)",
            data=csv_data,
            file_name=f"dataset_{target_domain}.csv",
            mime="text/csv"
        )

# 应用结束
