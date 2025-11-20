import streamlit as st
import os
import json
import pandas as pd
import time
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

st.title("🏭 高质量数据集蒸馏工厂")
st.markdown("利用强大的大模型（Teacher Model）生成用于微调（SFT）的高质量指令数据集。")

# --- 侧边栏：模型配置 ---
with st.sidebar:
    st.header("⚙️ 模型设置")
    
    provider = st.selectbox(
        "选择模型服务商",
        ["OpenAI", "Anthropic", "Google", "Pollinations", "Custom"]
    )

    api_key = ""
    base_url = None
    model_name = ""

    # 动态显示配置项，优先读取 .env
    if provider == "OpenAI":
        api_key = st.text_input("API Key", value=os.getenv("OPENAI_API_KEY", ""), type="password", placeholder="sk-xxxxxxxxxxxxxxxxxxxxxxxx")
        model_name = st.selectbox("选择模型", ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"])
    
    elif provider == "Anthropic":
        api_key = st.text_input("API Key", value=os.getenv("ANTHROPIC_API_KEY", ""), type="password", placeholder="sk-ant-xxxxxxxxxxxxxxxxxxxxxxxx")
        model_name = st.selectbox("选择模型", ["claude-3-5-sonnet-20240620", "claude-3-opus-20240229"])
        
    elif provider == "Google":
        api_key = st.text_input("API Key", value=os.getenv("GOOGLE_API_KEY", ""), type="password", placeholder="AIxxxxxxxxxxxxxxxxxxxxxxxx")
        model_name = st.selectbox("选择模型", ["gemini-1.5-pro", "gemini-1.5-flash"])
        
    elif provider == "Pollinations":
        st.info("🌸 Pollinations AI - 免费无需注册的AI生成平台")
        api_key = "pollinations"  # Pollinations AI不需要API密钥
        
        # 文本模型选择
        text_models = ["openai", "mistral", "mistral-large", "claude", "gemini", 
                      "llama", "llamaguard", "command", "searchgpt", "unity"]
        model_name = st.selectbox("选择文本模型", text_models, help="选择用于文本生成的模型")
        
        # 高级参数配置
        with st.expander("🔧 高级参数配置"):
            st.write("**文本生成参数**:")
            pollinations_seed = st.number_input("随机种子 (文本)", min_value=0, max_value=999999, value=0, help="0表示随机")
            pollinations_private = st.checkbox("私有模式", value=True, help="生成的内容不显示在公共流中")
        
    elif provider == "Custom":
        st.info("适用于 DeepSeek, Groq, Moonshot 或 本地 vLLM/Ollama")
        base_url = st.text_input("Base URL", value=os.getenv("CUSTOM_BASE_URL", ""), placeholder="https://api.example.com/v1")
        api_key = st.text_input("API Key", value=os.getenv("CUSTOM_API_KEY", ""), type="password", placeholder="sk-xxxxxxxxxxxxxxxxxxxxxxxx")
        model_name = st.text_input("Model Name", value="", placeholder="llama3-70b, deepseek-chat, gpt-4")

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
            client = LLMClient(provider, api_key, base_url, model_name)
        # 传递Pollinations AI的高级参数
        if provider == "Pollinations":
            client.seed = pollinations_seed if pollinations_seed > 0 else None
            client.private = pollinations_private
        
        with st.spinner(f"正在让 {model_name} 分析领域知识..."):
            system_prompt = "你是一位专家级数据架构师。请根据用户输入的领域，拆解出具体的细分任务场景。"
            user_prompt = f"""
            领域：{target_domain}
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
        if provider == "PollinationsAI":
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

# 区域 4: Pollinations AI 图像生成 (已移除)
# st.divider()
# st.subheader("4. 🎨 AI 图像生成 (Pollinations AI)")
# 
# # 初始化图像生成状态
# if "generated_images" not in st.session_state:
#     st.session_state.generated_images = []
#
# ... (完整图像生成功能代码已注释移除)
