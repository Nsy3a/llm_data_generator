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

class LLMClient:
    def __init__(self, provider, api_key, base_url=None, model_name=None):
        self.provider = provider
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name

    def generate(self, system_prompt, user_prompt):
        """统一的生成接口，屏蔽不同厂商 SDK 的差异"""
        try:
            if self.provider == "OpenAI" or self.provider == "Custom (OpenAI-Compatible)":
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

            elif self.provider == "Anthropic (Claude)":
                client = anthropic.Anthropic(api_key=self.api_key)
                message = client.messages.create(
                    model=self.model_name,
                    max_tokens=4096,
                    temperature=0.7,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}]
                )
                return message.content[0].text

            elif self.provider == "Google (Gemini)":
                genai.configure(api_key=self.api_key)
                model = genai.GenerativeModel(
                    self.model_name,
                    generation_config={"response_mime_type": "application/json"}
                )
                # Gemini system prompt 需要在实例化时配置或拼接到 user prompt，这里简化处理
                full_prompt = f"System Instruction:\n{system_prompt}\n\nUser Task:\n{user_prompt}"
                response = model.generate_content(full_prompt)
                return response.text

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

# 添加自定义CSS样式
st.markdown("""
<style>
/* 优化侧边栏样式 */
.css-1d391kg {
    padding: 1rem 1.5rem;
}

/* 优化标题样式 */
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    color: #1f4788;
    font-weight: 600;
}

/* 优化容器样式 */
.stContainer {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    background-color: #f8f9fa;
}

/* 优化按钮样式 */
.stButton > button {
    border-radius: 8px;
    font-weight: 500;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

/* 优化选择框样式 */
.stSelectbox > div > div {
    border-radius: 8px;
}

/* 优化输入框样式 */
.stTextInput > div > div > input {
    border-radius: 8px;
    border: 1px solid #ccc;
}

/* 优化警告和成功消息 */
.stAlert {
    border-radius: 8px;
    padding: 1rem;
}
</style>
""", unsafe_allow_html=True)

st.title("🏭 高质量数据集蒸馏工厂")
st.markdown("利用强大的大模型（Teacher Model）生成用于微调（SFT）的高质量指令数据集。")

# --- 侧边栏：模型配置 ---
with st.sidebar:
    st.header("⚙️ 模型设置")
    
    # 添加分隔线和说明
    st.markdown("---")
    st.markdown("🎯 **选择AI模型服务商**")
    
    provider = st.selectbox(
        "服务商",
        ["OpenAI", "Anthropic (Claude)", "Google (Gemini)", "Custom (OpenAI-Compatible)"],
        help="选择您要使用的AI模型服务商"
    )

    api_key = ""
    base_url = None
    model_name = ""

    # 动态显示配置项，优先读取 .env
    if provider == "OpenAI":
        with st.container():
            st.markdown("**🔑 认证配置**")
            api_key = st.text_input("API Key", value=os.getenv("OPENAI_API_KEY", ""), type="password", help="您的OpenAI API密钥")
            
            st.markdown("**🤖 模型配置**")
            model_name = st.selectbox("选择模型", ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"], help="选择OpenAI模型")
    
    elif provider == "Anthropic (Claude)":
        with st.container():
            st.markdown("**🔑 认证配置**")
            api_key = st.text_input("API Key", value=os.getenv("ANTHROPIC_API_KEY", ""), type="password", help="您的Anthropic API密钥")
            
            st.markdown("**🤖 模型配置**")
            model_name = st.selectbox("选择模型", ["claude-3-5-sonnet-20240620", "claude-3-opus-20240229"], help="选择Claude模型")
        
    elif provider == "Google (Gemini)":
        with st.container():
            st.markdown("**🔑 认证配置**")
            api_key = st.text_input("API Key", value=os.getenv("GOOGLE_API_KEY", ""), type="password", help="您的Google API密钥")
            
            st.markdown("**🤖 模型配置**")
            model_name = st.selectbox("选择模型", ["gemini-1.5-pro", "gemini-1.5-flash"], help="选择Gemini模型")
        
    elif provider == "Custom (OpenAI-Compatible)":
        st.info("💡 适用于 DeepSeek, Groq, Moonshot 或 本地 vLLM/Ollama")
        
        # 移除容器包装，避免悬停时出现多余容器
        # 使用容器来组织自定义配置
        st.markdown("**🔧 服务器配置**")
        
        # 统一的URL输入框（自动解析端口）
        base_url = st.text_input(
            "Base URL", 
            value=os.getenv("CUSTOM_BASE_URL", "https://api.openai.com/v1"), 
            help="完整的API服务器地址，如：https://api.deepseek.com/v1 或 http://localhost:8000/v1"
        )
        
        # URL解析和验证提示
        if base_url:
            import re
            url_pattern = r'^(https?://)([^:/]+)(:(\d+))?(/.*)?$'
            match = re.match(url_pattern, base_url)
            
            if match:
                protocol, domain, _, port, path = match.groups()
                port_display = f":{port}" if port else ""
                st.info(f"📍 解析结果: {protocol}{domain}{port_display}{path or ''}")
            else:
                st.warning("⚠️ URL格式不正确，请检查输入")
        
        st.markdown("**🔑 认证配置**")
        
        # 第二行：API密钥（带显隐切换）
        col_key, col_toggle = st.columns([4, 1])
        with col_key:
            if 'show_custom_key' not in st.session_state:
                st.session_state.show_custom_key = False
            
            key_type = "text" if st.session_state.show_custom_key else "password"
            api_key = st.text_input(
                "API Key", 
                value=os.getenv("CUSTOM_API_KEY", "sk-xxxx"), 
                type=key_type,
                help="您的API访问密钥"
            )
        
        with col_toggle:
            st.write("")  # 添加空行对齐
            if st.button("👁️" if not st.session_state.show_custom_key else "🙈", 
                       help="显示/隐藏密钥",
                       key="toggle_custom_key"):
                st.session_state.show_custom_key = not st.session_state.show_custom_key
                st.rerun()
        
        st.markdown("**🤖 模型配置**")
        
        # 模型选择 - 分离布局
        # 第一行：模型选择下拉菜单
        preset_models = [
            "llama3-70b", "llama3-8b", "deepseek-chat", "deepseek-coder", 
            "mixtral-8x7b", "qwen-14b", "gpt-3.5-turbo", "gpt-4"
        ]
        selected_model = st.selectbox(
            "选择模型", 
            preset_models + ["自定义"],
            help="选择预设模型或自定义输入"
        )
        
        # 第二行：模型显示或自定义输入
        if selected_model == "自定义":
            model_name = st.text_input(
                "自定义模型名称", 
                value="custom-model",
                help="输入您的自定义模型名称"
            )
        else:
            # 显示当前选中的模型（只读）
            st.text_input(
                "当前模型", 
                value=selected_model, 
                disabled=True,
                help="当前选中的模型"
            )
            model_name = selected_model
    
    # 快速配置提示
    with st.expander("📚 快速配置指南"):
            st.markdown("""
            **常用配置示例：**
            - **DeepSeek**: `https://api.deepseek.com/v1` + `deepseek-chat`
            - **Groq**: `https://api.groq.com/openai/v1` + `mixtral-8x7b`
            - **Moonshot**: `https://api.moonshot.cn/v1` + `moonshot-v1-8k`
            - **本地vLLM**: `http://localhost:8000/v1` + 自定义模型名
            - **本地Ollama**: `http://localhost:11434/v1` + 自定义模型名
            """)

    # 添加分隔线和状态提示
    st.markdown("---")
    
    if not api_key:
        st.warning("⚠️ 请在 .env 文件中配置密钥或在上方输入", icon="🔑")
    else:
        st.success(f"✅ {provider} 配置就绪", icon="✨")

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
    if not api_key:
        st.error("请先配置 API Key")
    else:
        client = LLMClient(provider, api_key, base_url, model_name)
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
