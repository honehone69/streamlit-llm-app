import streamlit as st
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain

# Webアプリの概要や操作方法をユーザーに明示するためのテキスト
st.title("# 専門家LLM応答アプリ") # st.title を Markdown ヘッダーで大きく表示
st.write("") # 空行

st.write("このアプリでは、テキスト入力と専門家の種類を選択することで、 विभिन्न専門家としてLLMに回答させることができます。")
st.write("画面上部のテキストボックスに質問を入力し、中央のラジオボタンで専門家の種類を選択してください。")
st.write("") # 空行

# OpenAI APIキーの設定 (Streamlit secretsから取得)
openai_api_key = st.secrets["OPENAI_API_KEY"]

# LLMの準備 (gpt-3.5-turbo モデルを使用)
llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo")

# 専門家の種類とシステムメッセージの定義
expert_options = {
    "旅行ガイド": "あなたは経験豊富な旅行ガイドです。旅行に関する質問に、親切かつ丁寧に答えてください。旅行先の情報、おすすめの観光スポット、旅行の計画の立て方など、旅行者が知りたいと思う情報を提供してください。",
    "シェフ": "あなたは一流のシェフです。料理に関する質問に、専門的な知識と情熱をもって答えてください。レシピ、調理方法、食材の選び方、料理のコツなど、料理をする人が役立つ情報を教えてください。",
    "親切なアシスタント": "あなたは親切なアシスタントです。質問に丁寧に答えてください。" # デフォルトの専門家
}

# LLMからの回答を戻り値として返す関数を定義
def get_llm_response(input_text, expert_type):
    system_message_template = SystemMessagePromptTemplate.from_template(expert_options[expert_type])
    human_message_template = HumanMessagePromptTemplate.from_template("{user_input}")
    chat_prompt = ChatPromptTemplate.from_messages([system_message_template, human_message_template])

    chain = LLMChain(llm=llm, prompt=chat_prompt)
    response = chain.run(user_input=input_text)
    return response

# インプットフォーム
st.header("質問入力") # st.header でセクション見出しを追加
input_text = st.text_area("質問を入力してください", "")
st.write("") # 空行

# ラジオボタンで専門家の種類を選択
st.header("専門家の種類選択") # st.header でセクション見出しを追加
expert_type = st.radio(
    "専門家の種類を選択してください",
    list(expert_options.keys()),
    index=0 # デフォルトを「旅行ガイド」にする場合は 0, 「親切なアシスタント」にする場合は 2 など
)
st.write("") # 空行

# 回答を表示する領域
response_area = st.empty()

# 入力テキストと専門家の種類が変更されたらLLMに問い合わせて回答を表示
if input_text:
    with st.spinner("LLMが回答を生成中..."):
        llm_response = get_llm_response(input_text, expert_type)
        response_area.markdown("### 回答:") # 回答部分を Markdown ヘッダーで強調
        response_area.write(llm_response)