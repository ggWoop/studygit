import openai
import streamlit as st

from supabase import create_client


@st.cache_resource
def init_connection():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)


supabase_client = init_connection()

st.markdown(
    """
<style>
footer {
    visibility: hidden;
}
</style>
""",
    unsafe_allow_html=True,
)


openai.api_key = st.secrets.OPENAI_TOKEN
openai_model_version = "gpt-3.5-turbo-0613"

st.title("AI Copywriter✍️")
st.subheader("AI를 이용하여 손쉽게 마케팅 문구를 생성해보세요.")
st.text(f"Powerd by {openai_model_version}")


def generate_prompt(name, description, keywords, n=3):
    prompt = f""" 
특정 제품 혹은 브랜드를 광고하기 위한 문구를 {n}개 생성해주세요.
제품의 특징이 드러나야 합니다.
키워드가 주어질 경우, 반드시 키워드 중 하나를 포함해야 합니다.
간결하게 한 문장으로 작성해주세요.
---
제품/브랜드 이름: {name}
제품 간단 정보: {description}
키워드: {keywords}
---
"""
    return prompt.strip()


def request_chat_completion(prompt):
    response = openai.ChatCompletion.create(
        model=openai_model_version,
        messages=[
            {"role": "system", "content": "당신은 전문 카피라이터입니다."},
            {"role": "user", "content": prompt}
        ]
    )
    return response["choices"][0]["message"]["content"]


def write_prompt_result(prompt, result):
    response = supabase_client.table("prompt_results").insert(
        {
            "prompt": prompt,
            "result": result
        }
    ).execute()
    print(response)
with st.form("form"):
    name = st.text_input("제품/브랜드 이름(필수)")
    desc = st.text_input("제품 간단 정보(필수)")
    st.text("포함할 키워드(최대 3개까지 허용)")
    col1, col2, col3 = st.columns(3)
    with col1:
        keyword_one = st.text_input(placeholder="키워드 1", label="keyword_1", label_visibility="collapsed")
    with col2:
        keyword_two = st.text_input(placeholder="키워드 2", label="keyword_2", label_visibility="collapsed")
    with col3:
        keyword_three = st.text_input(placeholder="키워드 3", label="keyword_3", label_visibility="collapsed")

    submitted = st.form_submit_button("Submit")
    if submitted:
        if not name:
            st.error("브랜드 혹은 제품의 이름을 입력해주세요")
        elif not desc:
            st.error("제품의 간단한 정보를 입력해주세요")
        else:
            with st.spinner('AI 카피라이터가 광고 문구를 생성 중입니다...'):
                keywords = [keyword_one, keyword_two, keyword_three]
                keywords = [x for x in keywords if x]
                prompt = generate_prompt(name, desc, keywords)
                response = request_chat_completion(prompt)
                write_prompt_result(prompt, response)
                st.text_area(
                    label="마케팅 문구 생성 결과",
                    value=response,
                    placeholder="아직 생성된 문구가 없습니다.",
                    height=200
                )