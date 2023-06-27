import csv
import pickle

import numpy as np
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt


st.markdown(

    """
<style>
[data-testid="stMetricLabel"] p {
    font-size: 25px;
    color: red;
}
</style>
""",
    unsafe_allow_html=True,


)

st.title("Titanic")
st.subheader("만약 내가 타이타닉호에 탔다면 나의 생존율은?")
st.image("./data/banner.png")

def lode_model():


    with open("./data/model.pkl", "rb") as fr:
        model = pickle.load(fr)

    return model

@st.cache_data
def lode_scores():
    scores = []
    with open("./data/scores.csv") as fr:
        reader = csv.reader(fr)
        for row in reader:
            scores.append(float(row[0]))

    return scores

def inference(sex, age, pclass):
    sex_label = 1 if sex == "Male" else 0
    input_data = [[pclass, sex_label, age]]
    survival_ratio = model.predict(input_data)[0]
    return survival_ratio


def get_distribution_plot(survival_ratio):
    # sns.set(rc={'axes.facecolor': '#FFFFFFFF', 'figure.facecolor': (0, 0, 0, 0)})
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot()
    sns.set_style("whitegrid")
    sns.kdeplot(
        scores,
        fill=True,
        color="#1263e3",
        alpha=0.5,
        linewidth=0,
        ax=ax,
    )
    plt.axvline(survival_ratio, color="coral", linewidth=3, label="YOU")
    plt.xticks(np.arange(0, 1.2, 0.2))
    plt.legend()
    return fig


data_load_state = st.text("모델 로딩중...")
model = lode_model()
data_load_state.text("모델 로딩... 완료!")
scores = lode_scores()

with st.form("my_form"):
    name = st.text_input("이름")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("나이", format="%d", min_value=0, max_value=100, step=1)
    with col2:

        sex = st.selectbox("성별", ["Male", "Female"])
    with col3:
        pclass = st.selectbox("좌석 등급", [1, 2, 3])

    submitted = st.form_submit_button("Submit")

    if submitted:

        if not name:
            st.error("이름을 입력해주세요!")
        else:

            survival_ratio = inference(sex,age,pclass)
            fig = get_distribution_plot(survival_ratio)

            st.metric(
                label = f"{name}님의 생존율",
                value=f"{round(survival_ratio * 100, 2)}%"

                )
                    #st.success("DONE!")

            with st.spinner('Plotting...'):
                st.pyplot(fig)
