import numpy as np
from matplotlib import pyplot as plt
import streamlit as st
import pandas as pd
import seaborn as sns

st.title("Uber Pickup")

st.text("Uber pickup data analytics project 3.")
st.image("./data/banner.png")

DATE_COLUMN = "date/time"


# @st.cache_data
def load_data():
    df = pd.read_csv("./data/uber.csv")
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    return df

# 데이터 로딩하기
data_load_state = st.text("Loading data...")
df = load_data()
data_load_state.text("Loading data...done!")

if st.checkbox("Show raw data"):
    st.subheader("Raw Data")
    st.write(df.head())

st.subheader("Major metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Temperature", "70", "1.2")
col2.metric("Wind", "9mhp", "-8%")
col3.metric("Humidity", "86%", "-40")
col4.metric("Sales", "45K", "12%")

# matplotlib 이용해서 histogram 그리고 출력하기
st.subheader("Pickup Histogram")
bins = np.arange(0, 25, 1)
plt.figure(figsize=(10, 6))
plt.hist(
    df[DATE_COLUMN].dt.hour,
    bins=bins,
    label="count",
    width=0.8
)
plt.xlim(0, 24)
plt.xticks(bins, fontsize=8)
plt.legend()
st.pyplot(plt)

# seaborn 이용해서 histogram 그리고 출력하기
st.subheader("Pickup Histogram using seaborn")
ax = sns.histplot(
    df[DATE_COLUMN].dt.hour,
    bins=bins,
    kde=True
)
st.pyplot(ax.figure)

# streamlit 내장 bar_chart 함수 이용해서 histogram 그리고 출력하기
st.subheader("Pickup Histogram using streamtlit bar_chart")
hist_values = np.histogram(df[DATE_COLUMN].dt.hour, bins=24, range=(0, 24))[0]
st.bar_chart(hist_values)











