import pandas as pd
from matplotlib import pyplot as plt
import streamlit as st
import numpy as np
import seaborn as sns


st.title("uber Pickup")

st.text("Uber pickup in NYC data analytics project.")
st.image("./data/banner.png")

DATE_COLUMN = "date/time"


@st.cache_data
def lord_data():
    df = pd.read_csv("./data/uber.csv")
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    return df


data_load_state = st.text("Loding data... ")
df = lord_data()
data_load_state.text("Loding data... Done! ")


if st.checkbox("Show raw data"):
    st.subheader("Raw Data")
    st.write(df.head())


# 주요 지표 나타내기 위한 col 사용
st.subheader("Major metrics")
col1, col2, col3 ,col4= st.columns(4)
col1.metric("Temperature", "70", "1.2")
col2.metric("Wind", "9mph", "-8%")
col3.metric("Humidity", "86%", "4%")
col4.metric("dd", "111%", "11%")

st.subheader("Pickup Histogrem")
bins = np.arange(0, 25, 1)
plt.figure(figsize=(10, 5))
plt.hist(
    df[DATE_COLUMN].dt.hour,
    bins=bins,
    label="count",
    width=0.8,
)
plt.legend()
plt.xlim(0, 24)
plt.xticks(bins, fontsize=8)
st.pyplot(plt)

st.subheader("Pickup Histogrem using seaborn")

ax = sns.histplot(

        df[DATE_COLUMN].dt.hour,
        bins=bins,
        kde=True,
)

st.pyplot(ax.figure)


st.subheader("ickup Histogrem using sttreamlit bar_chart")

hist_values = np.histogram(df[DATE_COLUMN].dt.hour, bins = 24, range=(0, 24))[0]
st.bar_chart(hist_values)













