import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from supabase import create_client


# Supabase setup
@st.cache_resource
def init_connection():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)


supabase_client = init_connection()


@st.cache_data
def load_data():
    df = pd.read_csv("./data/movies_preprocessed.csv")
    df["all_tokens"] = df["all_tokens"].apply(lambda x: eval(x))
    df['all_tokens'] = df['all_tokens'].apply(lambda tokens: ' '.join(tokens))

    # Vectorize the 'all_tokens' column
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['all_tokens'])
    tfidf_csr_matrix = tfidf_matrix.tocsr()
    return df, tfidf_vectorizer, tfidf_csr_matrix



df, tfidf_vectorizer, tfidf_csr_matrix = load_data()

def add_to_supabase(user_ask, results):
    data = {"user_ask": user_ask, "results": results}
    supabase_client.table('movie_recommend').insert(data).execute()


def tfidf_search(query, k=5):
    query_csr_matrix = tfidf_vectorizer.transform([query])
    similarities = cosine_similarity(query_csr_matrix, tfidf_csr_matrix).flatten()
    top_similarities = sorted(similarities, reverse=True)[:k]
    top_indices = similarities.argsort()[-k:][::-1]
    top_titles = [df.iloc[i]['title'] for i in top_indices]
    results = [(round(top_similarity, 4), top_title) for top_title, top_similarity in zip(top_titles, top_similarities)]

    return results





# Streamlit code

image_url = "https://i.ibb.co/f15Tf0w/image.png"
st.image(image_url, caption='', use_column_width=True)

st.markdown("""<h1 style='text-align: center;'>TF-IDF를 이용한 영화 추천 프로그램 🍿</h1>""", unsafe_allow_html=True)


user_input = st.text_input("관심 있는 영화 제목이나 특성, 장르, 배우 이름을 영어로 입력하세요.",
                          placeholder="꼭!!!!!!!!!!!!!!!!!!! 영어로 입력 하세요!!!!!!!!!!")

if st.button('검색'):
    results = tfidf_search(user_input)
    add_to_supabase(str(user_input), str(results))
    for score, title in results:
        st.write(f'{title} 의 유사도 점수 {score}')
