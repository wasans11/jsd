import streamlit as st
import pickle
import numpy as np
from scipy import sparse
from scipy.spatial.distance import jensenshannon
import pandas as pd

# 로드
X = sparse.load_npz('X.npz')
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
df = pd.read_csv('games.csv')

st.title('🎮 스팀 게임 추천')

user_input = st.text_input('좋아하는 장르/태그 입력')

if user_input:
    user_vec = vectorizer.transform([user_input]).toarray()[0]
    user_prob = (user_vec + 1e-10) / (user_vec.sum() + 1e-10)
    
    jsd_scores = []
    for i in range(len(df)):
        game_vec = X[i].toarray()[0]
        game_prob = (game_vec + 1e-10) / (game_vec.sum() + 1e-10)
        jsd_scores.append(jensenshannon(user_prob, game_prob))
    
    jsd_scores = np.array(jsd_scores)
    top_idx = np.argsort(jsd_scores)[:10]
    rec = df.iloc[top_idx].copy()
    rec['유사도'] = (1 - jsd_scores[top_idx]).round(3)
    
    st.dataframe(rec)
