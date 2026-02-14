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

# 사이드바
with st.sidebar:
    st.markdown("""
    ## 📊 데이터 전처리 과정
    ### 1. 컬럼 정리 (113,000개 → 82,000개 게임)
    - **제거**: 식별자, 플랫폼, 인기지표, 개발사/퍼블리셔
    - **점수 변환**: has_meta (0/1 이진값)
    - **결측 처리**: Tags 없는 게임 27% 제거
    
    ### 2. 텍스트 전처리
    - **통합**: 게임설명 + 장르 + 카테고리 + 태그
    - **정제**: 대소문자 통일, 특수문자/숫자 제거
    - **벡터화**: TF-IDF 200차원
    
    ## 🎯 JSD 추천 시스템
    - 확률 기반 의미론적 유사도
    - 장르/분위기/스타일 구분
    """)

# 키워드 expander
keywords = list(vectorizer.get_feature_names_out())
with st.expander("💡 사용 가능한 키워드 목록 (총 200개)"):
    # 10개씩 나눠서 컬럼으로 보기 좋게
    cols = st.columns(4)
    chunk_size = len(keywords) // 4
    for i, col in enumerate(cols):
        start = i * chunk_size
        end = start + chunk_size if i < 3 else len(keywords)
        col.write(keywords[start:end])

st.info("💬 **입력 방법**: 키워드를 띄어쓰기로 구분해서 입력하세요 (쉼표 사용 X)")
user_input = st.text_input('장르/태그/키워드 입력 (예: action rpg multiplayer dark fantasy)')

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
