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

st.title('🎮 정보이론 기반(JSD)스팀 게임 추천 웹앱')

# 사이드바
with st.sidebar:
    st.markdown("""
      ## 📊 데이터 전처리 과정
### 1. 컬럼 정리 (113,000개 → 82,000개 게임)
- **제거**: 식별자(URL, 이미지), 플랫폼(Windows/Mac/Linux), 인기지표(추천수, 플레이타임), 개발사/퍼블리셔
- **점수 변환**: Metacritic/User score → has_meta (0/1 이진값)
- **결측 처리**: Tags 없는 게임 27% 제거 (정보 밀도 확보)
### 2. 텍스트 전처리
- **통합**: 게임설명 + 장르 + 카테고리 + 태그 → 단일 텍스트
- **정제**: 대소문자 통일, 특수문자/숫자 제거, 영어만 추출
- **벡터화**: TF-IDF (최대 150개 핵심 단어, 상위 1% 초고빈도 단어 제거)
### 3. 최종 데이터
- 82,129개 게임 × 150차원 벡터
- 각 게임: 이름, 가격, 메타점수 유무
---
## 🎯 JSD 추천 시스템
### 작동 원리
1. **입력**: 사용자가 선호하는 장르/태그 입력
2. **벡터화**: 입력을 150차원 확률분포로 변환
3. **유사도 계산**: Jensen-Shannon Divergence로 모든 게임과 비교
4. **추천**: 가장 낮은 JSD 값 상위 10개 게임 출력
### JSD의 장점
- **확률 기반**: 단순 키워드 매칭이 아닌 의미론적 유사도 측정
- **대칭성**: 거리 개념으로 해석 가능 (0=동일, 1=완전 다름)
- **노이즈 강건**: KL Divergence와 달리 무한대 발산 없음
- **세밀함**: 150개 단어 분포로 장르/분위기/스타일까지 구분
    """)

# 키워드 expander
keywords = list(vectorizer.get_feature_names_out())
with st.expander("💡 사용 가능한 키워드 목록 (총 150개)"):
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

