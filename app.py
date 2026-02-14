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
- **제거**: 식별자(URL, 이미지), 플랫폼(Windows/Mac/Linux), 인기지표(추천수, 플레이타임), 개발사/퍼블리셔
- **점수 변환**: Metacritic/User score → has_meta (0/1 이진값)
- **결측 처리**: Tags 없는 게임 27% 제거 (정보 밀도 확보)

### 2. 텍스트 전처리
- **통합**: 게임설명 + 장르 + 카테고리 + 태그 → 단일 텍스트
- **정제**: 대소문자 통일, 특수문자/숫자 제거, 영어만 추출
- **벡터화**: TF-IDF (최대 200개 핵심 단어, 상위 1% 초고빈도 단어 제거)

### 3. 최종 데이터
- 82,129개 게임 × 200차원 벡터
- 각 게임: 이름, 가격, 메타점수 유무

---

## 🎯 JSD 추천 시스템

### 작동 원리
1. **입력**: 사용자가 선호하는 장르/태그 입력
2. **벡터화**: 입력을 200차원 확률분포로 변환
3. **유사도 계산**: Jensen-Shannon Divergence로 모든 게임과 비교
4. **추천**: 가장 낮은 JSD 값 상위 10개 게임 출력

### JSD의 장점
- **확률 기반**: 단순 키워드 매칭이 아닌 의미론적 유사도 측정
- **대칭성**: 거리 개념으로 해석 가능 (0=동일, 1=완전 다름)
- **노이즈 강건**: KL Divergence와 달리 무한대 발산 없음
- **세밀함**: 200개 단어 분포로 장르/분위기/스타일까지 구분
    """)
    
    # 사용 가능한 키워드 보기
    if st.checkbox('사용 가능한 키워드 보기'):
        keywords = list(vectorizer.get_feature_names_out())
        st.write(f"총 {len(keywords)}개")
        st.text_area('키워드 목록', ', '.join(keywords), height=300)

user_input = st.text_input('장르/태그/키워드 입력 (예: action rpg multiplayer)')

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
