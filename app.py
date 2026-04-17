import streamlit as st
import pickle
import numpy as np
from scipy import sparse
from scipy.spatial.distance import jensenshannon
import pandas as pd
import plotly.graph_objects as go

# ──────────────────────────────────────────
# 페이지 설정
# ──────────────────────────────────────────
st.set_page_config(
    page_title="GameMatch · JSD 추천",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────
# 커스텀 CSS
# ──────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=Inter:wght@300;400;500&display=swap');

/* ── 전체 배경 ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0a0a0f;
    color: #e8e8f0;
}
[data-testid="stAppViewContainer"] > .main {
    background: #0a0a0f;
}
[data-testid="stHeader"] { background: transparent; }
section[data-testid="stSidebar"] { background: #10101a; }

/* ── 폰트 기본 ── */
html, body, * { font-family: 'Inter', sans-serif; }

/* ── 히어로 타이틀 ── */
.hero {
    text-align: center;
    padding: 3rem 0 2rem;
}
.hero h1 {
    font-family: 'Rajdhani', sans-serif;
    font-size: 3.4rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    background: linear-gradient(135deg, #66c0f4 0%, #c6e4ff 50%, #4a9fd4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    line-height: 1.1;
}
.hero p {
    color: #6b7280;
    font-size: 0.95rem;
    margin-top: 0.5rem;
    font-weight: 300;
    letter-spacing: 0.05em;
}

/* ── 구분선 ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #1e3a5f, #66c0f4, #1e3a5f, transparent);
    margin: 1.5rem 0;
}

/* ── 섹션 라벨 ── */
.section-label {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #66c0f4;
    margin-bottom: 0.5rem;
}

/* ── 퀵태그 버튼 ── */
.stButton > button {
    background: #0f1923 !important;
    color: #8ab4cc !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 4px !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.1em !important;
    padding: 0.3rem 0.8rem !important;
    transition: all 0.15s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: #1e3a5f !important;
    color: #66c0f4 !important;
    border-color: #66c0f4 !important;
    box-shadow: 0 0 12px rgba(102,192,244,0.2) !important;
}

/* ── 추천 버튼 ── */
.recommend-btn > button {
    background: linear-gradient(135deg, #1b5276, #2980b9) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em !important;
    padding: 0.6rem 2rem !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 0 20px rgba(41,128,185,0.3) !important;
}
.recommend-btn > button:hover {
    background: linear-gradient(135deg, #2471a3, #3498db) !important;
    box-shadow: 0 0 30px rgba(102,192,244,0.5) !important;
    transform: translateY(-1px) !important;
}

/* ── multiselect ── */
[data-testid="stMultiSelect"] div[data-baseweb="select"] > div {
    background: #0f1923 !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 6px !important;
    color: #e8e8f0 !important;
}
[data-testid="stMultiSelect"] span[data-baseweb="tag"] {
    background-color: #1b3a5c !important;
    border: 1px solid #2980b9 !important;
    color: #66c0f4 !important;
    border-radius: 3px !important;
}

/* ── slider ── */
[data-testid="stSlider"] div[data-baseweb="slider"] div[role="slider"] {
    background: #66c0f4 !important;
}

/* ── selectbox ── */
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background: #0f1923 !important;
    border: 1px solid #1e3a5f !important;
    color: #e8e8f0 !important;
}

/* ── 게임 카드 ── */
.game-card {
    background: linear-gradient(145deg, #0f1923 0%, #0d1520 100%);
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
.game-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, #66c0f4, #2980b9);
}
.game-card:hover {
    border-color: #2980b9;
    box-shadow: 0 4px 20px rgba(102,192,244,0.12);
}
.card-rank {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    color: #4a7fa5;
    text-transform: uppercase;
}
.card-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: #e8f4fd;
    margin: 0.15rem 0 0.6rem;
    letter-spacing: 0.03em;
}
.card-meta {
    display: flex;
    gap: 1rem;
    font-size: 0.82rem;
    color: #6b8fa8;
    align-items: center;
}
.meta-badge {
    background: #0a1520;
    border: 1px solid #1e3a5f;
    border-radius: 3px;
    padding: 0.1rem 0.5rem;
    font-family: 'Rajdhani', sans-serif;
    font-weight: 600;
    letter-spacing: 0.05em;
}
.sim-bar-bg {
    background: #0a1520;
    border-radius: 2px;
    height: 4px;
    margin-top: 0.8rem;
    overflow: hidden;
}
.sim-bar-fill {
    height: 4px;
    border-radius: 2px;
    background: linear-gradient(90deg, #2980b9, #66c0f4);
}
.sim-label {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.75rem;
    color: #4a7fa5;
    letter-spacing: 0.1em;
    margin-top: 0.3rem;
}

/* ── expander ── */
[data-testid="stExpander"] {
    background: #0a0f18 !important;
    border: 1px solid #1a2a3a !important;
    border-radius: 6px !important;
}
[data-testid="stExpander"] summary {
    color: #4a7fa5 !important;
    font-size: 0.8rem !important;
}

/* ── info/warning 박스 ── */
[data-testid="stAlert"] {
    background: #0a1520 !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 6px !important;
    color: #8ab4cc !important;
}

/* label 색상 */
label, .stSelectbox label, .stSlider label, .stMultiSelect label {
    color: #8ab4cc !important;
    font-size: 0.85rem !important;
    font-weight: 400 !important;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────
# 데이터 로드 (캐시)
# ──────────────────────────────────────────
@st.cache_resource
def load_data():
    X = sparse.load_npz('X.npz')
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    df = pd.read_csv('games.csv')
    # 확률 행렬 미리 계산
    X_dense = X.toarray().astype(np.float32)
    X_dense += 1e-10
    X_prob = X_dense / X_dense.sum(axis=1, keepdims=True)
    return vectorizer, df, X_prob

vectorizer, df, X_prob = load_data()

# ──────────────────────────────────────────
# session_state 초기화
# ──────────────────────────────────────────
if "selected_tags" not in st.session_state:
    st.session_state.selected_tags = []

# ──────────────────────────────────────────
# JSD 계산 (캐시)
# ──────────────────────────────────────────
@st.cache_data
def compute_jsd(user_input: str):
    user_vec = vectorizer.transform([user_input]).toarray()[0].astype(np.float32)
    user_vec += 1e-10
    user_prob = user_vec / user_vec.sum()
    scores = np.array([jensenshannon(user_prob, row) for row in X_prob], dtype=np.float32)
    return scores

# ──────────────────────────────────────────
# 히어로
# ──────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>GAMEMATCH</h1>
  <p>Jensen-Shannon Divergence 기반 스팀 게임 추천 · 82,000+ 타이틀</p>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────
# 메인 레이아웃
# ──────────────────────────────────────────
left, right = st.columns([2, 3], gap="large")

with left:
    # 키워드 선택
    st.markdown('<div class="section-label">🔍 키워드 선택</div>', unsafe_allow_html=True)
    keywords = list(vectorizer.get_feature_names_out())
    selected = st.multiselect(
        label="키워드",
        options=keywords,
        default=st.session_state.selected_tags,
        placeholder="키워드를 검색하거나 선택하세요",
        label_visibility="collapsed",
    )
    st.session_state.selected_tags = selected

    # 퀵 태그
    st.markdown('<div class="section-label" style="margin-top:1rem">⚡ 빠른 선택</div>', unsafe_allow_html=True)
    quick_tags = ["action", "rpg", "horror", "strategy", "multiplayer",
                  "indie", "puzzle", "simulation", "sports", "adventure"]
    cols = st.columns(5)
    for i, tag in enumerate(quick_tags):
        with cols[i % 5]:
            if st.button(tag, key=f"quick_{tag}"):
                if tag not in st.session_state.selected_tags:
                    st.session_state.selected_tags.append(tag)
                    st.rerun()

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # 필터
    st.markdown('<div class="section-label">⚙️ 필터</div>', unsafe_allow_html=True)
    price_range = st.slider("가격 범위 ($)", 0, 100, (0, 60), step=5)
    meta_filter = st.selectbox("메타크리틱 점수", ["전체", "있음", "없음"])

    st.markdown("<br>", unsafe_allow_html=True)

    # 추천 버튼
    st.markdown('<div class="recommend-btn">', unsafe_allow_html=True)
    run = st.button("🎮  추천 받기", key="run_btn")
    st.markdown('</div>', unsafe_allow_html=True)

    # 키워드 목록 expander
    with st.expander("💡 전체 키워드 목록 (150개)"):
        chunk = len(keywords) // 3
        c1, c2, c3 = st.columns(3)
        c1.write(keywords[:chunk])
        c2.write(keywords[chunk:chunk*2])
        c3.write(keywords[chunk*2:])

# ──────────────────────────────────────────
# 결과 패널
# ──────────────────────────────────────────
with right:
    if run and selected:
        user_input = " ".join(selected)
        with st.spinner("유사도 계산 중..."):
            jsd_scores = compute_jsd(user_input)

        top_idx = np.argsort(jsd_scores)[:50]
        rec = df.iloc[top_idx].copy()
        rec["_jsd"] = jsd_scores[top_idx]
        rec["유사도"] = (1 - rec["_jsd"]).clip(0, 1)

        # 필터
        if "price" in rec.columns:
            rec = rec[(rec["price"] >= price_range[0]) & (rec["price"] <= price_range[1])]
        if meta_filter == "있음":
            rec = rec[rec["has_meta"] == 1]
        elif meta_filter == "없음":
            rec = rec[rec["has_meta"] == 0]

        rec = rec.head(10).reset_index(drop=True)

        if rec.empty:
            st.info("조건에 맞는 게임이 없어요. 필터를 완화해보세요.")
        else:
            # 유사도 차트
            st.markdown('<div class="section-label">📊 유사도 분포</div>', unsafe_allow_html=True)
            fig = go.Figure(go.Bar(
                x=rec["name"].str[:20],
                y=rec["유사도"],
                marker=dict(
                    color=rec["유사도"],
                    colorscale=[[0, "#1b3a5c"], [0.5, "#2980b9"], [1, "#66c0f4"]],
                    line=dict(color="#0a0f18", width=0.5),
                ),
                text=rec["유사도"].map(lambda x: f"{x:.3f}"),
                textposition="outside",
                textfont=dict(color="#8ab4cc", size=11, family="Rajdhani"),
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#8ab4cc", family="Rajdhani"),
                xaxis=dict(
                    tickfont=dict(size=11, color="#6b8fa8"),
                    gridcolor="#0f1923",
                    tickangle=-30,
                ),
                yaxis=dict(
                    range=[0, 1],
                    gridcolor="#0f1923",
                    tickfont=dict(size=11, color="#6b8fa8"),
                ),
                margin=dict(l=0, r=0, t=10, b=0),
                height=200,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-label">🏆 추천 결과</div>', unsafe_allow_html=True)

            # 카드
            col_a, col_b = st.columns(2)
            for i, row in rec.iterrows():
                target_col = col_a if i % 2 == 0 else col_b
                with target_col:
                    sim_pct = int(row["유사도"] * 100)
                    sim_width = f"{sim_pct}%"
                    price_val = row.get("price", "N/A")
                    price_str = f"${price_val:.2f}" if isinstance(price_val, (int, float)) else str(price_val)
                    meta_str = "✔ Metacritic" if row.get("has_meta", 0) == 1 else "—"

                    st.markdown(f"""
                    <div class="game-card">
                        <div class="card-rank">#{i+1} MATCH</div>
                        <div class="card-title">{row['name']}</div>
                        <div class="card-meta">
                            <span class="meta-badge">{price_str}</span>
                            <span>{meta_str}</span>
                        </div>
                        <div class="sim-bar-bg">
                            <div class="sim-bar-fill" style="width:{sim_width}"></div>
                        </div>
                        <div class="sim-label">유사도 {sim_pct}%</div>
                    </div>
                    """, unsafe_allow_html=True)

    elif run and not selected:
        st.info("키워드를 하나 이상 선택해주세요.")
    else:
        st.markdown("""
        <div style="
            height: 400px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            color: #1e3a5f;
            border: 1px dashed #1a2a3a;
            border-radius: 8px;
        ">
            <div style="font-size:3rem">🎮</div>
            <div style="font-family:'Rajdhani',sans-serif; font-size:1.1rem; letter-spacing:0.15em; margin-top:1rem; color:#2a4a6a">
                키워드를 선택하고 추천 받기를 누르세요
            </div>
        </div>
        """, unsafe_allow_html=True)

# ──────────────────────────────────────────
# 푸터
# ──────────────────────────────────────────
st.markdown('<div class="divider" style="margin-top:3rem"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#2a3a4a; font-size:0.75rem; letter-spacing:0.1em; padding-bottom:1rem; font-family:'Rajdhani',sans-serif">
    GAMEMATCH · JSD 기반 추천 · 82,129 TITLES
</div>
""", unsafe_allow_html=True)

