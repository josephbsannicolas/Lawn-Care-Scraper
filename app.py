import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# 1. PROFESSIONAL THEME & CONFIG
st.set_page_config(page_title="Market Intel: Weedman Pricing Analysis", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for a "Premium" look
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 10px; border: 1px solid #e9ecef; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    div[data-testid="stExpander"] { border: none !important; box-shadow: none !important; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("weedman_sample_quotes_clean.csv")
    df['lot_size'] = df['lot_size'].astype(int)
    df['scrape_timestamp'] = pd.to_datetime(df['scrape_timestamp'])
    return df

df = load_data()

# --- SIDEBAR: METHODOLOGY ---
with st.sidebar:
    st.title("Project Intel")
    st.info("""
    **Objective:** Reverse-engineer Weedman’s dynamic pricing engine across key MSAs.
    
    **Methodology:** * Automated lead generation via Python/Playwright.
    * Geo-distributed address sampling.
    * Statistical OLS modeling to isolate fixed vs. variable cost drivers.
    """)
    st.caption(f"Last Intelligence Update: {df['scrape_timestamp'].max().strftime('%b %d, %Y')}")

# --- HEADER: EXECUTIVE SUMMARY ---
st.title("📊 Competitive Intelligence: Weedman Pricing Strategy")

m1, m2, m3 = st.columns(3)
m1.metric("Total Quotes Captured", f"{len(df):,}")
m2.metric("Market Footprint", f"{df['cbsa_name'].nunique()} MSAs")
m3.metric("Data Recency", df['scrape_timestamp'].max().strftime('%Y-%m-%d'))

with st.expander("📝 Strategic Key Takeaways", expanded=True):
    st.markdown("""
    * **Regional Multipliers:** Significant pricing variance observed between MSAs for identical lot sizes.
    * **Fixed Cost Floor:** Analysis reveals a consistent 'Base Trip Fee' (Y-Intercept) indicating a minimum service threshold.
    * **Linear Scalability:** Pricing follows a high-confidence linear model, suggesting a centralized quoting algorithm.
    """)

st.markdown("---")

# --- SECTION 1: PRICING LOGIC ---
st.header("1. Pricing Logic & Scalability")
c1_col1, c1_col2 = st.columns([1, 3])

with c1_col1:
    st.write("### Controls")
    c1_msa = st.multiselect("Focus Markets:", options=sorted(df['cbsa_name'].unique()), default=df['cbsa_name'].unique()[0], key="logic_msa")
    c1_svc = st.selectbox("Select Service Line:", options=sorted(df['service_name_group'].unique()), key="logic_svc")

df_c1 = df[(df['cbsa_name'].isin(c1_msa)) & (df['service_name_group'] == c1_svc)]

if not df_c1.empty:
    fig1 = px.scatter(
        df_c1, x="lot_size", y="cost", color="cbsa_name", trendline="ols",
        hover_data=["input_address"],
        labels={"lot_size": "Lot Size (sq ft)", "cost": "Quote Amount"},
        template="plotly_white", height=500
    )
    # FORMAT: Add $ to Y-axis
    fig1.update_layout(yaxis_tickprefix='$', yaxis_tickformat=',.2f')
    st.plotly_chart(fig1, use_container_width=True)

st.divider()

# --- SECTION 2: MARKET BENCHMARKING ---
st.header("2. Regional Market Benchmarking")
c2_col1, c2_col2 = st.columns([1, 3])

with c2_col1:
    st.write("### Controls")
    c2_msa = st.multiselect("Compare Markets:", options=sorted(df['cbsa_name'].unique()), default=df['cbsa_name'].unique(), key="bench_msa")
    c2_svc = st.selectbox("Service Line:", options=sorted(df['service_name_group'].unique()),
