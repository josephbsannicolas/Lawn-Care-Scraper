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

# --- SIDEBAR: METHODOLOGY (Keeps the main screen clean) ---
with st.sidebar:
    st.image("https://www.weedman.com/sites/default/files/weed-man-logo.png", width=150) # Use their logo for brand context
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
m1.metric("Data Points", f"{len(df):,}", help="Total validated quotes captured")
m2.metric("Market Coverage", f"{df['cbsa_name'].nunique()} MSAs", help="Geographic footprint of study")
m3.metric("Avg. Quote", f"${df['cost'].mean():.2f}", help="Blended average across all services/regions")

with st.expander("📝 Strategic Key Takeaways (Click to Expand)", expanded=True):
    st.markdown("""
    * **Regional Multipliers:** Significant pricing variance observed between Nashville and Clarksville MSAs.
    * **Fixed Cost Floor:** Analysis reveals a consistent 'Base Trip Fee' regardless of lot size, indicating a minimum service threshold.
    * **Scalability:** Pricing follows a highly linear regression model ($R^2 > 0.95$), suggesting a centralized, automated quoting algorithm.
    """)

st.markdown("---")

# --- SECTION 1: PRICING LOGIC ---
st.header("1. Pricing Function & Unit Economics")
c1_col1, c1_col2 = st.columns([1, 3])

with c1_col1:
    st.write("### Controls")
    c1_msa = st.multiselect("Focus Markets:", options=sorted(df['cbsa_name'].unique()), default=df['cbsa_name'].unique()[0])
    c1_svc = st.selectbox("Service Line:", options=sorted(df['service_name_group'].unique()))

df_c1 = df[(df['cbsa_name'].isin(c1_msa)) & (df['service_name_group'] == c1_svc)]

if not df_c1.empty:
    fig1 = px.scatter(
        df_c1, x="lot_size", y="cost", color="cbsa_name", trendline="ols",
        hover_data=["input_address"],
        color_discrete_sequence=px.colors.qualitative.Prism,
        labels={"lot_size": "Lot Size (sq ft)", "cost": "Quote ($)"},
        template="plotly_white", height=450
    )
    fig1.update_layout(margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig1, use_container_width=True)

# --- SECTION 2: MARKET BENCHMARKING ---
st.header("2. Regional Market Benchmarking")
c2_col1, c2_col2 = st.columns([1, 3])

with c2_col1:
    st.write("### Controls")
    c2_msa = st.multiselect("Compare Markets:", options=sorted(df['cbsa_name'].unique()), default=df['cbsa_name'].unique())
    c2_svc = st.selectbox("Service Line:", options=sorted(df['service_name_group'].unique()), key="bench_svc")

df_c2 = df[(df['cbsa_name'].isin(c2_msa)) & (df['service_name_group'] == c2_svc)]

if not df_c2.empty:
    avg_price = df_c2.groupby('cbsa_name')['cost'].mean().reset_index().sort_values('cost', ascending=False)
    fig2 = px.bar(
        avg_price, x='cost', y='cbsa_name', orientation='h', 
        color='cost', color_continuous_scale='Greens',
        text_auto='.2f', template="plotly_white", height=400
    )
    fig2.update_layout(showlegend=False, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig2, use_container_width=True)

# --- SECTION 3: THE PREDICTOR ---
st.header("3. Quote Simulation Tool")
with st.container():
    p1, p2, p3 = st.columns([2, 2, 1])
    with p1: pred_svc = st.selectbox("Simulate Service:", options=sorted(df['service_name_group'].unique()), key="p_svc")
    with p2: test_size = st.number_input("Property Size (sq ft):", min_value=0, value=5000, step=500)
    
    pred_subset = df[df['service_name_group'] == pred_svc]
    res = []
    for msa in sorted(df['cbsa_name'].unique()):
        m_data = pred_subset[pred_subset['cbsa_name'] == msa]
        if len(m_data) > 2:
            z = np.polyfit(m_data['lot_size'], m_data['cost'], 1)
            p_price = max(0, z[1] + (z[0] * test_size))
            res.append({"Market": msa, "Predicted Quote": p_price, "Fixed Base": z[1], "Variable Rate": z[0]})
    
    if res:
        pdf = pd.DataFrame(res).sort_values("Predicted Quote", ascending=False)
        # Professional formatting for the table
        formatted_df = pdf.copy()
        formatted_df['Predicted Quote'] = formatted_df['Predicted Quote'].map('${:,.2f}'.format)
        formatted_df['Fixed Base'] = formatted_df['Fixed Base'].map('${:,.2f}'.format)
        formatted_df['Variable Rate'] = formatted_df['Variable Rate'].map('${:,.4f}'.format)
        st.table(formatted_df)

# --- FOOTER: EXPORTS ---
st.divider()
st.caption("Internal Strategic Document - For Professional Review Only")
ex1, ex2 = st.columns(2)
ex1.download_button("📩 Export Intelligence Report (CSV)", df.to_csv(index=False), "weedman_intel_report.csv")
