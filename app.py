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
    /* Formatting the table for better readability */
    .stTable { background-color: #ffffff; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    # Ensure this filename matches your GitHub upload exactly
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
    c1_svc = st.selectbox("Service Line:", options=sorted(df['service_name_group'].unique()), key="logic_svc")

df_c1 = df[(df['cbsa_name'].isin(c1_msa)) & (df['service_name_group'] == c1_svc)]

if not df_c1.empty:
    fig1 = px.scatter(
        df_c1, x="lot_size", y="cost", color="cbsa_name", trendline="ols",
        hover_data=["input_address"],
        labels={"lot_size": "Lot Size (sq ft)", "cost": "Quote Amount"},
        template="plotly_white", height=500
    )
    # FORMAT: Currency on Y-Axis
    fig1.update_layout(yaxis_tickprefix='$', yaxis_tickformat=',.2f')
    st.plotly_chart(fig1, use_container_width=True)
else:
    st.warning("No data for this selection.")

st.divider()

# --- SECTION 2: MARKET BENCHMARKING ---
st.header("2. Regional Market Benchmarking")
c2_col1, c2_col2 = st.columns([1, 3])

with c2_col1:
    st.write("### Controls")
    c2_msa = st.multiselect("Compare Markets:", options=sorted(df['cbsa_name'].unique()), default=df['cbsa_name'].unique(), key="bench_msa")
    c2_svc = st.selectbox("Service Line:", options=sorted(df['service_name_group'].unique()), key="bench_svc")

df_c2 = df[(df['cbsa_name'].isin(c2_msa)) & (df['service_name_group'] == c2_svc)]

if not df_c2.empty:
    avg_price = df_c2.groupby('cbsa_name')['cost'].mean().reset_index().sort_values('cost', ascending=False)
    
    # FIXED: This text_auto format specifically fixes the long decimals in your screenshot
    fig2 = px.bar(
        avg_price, x='cost', y='cbsa_name', orientation='h', 
        color='cost', color_continuous_scale='Greens',
        text_auto='$.,2f', 
        template="plotly_white", height=450
    )
    # FORMAT: Currency on X-Axis
    fig2.update_layout(showlegend=False, xaxis_tickprefix='$', xaxis_tickformat=',.0f')
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.warning("No data for this selection.")

st.divider()

# --- SECTION 3: THE PREDICTOR ---
st.header("3. Unit Economics Predictor")
st.caption("Compare the predicted 'Rate Card' across all available markets simultaneously.")

p1, p2 = st.columns(2)
with p1: pred_svc = st.selectbox("Select Service to Quote:", options=sorted(df['service_name_group'].unique()), key="p_svc")
with p2: test_size = st.number_input("Input Property Size (sq ft):", min_value=0, value=5000, step=500)

pred_subset = df[df['service_name_group'] == pred_svc]
res = []
for msa in sorted(df['cbsa_name'].unique()):
    m_data = pred_subset[pred_subset['cbsa_name'] == msa]
    if len(m_data) > 2:
        z = np.polyfit(m_data['lot_size'], m_data['cost'], 1)
        p_price = max(0, z[1] + (z[0] * test_size))
        res.append({
            "Market": msa, 
            "Predicted Quote": p_price, 
            "Base Fee (Fixed)": max(0, z[1]), 
            "Rate (Variable per sqft)": z[0]
        })

if res:
    predict_df = pd.DataFrame(res).sort_values("Predicted Quote", ascending=False)
    # Applying currency formatting to the table display
    disp_df = predict_df.copy()
    disp_df['Predicted Quote'] = disp_df['Predicted Quote'].map('${:,.2f}'.format)
    disp_df['Base Fee (Fixed)'] = disp_df['Base Fee (Fixed)'].map('${:,.2f}'.format)
    disp_df['Rate (Variable per sqft)'] = disp_df['Rate (Variable per sqft)'].map('${:,.4f}'.format)
    st.table(disp_df)

# --- FOOTER: DATA DOWNLOADS ---
st.divider()
st.subheader("📂 Data Access")
dl_col1, dl_col2 = st.columns(2)

with dl_col1:
    st.download_button(
        label="Download Predictions (CSV)",
        data=predict_df.to_csv(index=False).encode('utf-8'),
        file_name=f"weedman_predictions_{pred_svc}.csv",
        mime="text/csv"
    )

with dl_col2:
    st.download_button(
        label="Download Full Scraped Data (CSV)",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name="weedman_full_dataset.csv",
        mime="text/csv"
    )

st.caption("CONFIDENTIAL: For Strategic Review Only")
