import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime

# Page Config
st.set_page_config(page_title="Weedman Pricing Engine Analysis", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("weedman_sample_quotes_clean.csv")
    df['lot_size'] = df['lot_size'].astype(int)
    # Convert timestamp to datetime for the summary card
    df['scrape_timestamp'] = pd.to_datetime(df['scrape_timestamp'])
    return df

df = load_data()

# --- MAIN TITLE & SUMMARY CARDS ---
st.title("🎯 Weedman Pricing Engine: Reverse Engineered")

# DATA SUMMARY ROW
total_quotes = len(df)
unique_msas = df['cbsa_name'].nunique()
latest_scrape = df['scrape_timestamp'].max().strftime('%Y-%m-%d')
earliest_scrape = df['scrape_timestamp'].min().strftime('%Y-%m-%d')

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Quotes Scraped", f"{total_quotes:,}")
m2.metric("Markets Analyzed", unique_msas)
m3.metric("Data Recency", latest_scrape)
m4.metric("Avg. Price (All)", f"${df['cost'].mean():.2f}")

st.markdown("---")

# --- SECTION 1: INDEPENDENT CHARTS ---
col1, col2 = st.columns(2)

# --- LEFT COLUMN: Pricing Logic ---
with col1:
    st.subheader("📊 Pricing Logic & Scalability")
    st.caption("Analyze the correlation between property size and quoted cost.")
    
    c1_msa = st.multiselect("Select Markets:", options=sorted(df['cbsa_name'].unique()), default=df['cbsa_name'].unique()[0], key="c1_msa")
    c1_svc = st.selectbox("Select Service:", options=sorted(df['service_name_group'].unique()), key="c1_svc")
    
    df_c1 = df[(df['cbsa_name'].isin(c1_msa)) & (df['service_name_group'] == c1_svc)]
    
    if not df_c1.empty:
        fig1 = px.scatter(
            df_c1, x="lot_size", y="cost", color="cbsa_name", trendline="ols",
            hover_data=["input_address"],
            labels={"lot_size": "Lot Size (sq ft)", "cost": "Quote Amount ($)"},
            template="plotly_white"
        )
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.warning("No data for this selection.")

# --- RIGHT COLUMN: Benchmarking ---
with col2:
    st.subheader("📈 Regional Price Benchmarking")
    st.caption("Compare average costs across different geographic MSAs.")
    
    c2_msa = st.multiselect("Select Markets:", options=sorted(df['cbsa_name'].unique()), default=df['cbsa_name'].unique(), key="c2_msa")
    c2_svc = st.selectbox("Select Service:", options=sorted(df['service_name_group'].unique()), key="c2_svc")
    
    df_c2 = df[(df['cbsa_name'].isin(c2_msa)) & (df['service_name_group'] == c2_svc)]
    
    if not df_c2.empty:
        avg_price = df_c2.groupby('cbsa_name')['cost'].mean().reset_index()
        fig2 = px.bar(avg_price, x='cbsa_name', y='cost', color='cbsa_name', text_auto='.2f', template="plotly_white")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("No data for this selection.")

# --- SECTION 2: UNIT ECONOMICS PREDICTOR ---
st.divider()
st.subheader("🧮 Unit Economics Predictor")
st.info("Calculate predicted quotes based on the mathematical slopes (Linear Regression) found in the dataset.")

pred_col1, pred_col2, pred_col3 = st.columns([1.5, 1.5, 2])

with pred_col1:
    pred_svc = st.selectbox("Target Service Type:", options=sorted(df['service_name_group'].unique()), key="p_svc")

with pred_col2:
    pred_msas = st.multiselect("Target Markets to Compare:", options=sorted(df['cbsa_name'].unique()), default=df['cbsa_name'].unique(), key="p_msa")

with pred_col3:
    test_size = st.number_input("Enter Property Size for Quote (sq ft):", min_value=0, value=5000, step=500)

# Prediction Logic
pred_subset = df[(df['service_name_group'] == pred_svc) & (df['cbsa_name'].isin(pred_msas))]

if len(pred_subset) > 5:
    res = []
    for msa in pred_msas:
        m_data = pred_subset[pred_subset['cbsa_name'] == msa]
        if len(m_data) > 2:
            # OLS Regression
            z = np.polyfit(m_data['lot_size'], m_data['cost'], 1)
            p_price = max(0, z[1] + (z[0] * test_size))
            res.append({
                "Market": msa, 
                "Predicted Quote": f"${p_price:.2f}", 
                "Base Trip Fee": f"${max(0, z[1]):.2f}", 
                "Variable Rate (per sqft)": f"${z[0]:.4f}"
            })
    
    if res:
        predict_df = pd.DataFrame(res)
        st.table(predict_df)
    else:
        st.warning("Insufficient data in selected markets to generate a formula.")
else:
    st.info("The sample size for this service is currently too small for accurate regression modeling.")

# --- EXPORT SECTION ---
st.divider()
st.subheader("📂 Data Portability")
c_dl1, c_dl2 = st.columns(2)

with c_dl1:
    # Full CSV Download
    csv_full = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Full Scraped Dataset (CSV)", csv_full, "weedman_full_analysis.csv", "text/csv")

with c_dl2:
    # Prediction Results Download
    if 'predict_df' in locals() and not predict_df.empty:
        csv_pred = predict_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Current Predictions (CSV)", csv_pred, "pricing_predictions.csv", "text/csv")
