import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# Page Config
st.set_page_config(page_title="Weedman Pricing Engine Reverse Engineering", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("weedman_sample_quotes_clean.csv")
    # Clean up any trailing decimals in lot_size for the UI
    df['lot_size'] = df['lot_size'].astype(int)
    return df

df = load_data()

# --- SIDEBAR FILTERS ---
st.sidebar.header("Filter Analytics")
selected_msa = st.sidebar.multiselect("Select MSA (CBSA)", options=df['cbsa_name'].unique(), default=df['cbsa_name'].unique())
selected_service = st.sidebar.selectbox("Select Service to Analyze", options=df['service_name_group'].unique())

# Filter data based on selection
filtered_df = df[(df['cbsa_name'].isin(selected_msa)) & (df['service_name_group'] == selected_service)]

# --- MAIN DASHBOARD ---
st.title("🎯 Weedman Pricing Engine: Reverse Engineered")
st.markdown(f"Analysis of **{selected_service}** across **{len(selected_msa)}** markets.")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Price Elasticity (Cost vs. Lot Size)")
    # Scatter with Trendline (The Regression Line IS the Engine)
    fig = px.scatter(
        filtered_df, x="lot_size", y="cost", 
        color="cbsa_name", trendline="ols",
        hover_data=["input_address"],
        labels={"lot_size": "Lot Size (sq ft)", "cost": "Quote ($)"},
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Market Comparison")
    # Show average cost per MSA for the selected service
    avg_price = filtered_df.groupby('cbsa_name')['cost'].mean().reset_index()
    fig2 = px.bar(avg_price, x='cbsa_name', y='cost', color='cbsa_name', text_auto='.2f')
    st.plotly_chart(fig2, use_container_width=True)

# --- THE "INTERVIEW WOW" SECTION ---
st.divider()
st.subheader("🧮 The 'Predictor' (Reverse Engineered Formula)")

# Simple Linear Regression Logic for the selected view
if len(filtered_df) > 1:
    z = np.polyfit(filtered_df['lot_size'], filtered_df['cost'], 1)
    slope, intercept = z[0], z[1]
    
    st.info(f"**Calculated Formula for {selected_service}:** \n"
            f"Base Trip Charge: **${intercept:.2f}** \n"
            f"Price per sq ft: **${slope:.4f}**")
    
    # Interactive Predictor
    test_size = st.slider("Test a Lot Size (sq ft):", 1000, 20000, 5000)
    predicted_price = intercept + (slope * test_size)
    st.metric("Predicted Weedman Quote", f"${predicted_price:.2f}")
else:
    st.warning("Not enough data points in this selection to calculate a pricing formula.")

st.dataframe(filtered_df.sort_values("lot_size"), use_container_width=True)
