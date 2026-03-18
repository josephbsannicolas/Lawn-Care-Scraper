import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# Page Config
st.set_page_config(page_title="Weedman Pricing Engine Analysis", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("weedman_sample_quotes_clean.csv")
    df['lot_size'] = df['lot_size'].astype(int)
    return df

df = load_data()

# --- SIDEBAR FILTERS ---
st.sidebar.header("Global Filters")
selected_msa = st.sidebar.multiselect("Select Markets (CBSA)", options=df['cbsa_name'].unique(), default=df['cbsa_name'].unique())
selected_service = st.sidebar.selectbox("Select Service to Analyze", options=df['service_name_group'].unique())

filtered_df = df[(df['cbsa_name'].isin(selected_msa)) & (df['service_name_group'] == selected_service)]

# --- MAIN DASHBOARD ---
st.title("🎯 Weedman Pricing Engine: Reverse Engineered")
st.markdown(f"Uncovering the logic for **{selected_service}** across **{len(selected_msa)}** regions.")

col1, col2 = st.columns([2, 1])

with col1:
    # UPDATED LABEL: From Elasticity to Pricing Logic
    st.subheader("Pricing Logic & Scalability")
    fig = px.scatter(
        filtered_df, x="lot_size", y="cost", 
        color="cbsa_name", trendline="ols",
        hover_data=["input_address"],
        labels={"lot_size": "Lot Size (sq ft)", "cost": "Quote Amount ($)"},
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # UPDATED LABEL: Market Variation
    st.subheader("Regional Price Benchmarking")
    avg_price = filtered_df.groupby('cbsa_name')['cost'].mean().reset_index()
    fig2 = px.bar(avg_price, x='cbsa_name', y='cost', color='cbsa_name', text_auto='.2f')
    st.plotly_chart(fig2, use_container_width=True)

# --- THE "INTERVIEW WOW" SECTION ---
st.divider()
st.subheader("🧮 Unit Economics Predictor")

pred_col1, pred_col2 = st.columns(2)

with pred_col1:
    prediction_service = st.selectbox(
        "Service Type:", 
        options=df['service_name_group'].unique(),
        key="predictor_service_select"
    )

with pred_col2:
    test_size = st.number_input(
        "Property Size (sq ft):", 
        min_value=0, 
        value=5000, 
        step=500,
        key="predictor_size_input"
    )

pred_df_subset = df[df['service_name_group'] == prediction_service]

if len(pred_df_subset) > 5:
    prediction_results = []
    for msa in selected_msa:
        msa_data = pred_df_subset[pred_df_subset['cbsa_name'] == msa]
        if len(msa_data) > 2:
            z = np.polyfit(msa_data['lot_size'], msa_data['cost'], 1)
            slope, intercept = z[0], z[1]
            predicted_price = max(0, intercept + (slope * test_size))
            
            prediction_results.append({
                "Market": msa,
                "Predicted Quote": f"${predicted_price:.2f}",
                "Base Fee (Fixed)": f"${max(0, intercept):.2f}",
                "Rate (Variable per sqft)": f"${slope:.4f}"
            })

    if prediction_results:
        st.write(f"### Regional Quote Breakdown for: **{prediction_service}**")
        predict_df = pd.DataFrame(prediction_results)
        st.table(predict_df)
        st.success("Mathematical model successfully extracted from scraped data.")
    else:
        st.warning(f"Insufficient data for {prediction_service} in these markets.")

# --- EXPORT SECTION ---
st.divider()
st.subheader("📂 Export Data & Insights")
col_dl1, col_dl2 = st.columns(2)

if 'predict_df' in locals() and not predict_df.empty:
    with col_dl1:
        csv_predictions = predict_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Predictions (CSV)",
            data=csv_predictions,
            file_name=f"weedman_predictions_{test_size}sqft.csv",
            mime="text/csv"
        )

with col_dl2:
    csv_raw = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Raw Dataset (CSV)",
        data=csv_raw,
        file_name="weedman_raw_data.csv",
        mime="text/csv"
    )
