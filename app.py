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
st.subheader("🧮 Regional Price Predictor")

# 1. INPUT: Changed from Slider to Number Input (Free Text)
test_size = st.number_input("Enter a Lot Size to Quote (sq ft):", min_value=0, value=5000, step=100)

# 2. CALCULATION: Create a prediction for EVERY selected MSA
if len(filtered_df) > 5:  # Need at least a few points for a valid regression
    prediction_results = []
    
    # We loop through each MSA to find its specific slope/intercept
    for msa in selected_msa:
        msa_data = filtered_df[filtered_df['cbsa_name'] == msa]
        
        if len(msa_data) > 2:
            z = np.polyfit(msa_data['lot_size'], msa_data['cost'], 1)
            slope, intercept = z[0], z[1]
            predicted_price = intercept + (slope * test_size)
            
            prediction_results.append({
                "MSA": msa,
                "Predicted Quote": f"${predicted_price:.2f}",
                "Base Fee": f"${intercept:.2f}",
                "Price/sqft": f"${slope:.4f}"
            })

    if prediction_results:
        st.write(f"**Predicted Quotes for {selected_service} at {test_size:,} sq ft:**")
        
        # Display as a clean, professional table
        predict_df = pd.DataFrame(prediction_results)
        st.table(predict_df)
        
        st.caption("Note: Base Fee represents the 'trip charge' intercept, while Price/sqft is the variable rate.")
    else:
        st.warning("Select more MSAs or a service with more data to see regional predictions.")
else:
    st.info("Gathering more data points will increase the accuracy of these regional formulas.")

# --- EXPORT SECTION ---
st.divider()
st.subheader("📂 Export Analysis for Review")

col_dl1, col_dl2 = st.columns(2)

# 1. Export the Predictions Table (The "Insights")
if 'predict_df' in locals() and not predict_df.empty:
    with col_dl1:
        csv_predictions = predict_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Pricing Predictions (CSV)",
            data=csv_predictions,
            file_name=f"weedman_predictions_{test_size}sqft.csv",
            mime="text/csv",
            help="Exports the calculated quotes for all selected MSAs"
        )

# 2. Export the Raw Filtered Data (The "Evidence")
with col_dl2:
    csv_raw = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Raw Filtered Data (CSV)",
        data=csv_raw,
        file_name="weedman_raw_analysis_data.csv",
        mime="text/csv",
        help="Exports the actual scraped records used for this view"
    )

st.caption("Note: For PDF reporting, please use the 'Print' function (Ctrl+P / Cmd+P) in your browser to save this dashboard as a PDF.")
