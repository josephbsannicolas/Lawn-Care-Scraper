import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# 1. PROFESSIONAL THEME & CONFIG
st.set_page_config(page_title="Market Intel: Weedman Pricing Analysis", layout="wide", initial_sidebar_state="expanded")

# --- MOBILE & CONTRAST OPTIMIZATION CSS ---
st.markdown("""
    <style>
    /* Main Background */
    .main { background-color: #fcfcfc; }
    
    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #cccccc; 
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    
    /* UNIVERSAL LABEL FIX: Targets the 'Quotes Captured', 'Market Footprint' etc. */
    /* This overrides Streamlit's gray/transparent mobile labels */
    [data-testid="stMetricLabel"] {
        color: #1a1a1a !important;
        opacity: 1 !important;
        visibility: visible !important;
    }
    
    [data-testid="stMetricLabel"] * {
        color: #1a1a1a !important;
        opacity: 1 !important;
    }
    
    /* UNIVERSAL VALUE FIX: Targets the actual numbers */
    [data-testid="stMetricValue"] {
        color: #000000 !important;
        opacity: 1 !important;
    }

    /* Responsive Stacking: Force 1-column on mobile portrait */
    @media (max-width: 640px) {
        div[data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 100% !important;
            margin-bottom: 12px;
        }
    }
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
    st.markdown("---")
    st.markdown("### 👤 Author Info")
    st.markdown("**Joseph San Nicolas**")
    st.caption("Analytics Professional")
    # Optional: Add your LinkedIn link
    # st.markdown("[View LinkedIn Profile](https://www.linkedin.com/in/yourprofile)")
    st.markdown("---")
    st.info("""
    **Objective:** Reverse-engineer Weedman’s dynamic pricing engine across key MSAs.
    **Methodology:** OLS Modeling ($y = mx + b$) to isolate fixed vs. variable drivers.
    """)
    st.caption(f"Last Update: {df['scrape_timestamp'].max().strftime('%b %d, %Y')}")

# --- HEADER: EXECUTIVE SUMMARY ---
st.title("📊 Competitive Intelligence: Weedman Pricing Strategy")

# Metric cards - Forced high-contrast labels and values
m1, m2, m3 = st.columns(3)
m1.metric("Quotes Captured", f"{len(df):,}")
m2.metric("Market Footprint", f"{df['cbsa_name'].nunique()} MSAs")
m3.metric("Data Recency", df['scrape_timestamp'].max().strftime('%Y-%m-%d'))

with st.expander("📝 Strategic Key Takeaways", expanded=True):
    st.markdown("""
    * **Dynamic Indexing:** Switch the 'Baseline Market' to perform a direct gap analysis between specific regions.
    * **Normalization:** Comparisons are based on a 'Standardized 5k sqft Property' to remove bias from varying lot sizes.
    * **Unit Economics:** The model separates the 'Base Fee' (Fixed) from the 'Variable Rate' (Per sqft).
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
        df_c1, x="lot_size", y="total_cost", color="cbsa_name", trendline="ols",
        hover_data=["input_address"],
        labels={"lot_size": "Lot Size (sq ft)", "total_cost": "Quote Amount"},
        template="plotly_white", height=500
    )
    fig1.update_layout(yaxis=dict(tickprefix="$", tickformat=",.2f"))
    st.plotly_chart(fig1, use_container_width=True)

st.divider()

# --- SECTION 2: REGIONAL PRICING INDEX ---
st.header("2. Regional Pricing Index & Gap Analysis")

with st.expander("❓ How to read this Gap Analysis"):
    st.markdown("""
    **The Goal:** To compare markets "Apples-to-Apples" regardless of house size.
    
    1.  **Normalization:** We calculate a quote for a **Standard 5,000 sq ft property** in every city using the pricing engine logic ($y = mx + b$).
    2.  **The Baseline (100):** We set one market (or the average) as the benchmark. Its value is exactly **100.0**.
    3.  **The Index:**
        * **Above 100:** That market is *more expensive* than your baseline (e.g., 110.5 = 10.5% higher).
        * **Below 100:** That market is *cheaper* than your baseline (e.g., 95.0 = 5% lower).
    """)

idx_col1, idx_col2 = st.columns([1, 1])
with idx_col1:
    c2_svc = st.selectbox("Calculate Index for:", options=sorted(df['service_name_group'].unique()), key="bench_svc")
with idx_col2:
    baseline_options = ["Market Average"] + sorted(df['cbsa_name'].unique().tolist())
    baseline_market = st.selectbox("Select Baseline Market (100.0):", options=baseline_options, key="baseline_market")

df_c2 = df[df['service_name_group'] == c2_svc]

if not df_c2.empty:
    index_results = []
    for msa in sorted(df['cbsa_name'].unique()):
        m_data = df_c2[df_c2['cbsa_name'] == msa]
        if len(m_data) > 3: 
            z = np.polyfit(m_data['lot_size'], m_data['total_cost'], 1)
            std_price = max(0, z[1] + (z[0] * 5000))
            index_results.append({"Market": msa, "Standard Price": std_price, "n": len(m_data)})
    
    if index_results:
        idx_df = pd.DataFrame(index_results)
        
        if baseline_market == "Market Average":
            baseline_price = idx_df['Standard Price'].mean()
        else:
            baseline_price = idx_df[idx_df['Market'] == baseline_market]['Standard Price'].iloc[0]
            
        idx_df['Pricing Index'] = (idx_df['Standard Price'] / baseline_price) * 100
        idx_df = idx_df.sort_values('Pricing Index', ascending=False)
        idx_df['label'] = idx_df.apply(lambda x: f"{x['Market']} (n={int(x['n'])})", axis=1)

        fig2 = px.bar(
            idx_df, x='Pricing Index', y='label', orientation='h',
            color='Pricing Index', color_continuous_scale='RdYlGn_r',
            text='Pricing Index',
            template="plotly_white", height=500
        )
        fig2.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        fig2.add_vline(x=100, line_dash="dash", line_color="black", annotation_text=f"Baseline: {baseline_market}")
        
        fig2.update_layout(showlegend=False, yaxis_title=None, xaxis_title=f"Index (100 = {baseline_market})", margin=dict(r=50))
        st.plotly_chart(fig2, use_container_width=True)

st.divider()

# --- SECTION: MARKET PRICE DISPERSION (HORIZONTAL) ---
st.header("3. Price Distribution Analysis")
st.markdown("### Regional Variance & Range")

# Using the existing service selection from your sidebar or previous section
# If you don't have c2_svc defined yet, uncomment the line below:
# c2_svc = st.selectbox("Select Service for Distribution:", options=sorted(df['service_name_group'].unique()))

df_disp = df[df['service_name_group'] == c2_svc]

if not df_disp.empty:
    fig_disp = px.box(
        df_disp, 
        y="cbsa_name",  # Market on Y-axis for horizontal layout
        x="total_cost", # Price on X-axis
        color_discrete_sequence=["#2E7D32"], # Professional Forest Green
        points="outliers", # Show the specific noisy points we discussed
        labels={
            "cbsa_name": "Market", 
            "total_cost": "Quote Amount ($)"
        },
        template="plotly_white",
        height=600 # Taller height helps with long market lists
    )

    # Clean up the layout for a corporate feel
    fig_disp.update_layout(
        xaxis_tickprefix="$",
        yaxis_title=None,
        xaxis_title="Observed Quote Price Range",
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    # Force the chart to be fully visible on mobile
    st.plotly_chart(fig_disp, use_container_width=True)

with st.expander("📝 Strategic Interpretation of Distribution"):
    st.markdown("""
    * **Median Line:** Represents the 'typical' market price. 
    * **Box Width:** Shows the Middle 50% of the market. A **tight box** indicates a highly automated, disciplined competitor.
    * **Whiskers & Dots:** Identify the range and 'Edge Cases'—useful for spotting where the competitor's pricing logic breaks down.
    """)

st.divider()

# --- SECTION 3: UNIT ECONOMICS PREDICTOR ---
st.header("3. Unit Economics Predictor")
st.caption("Detailed rate card components reverse-engineered from captured data.")

p1, p2 = st.columns(2)
with p1: pred_svc = st.selectbox("Select Service to Quote:", options=sorted(df['service_name_group'].unique()), key="p_svc")
with p2: test_size = st.number_input("Input Property Size (sq ft):", min_value=0, value=5000, step=500)

pred_subset = df[df['service_name_group'] == pred_svc]
res = []
for msa in sorted(df['cbsa_name'].unique()):
    m_data = pred_subset[pred_subset['cbsa_name'] == msa]
    if len(m_data) > 2:
        z = np.polyfit(m_data['lot_size'], m_data['total_cost'], 1)
        p_price = max(0, z[1] + (z[0] * test_size))
        res.append({
            "Market": msa, 
            "Predicted Quote": p_price, 
            "Base Fee (Fixed)": max(0, z[1]), 
            "Rate (Variable per sqft)": z[0]
        })

if res:
    predict_df = pd.DataFrame(res).sort_values("Predicted Quote", ascending=False)
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
    st.download_button(label="Download Predictions (CSV)", data=predict_df.to_csv(index=False).encode('utf-8'), file_name=f"weedman_predictions_{pred_svc}.csv", mime="text/csv")

with dl_col2:
    st.download_button(label="Download Full Scraped Data (CSV)", data=df.to_csv(index=False).encode('utf-8'), file_name="weedman_full_dataset.csv", mime="text/csv")

st.caption("CONFIDENTIAL: For Strategic Review Only")
