"""
Sri Lanka Used Bike Price — Ensemble Learning & Hybridization Dashboard
Student Index: 224114G
Dataset: 5,016 Sri Lankan Used Bike Listings scraped from ikman.lk (used-bikes.csv)

GenAI Disclosure: Front-end Streamlit UI layout and Plotly charting components were
developed with Generative AI (Google Gemini) assistance.
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure custom hybrid regressor class is importable when unpickling models
from hybrid_model import RFANNHybridRegressor  # noqa: F401

st.set_page_config(
    page_title="Sri Lanka Used Bike Ensemble Dashboard",
    page_icon="🏍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom Styling ---
st.markdown(
    """
    <style>
    .metric-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 18px;
        color: #f8fafc;
    }
    .metric-label { font-size: 0.82rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.04em; }
    .metric-value { font-size: 1.55rem; font-weight: 700; color: #38bdf8; margin-top: 4px; }
    .metric-sub { font-size: 0.8rem; color: #cbd5e1; margin-top: 2px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    df_clean = pd.read_csv("models/cleaned_bikes.csv")
    metrics_df = pd.read_csv("models/model_metrics.csv")
    pred_df = pd.read_csv("models/test_predictions.csv")
    shap_df = pd.read_csv("models/shap_importance.csv")
    with open("models/feature_metadata.json", "r") as f:
        meta = json.load(f)
    return df_clean, metrics_df, pred_df, shap_df, meta


@st.cache_resource
def load_models(model_files):
    loaded = {}
    for name, path in model_files.items():
        if os.path.exists(path):
            loaded[name] = joblib.load(path)
    return loaded


if not os.path.exists("models/feature_metadata.json"):
    st.error("Model artifacts not found in `models/`. Please run `sri_lanka_bike_ensemble.ipynb` first.")
    st.stop()

df_clean, metrics_df, pred_df, shap_df, meta = load_data()
models = load_models(meta["model_files"])

# --- Sidebar ---
with st.sidebar:
    st.title("🏍️ SL Bike Ensemble AI")
    st.caption("Index: **224114G** | Sri Lankan Context ML Project")
    st.markdown(
        f"""
        **Dataset:** `used-bikes.csv` (ikman.lk)  
        **Raw Listings:** {meta['dataset_raw_rows']:,}  
        **Cleaned Listings:** {meta['dataset_clean_rows']:,}  
        **Dealer Down-Payments Filtered:** {meta['removed_down_payments']:,}  
        **Top Model:** `{meta['best_model']}`
        """
    )
    st.divider()
    st.info(
        "**GenAI Disclosure:** Generative AI (Google Gemini) assisted in writing Python "
        "modeling boilerplate and designing this Streamlit dashboard UI. All Sri Lankan "
        "data cleaning rules, model evaluations, and interpretations were verified locally."
    )

# --- Header ---
st.title("Sri Lankan Used Motorbike Price — Ensemble Learning Dashboard")
st.caption(
    "Comparing **Bagging** (Random Forest), **Boosting** (XGBoost, LightGBM, CatBoost), "
    "**Stacking** (Multi-Model Meta-Ensemble), and **Hybridization** (RF + ANN) on real ikman.lk listings."
)

# Top KPI Row
best_row = metrics_df.iloc[0]
stack_row = metrics_df[metrics_df["Family"] == "Stacking"].iloc[0]
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Clean SL Listings</div>'
        f'<div class="metric-value">{meta["dataset_clean_rows"]:,}</div>'
        f'<div class="metric-sub">{meta["removed_down_payments"]} dealer down-payments removed</div></div>',
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Median Market Price</div>'
        f'<div class="metric-value">Rs {df_clean["Price_LKR"].median():,.0f}</div>'
        f'<div class="metric-sub">Across {df_clean["District"].nunique()} Sri Lankan districts</div></div>',
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Best RMSE / R²</div>'
        f'<div class="metric-value">R² {best_row["R2"]:.4f}</div>'
        f'<div class="metric-sub">{best_row["Model"]} (Rs {best_row["RMSE_LKR"]:,.0f})</div></div>',
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Best MAPE (Stacking)</div>'
        f'<div class="metric-value">{stack_row["MAPE_pct"]:.2f}%</div>'
        f'<div class="metric-sub">MAE: Rs {stack_row["MAE_LKR"]:,.0f}</div></div>',
        unsafe_allow_html=True,
    )

st.write("")

# --- 4 Main Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Sri Lankan Market EDA",
    "🏆 Ensemble Leaderboard",
    "🧠 Explainability & Research",
    "⚡ Live Bike Price Predictor",
])

# ===================== TAB 1: EDA =====================
with tab1:
    st.subheader("Sri Lankan Used Two-Wheeler Market Insights (ikman.lk)")
    st.write(
        "A major local domain finding in raw ikman.lk listings is that commercial **MEMBER** dealers "
        "often post **leasing down payments** (Rs 70,000–115,000) rather than full cash prices. "
        f"Filtering out **{meta['removed_down_payments']}** dealer down-payment anomalies and typo outliers "
        "produces a clean, realistic market dataset."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        fig_hist = px.histogram(
            df_clean,
            x="Price_LKR",
            color="Bike Type",
            nbins=45,
            title="Used Bike Price Distribution in Sri Lanka (LKR)",
            labels={"Price_LKR": "Selling Price (LKR)"},
            color_discrete_sequence=["#0ea5e9", "#f97316"],
        )
        fig_hist.update_layout(bargap=0.05)
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_b:
        top_brands = df_clean["Brand"].value_counts().head(10).index
        brand_df = df_clean[df_clean["Brand"].isin(top_brands)]
        brand_order = (
            brand_df.groupby("Brand")["Price_LKR"].median().sort_values(ascending=False).index.tolist()
        )
        fig_box = px.box(
            brand_df,
            x="Brand",
            y="Price_LKR",
            color="is_japanese_brand",
            category_orders={"Brand": brand_order},
            title="Price Tier by Brand (Japanese vs. Non-Japanese Brands)",
            labels={"Price_LKR": "Price (LKR)", "is_japanese_brand": "Japanese Brand (1=Yes)"},
            color_discrete_map={0: "#f97316", 1: "#2563eb"},
        )
        st.plotly_chart(fig_box, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        top_dist = df_clean["District"].value_counts().head(12).index
        dist_stats = (
            df_clean[df_clean["District"].isin(top_dist)]
            .groupby("District")
            .agg(Median_Price_LKR=("Price_LKR", "median"), Listings=("Price_LKR", "count"))
            .sort_values("Median_Price_LKR")
            .reset_index()
        )
        fig_dist = px.bar(
            dist_stats,
            x="Median_Price_LKR",
            y="District",
            orientation="h",
            color="Listings",
            title="Median Bike Price Across Top 12 Sri Lankan Districts",
            labels={"Median_Price_LKR": "Median Price (LKR)"},
            color_continuous_scale="Tealgrn",
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_d:
        sample_df = df_clean.sample(min(1500, len(df_clean)), random_state=42)
        fig_age = px.scatter(
            sample_df,
            x="Year",
            y="Price_LKR",
            color="Capacity_cc",
            hover_data=["Brand", "Model", "District", "Mileage_km"],
            title="Depreciation Curve: Manufacture Year vs. Price (Colored by Engine CC)",
            labels={"Price_LKR": "Price (LKR)", "Capacity_cc": "Engine (cc)"},
            color_continuous_scale="Viridis",
        )
        st.plotly_chart(fig_age, use_container_width=True)

# ===================== TAB 2: LEADERBOARD =====================
with tab2:
    st.subheader("Bagging vs. Boosting vs. Stacking vs. Hybrid (RF + ANN)")

    display_metrics = metrics_df.copy()
    display_metrics["RMSE_LKR"] = display_metrics["RMSE_LKR"].map(lambda x: f"Rs {x:,.0f}")
    display_metrics["MAE_LKR"] = display_metrics["MAE_LKR"].map(lambda x: f"Rs {x:,.0f}")
    display_metrics["R2"] = display_metrics["R2"].map(lambda x: f"{x:.4f}")
    display_metrics["MAPE_pct"] = display_metrics["MAPE_pct"].map(lambda x: f"{x:.2f}%")
    display_metrics["Train_Time_s"] = display_metrics["Train_Time_s"].map(lambda x: f"{x:.2f} s")
    st.dataframe(display_metrics, use_container_width=True, hide_index=True)

    c_left, c_right = st.columns(2)
    with c_left:
        fig_rmse = px.bar(
            metrics_df.sort_values("RMSE_LKR", ascending=False),
            x="RMSE_LKR",
            y="Model",
            color="Family",
            orientation="h",
            text_auto=",.0f",
            title="Test RMSE (LKR) — Lower is Better",
            labels={"RMSE_LKR": "RMSE (LKR)", "Model": ""},
        )
        st.plotly_chart(fig_rmse, use_container_width=True)

    with c_right:
        fig_r2 = px.scatter(
            metrics_df,
            x="Train_Time_s",
            y="R2",
            size="MAE_LKR",
            color="Family",
            text="Model",
            title="Accuracy (R²) vs. Training Time (Speed-Accuracy Trade-off)",
            labels={"Train_Time_s": "Training Time (seconds)", "R2": "Test R² Score"},
        )
        fig_r2.update_traces(textposition="top center")
        st.plotly_chart(fig_r2, use_container_width=True)

    st.divider()
    st.subheader("Diagnostic Inspector: Actual vs. Predicted & Residual Errors")
    chosen_model = st.selectbox("Select a model to inspect on the 837 test bikes:", metrics_df["Model"].tolist())

    d1, d2 = st.columns(2)
    with d1:
        max_val = max(pred_df["Actual_Price_LKR"].max(), pred_df[chosen_model].max())
        fig_avp = px.scatter(
            pred_df,
            x="Actual_Price_LKR",
            y=chosen_model,
            color="Brand",
            hover_data=["Model", "Year", "Capacity_cc", "District"],
            title=f"Actual vs. Predicted Price — {chosen_model}",
            labels={"Actual_Price_LKR": "Actual Price (LKR)", chosen_model: "Predicted Price (LKR)"},
            opacity=0.65,
        )
        fig_avp.add_trace(
            go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode="lines",
                line=dict(color="red", dash="dash"),
                name="Perfect Prediction",
            )
        )
        st.plotly_chart(fig_avp, use_container_width=True)

    with d2:
        residuals = pred_df["Actual_Price_LKR"] - pred_df[chosen_model]
        fig_res = px.histogram(
            x=residuals,
            nbins=45,
            title=f"Residual Distribution (Actual − Predicted) — {chosen_model}",
            labels={"x": "Prediction Error (LKR)"},
            color_discrete_sequence=["#10b981"],
        )
        fig_res.add_vline(x=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_res, use_container_width=True)

# ===================== TAB 3: EXPLAINABILITY & RESEARCH =====================
with tab3:
    st.subheader("Global Feature Importance (SHAP on LightGBM)")
    fig_shap = px.bar(
        shap_df.head(15).sort_values("Mean_Abs_SHAP"),
        x="Mean_Abs_SHAP",
        y="Feature",
        orientation="h",
        color="Mean_Abs_SHAP",
        color_continuous_scale="Blues",
        title="Top 15 Sri Lankan Used Bike Price Drivers (Mean |SHAP| on Log Price)",
        labels={"Mean_Abs_SHAP": "Mean |SHAP Value|", "Feature": "Feature"},
    )
    st.plotly_chart(fig_shap, use_container_width=True)

    st.subheader("Why Did Each Model Perform This Way?")
    st.markdown(
        """
        - **Single Decision Tree (R² = 0.5775):** High variance; creates jagged step-function price predictions and overfits noisy ikman.lk listings.
        - **Random Forest / Bagging (R² = 0.6852):** Averaging 300 bootstrapped trees cuts variance sharply (+0.108 R² over a single tree), but independent trees still leave bias on rare Japanese models and high-cc bikes.
        - **Gradient Boosting — CatBoost (R² = 0.7263), XGBoost (R² = 0.7131), LightGBM (R² = 0.6860):** Sequential error-correction focuses each new tree on hard-to-price bikes. **CatBoost** achieved the lowest RMSE (**Rs 109,798**) using symmetric (oblivious) trees that regularize high-cardinality `Brand × Model × District` splits, while **XGBoost** achieved the lowest MAE (**Rs 56,673**) and **LightGBM** trained in just **0.19s**.
        - **Stacking Ensemble (R² = 0.7203, Best MAPE = 20.67%):** Blending Random Forest, XGBoost, LightGBM, and CatBoost through a regularized `RidgeCV` meta-learner delivers the lowest percentage error across the full price spectrum.
        - **Hybrid RF + ANN (R² = 0.6358):** Using Random Forest's out-of-fold predictions + cross-tree uncertainty (`std`) as inputs to a residual-learning MLP Neural Network cleanly beats the baseline tree (+0.058 R²), while empirically showing why tree-based boosting still edges out neural networks on medium-sized (~4,181 row) sparse tabular datasets.
        """
    )

    st.subheader("Research Comparison (Lecture Slide 44): Tabular vs. Image/Text Data")
    research_table = pd.DataFrame(
        {
            "Model Family": [
                "Tree Ensembles (RF, XGBoost, LightGBM, CatBoost)",
                "Deep Learning (CNNs, Transformers, ANNs)",
            ],
            "Tabular Data (e.g., ikman.lk Bike Listings)": [
                "Dominant on <100k rows; naturally handles mixed numeric/categorical scales, sharp thresholds (Year, CC), and non-linear interactions with fast CPU training.",
                "Needs heavy scaling, regularization, and much larger datasets; works best on tabular data when hybridized with tree ensembles (e.g., RF + ANN).",
            ],
            "Image & Text Data (Unstructured)": [
                "Weak on raw pixels or free-form Sinhala/English ad text because axis-aligned splits lack spatial or sequential inductive bias.",
                "State-of-the-art; CNNs/ViTs detect bike scratches from photos, and Transformers extract condition/urgency cues from ad descriptions.",
            ],
        }
    )
    st.table(research_table)

# ===================== TAB 4: LIVE PREDICTOR =====================
with tab4:
    st.subheader("Live Sri Lankan Used Bike Price Estimator (All Ensembles Side-by-Side)")
    st.write("Configure any bike below to compare live predictions across Bagging, Boosting, Stacking, and Hybrid RF + ANN.")

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        bike_type = st.selectbox("Bike Type", meta["bike_types"], index=0)
        brand = st.selectbox("Brand", meta["brands"], index=meta["brands"].index("Bajaj") if "Bajaj" in meta["brands"] else 0)
    with f2:
        avail_models = meta["brand_models"].get(brand, ["Other Model"])
        model_name = st.selectbox("Model", avail_models)
        district = st.selectbox("District", meta["districts"], index=meta["districts"].index("Colombo") if "Colombo" in meta["districts"] else 0)
    with f3:
        year = st.slider("Year of Manufacture", min_value=meta["year_min"], max_value=meta["year_max"], value=2018)
        default_cc = int(meta["capacity_median_by_model"].get(model_name, 125))
        default_cc = max(48, min(650, default_cc))
        capacity_cc = st.number_input("Engine Capacity (cc)", min_value=45, max_value=1000, value=default_cc, step=5)
    with f4:
        mileage_km = st.number_input("Mileage (km)", min_value=500, max_value=250000, value=30000, step=1000)
        seller_type = st.radio("Seller Type", ["Private Seller", "MEMBER Dealer"], horizontal=True)

    if st.button("Predict Market Price (LKR)", type="primary", use_container_width=True):
        bike_age = max(0, 2022 - year)
        input_row = pd.DataFrame(
            [
                {
                    "Year": year,
                    "bike_age": bike_age,
                    "Mileage_km": float(mileage_km),
                    "km_per_year": float(mileage_km) / (bike_age + 1.0),
                    "log_mileage": float(np.log1p(mileage_km)),
                    "Capacity_cc": float(capacity_cc),
                    "is_member_dealer": 1 if seller_type == "MEMBER Dealer" else 0,
                    "is_western_province": 1 if district in ["Colombo", "Gampaha", "Kalutara"] else 0,
                    "is_japanese_brand": 1 if brand in ["Honda", "Yamaha", "Suzuki", "Kawasaki"] else 0,
                    "Bike Type": bike_type,
                    "Brand": brand,
                    "Model": model_name,
                    "District": district,
                }
            ]
        )

        pred_rows = []
        for m_name, pipe in models.items():
            if "Baseline" in m_name:
                continue
            p_val = float(np.clip(pipe.predict(input_row)[0], 35_000, None))
            fam = metrics_df.loc[metrics_df["Model"] == m_name, "Family"].values[0]
            pred_rows.append({"Model": m_name, "Family": fam, "Predicted_Price_LKR": p_val})

        live_df = pd.DataFrame(pred_rows).sort_values("Predicted_Price_LKR", ascending=False)
        best_pred = live_df.loc[live_df["Model"] == meta["best_model"], "Predicted_Price_LKR"].values[0]
        low_p, high_p = live_df["Predicted_Price_LKR"].min(), live_df["Predicted_Price_LKR"].max()

        m1, m2, m3 = st.columns(3)
        m1.metric(f"Recommended Estimate ({meta['best_model']})", f"Rs {best_pred:,.0f}")
        m2.metric("Ensemble Mean Estimate", f"Rs {live_df['Predicted_Price_LKR'].mean():,.0f}")
        m3.metric("Model Consensus Range", f"Rs {low_p:,.0f} – Rs {high_p:,.0f}")

        fig_live = px.bar(
            live_df,
            x="Predicted_Price_LKR",
            y="Model",
            color="Family",
            orientation="h",
            text=live_df["Predicted_Price_LKR"].map(lambda x: f"Rs {x:,.0f}"),
            title=f"Live Ensemble Price Comparison: {year} {brand} {model_name} ({capacity_cc}cc, {district})",
            labels={"Predicted_Price_LKR": "Predicted Price (LKR)", "Model": ""},
        )
        fig_live.update_traces(textposition="outside")
        st.plotly_chart(fig_live, use_container_width=True)
