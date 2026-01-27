import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import os

# =========================
# Load trained final model
# =========================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "sales_prediction_model.pkl")
model = joblib.load(MODEL_PATH)

# Load dataset
df = pd.read_csv("cleaned_superstore_sales.csv")

# =========================
# App Layout
# =========================
st.set_page_config(page_title="Sales Predictor", page_icon="🛒", layout="wide")

st.title("🛒 Superstore Sales Predictor")
st.markdown("Predict **Sales ($)** based on product and order specifications using a Random Forest Regression model.")

# Soft Divider CSS
st.markdown(
    """
    <style>
    .soft-divider {
        height: 1px;
        background: linear-gradient(to right, #e5e7eb, #cbd5f5, #e5e7eb);
        margin: 0.8rem 0 1.1rem 0;
    }
    h2 {
        font-size: 30px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# Sidebar Inputs
# =========================
st.sidebar.header("🔧 Enter Order Specifications")

ship_mode = st.sidebar.selectbox("Ship Mode", ["Standard Class", "Second Class", "First Class", "Same Day"])
segment = st.sidebar.selectbox("Segment", ["Consumer", "Corporate", "Home Office"])
region = st.sidebar.selectbox("Region", ["West", "East", "Central", "South"])
category = st.sidebar.selectbox("Category", ["Furniture", "Office Supplies", "Technology"])

sub_categories = {
    "Furniture": ["Bookcases", "Chairs", "Furnishings", "Tables"],
    "Office Supplies": ["Appliances", "Art", "Binders", "Envelopes", "Fasteners", "Labels", "Paper", "Storage", "Supplies"],
    "Technology": ["Accessories", "Copiers", "Machines", "Phones"]
}
sub_category = st.sidebar.selectbox("Sub-Category", sub_categories[category])

quantity = st.sidebar.number_input("Quantity", min_value=1, max_value=20, step=1, value=3)
discount = st.sidebar.slider("Discount", min_value=0.0, max_value=0.8, step=0.05, value=0.0)

predict_clicked = st.sidebar.button("Predict")

# =========================
# Predict & Show Results
# =========================
if predict_clicked:
    input_data = pd.DataFrame({
        "Ship Mode": [ship_mode],
        "Segment": [segment],
        "Region": [region],
        "Category": [category],
        "Sub-Category": [sub_category],
        "Quantity": [quantity],
        "Discount": [discount]
    })

    prediction = float(model.predict(input_data)[0])

    avg_sales = float(df["Sales"].mean())

    X_all = df[["Ship Mode", "Segment", "Region", "Category", "Sub-Category", "Quantity", "Discount"]]
    y_all = df["Sales"]
    y_pred_all = model.predict(X_all)

    r2 = r2_score(y_all, y_pred_all)
    rmse = np.sqrt(mean_squared_error(y_all, y_pred_all))
    mae = mean_absolute_error(y_all, y_pred_all)

    st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)

    st.subheader("🔮 Predicted Sales")
    st.metric("Estimated Sales ($)", f"${prediction:.2f}")

    st.write("")
    st.subheader("📊 Sales Comparison")

    comparison_df = pd.DataFrame({
        "Category": ["Dataset Average", "Estimated Sales"],
        "Sales ($)": [avg_sales, prediction]
    })

    # ✅ Reduced size using narrow column
    c1, c2, c3 = st.columns([2, 1, 1])

    with c1:
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.barplot(
            x="Category",
            y="Sales ($)",
            data=comparison_df,
            palette=["#93c5fd", "#fbbf77"],
            ax=ax,
    
        )
        ax.set_xlabel("")
        ax.set_ylabel("Sales ($)")
        ax.set_title("Estimated vs Average", fontsize=9)
        sns.despine()
        st.pyplot(fig, use_container_width=True)

    st.write("---")
    st.subheader("📈 Model Diagnostics")

    dcol1, dcol2 = st.columns(2)

    with dcol1:
        st.markdown("### 📉 Actual vs Predicted")
        fig1, ax1 = plt.subplots(figsize=(4, 3))
        sns.scatterplot(x=y_all, y=y_pred_all, alpha=0.6, ax=ax1)
        ax1.plot([y_all.min(), y_all.max()], [y_all.min(), y_all.max()], "r--", linewidth=1.5)
        ax1.set_xlabel("Actual Sales ($)")
        ax1.set_ylabel("Predicted Sales ($)")
        ax1.set_title(f"R² = {r2:.3f}")
        st.pyplot(fig1)

    with dcol2:
        st.markdown("### 📊 Residuals Distribution")
        residuals = y_all - y_pred_all
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        sns.histplot(residuals, bins=30, kde=True, ax=ax2, color="purple")
        ax2.set_xlabel("Prediction Error ($)")
        ax2.set_ylabel("Count")
        ax2.set_title("Residuals Distribution")
        st.pyplot(fig2)

    st.write("---")
    st.subheader("📌 Model Performance")

    m1, m2, m3 = st.columns(3)
    m1.metric("R² Score", f"{r2:.4f}")
    m2.metric("RMSE ($)", f"{rmse:.2f}")
    m3.metric("MAE ($)", f"{mae:.2f}")

    st.write("---")
    st.subheader("💡 Interpretation & Insights")

    if prediction < avg_sales:
        st.markdown(
        """
        <div style="font-size:20px; ">
        <ul>
            <li>Estimated sales are <b>below</b> the dataset average.</li>
            <li>High discounts or low quantity may reduce sales.</li>
            <li>Consider increasing quantity or reducing discount.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
        )
    else:
        st.markdown(
        """
        <div style="font-size:20px; ">
        <ul>
            <li>Estimated sales are <b>above</b> the dataset average.</li>
            <li>This order configuration looks stronger than typical orders.</li>
            <li>Good for premium shipping or upselling.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
        )

        st.markdown(
        """
        <div style="font-size:20px; margin-top:1rem;">
        <strong>💡 General Sales Optimization Tips:</strong>
        <ul>
            <li>Reduce discounts where possible</li>
            <li>Bundle products to increase quantity</li>
            <li>Focus on Technology & high-value sub-categories</li>
            <li>Use faster shipping for high-margin orders</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
        )

else:
    st.info("👈 Enter the order details in the sidebar and click Predict to see results.")


st.write("---")
st.markdown(
    """
    <div style="font-size:20px; ">
    "Built with ❤️ using **Python, Streamlit, and Scikit-learn**  \nProject by: Ajani"
    </div>
    """,
    unsafe_allow_html=True
    )
