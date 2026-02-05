## 🛒 Superstore Sales Analysis & Prediction

A complete data analytics and machine learning project that analyzes historical
Superstore sales data, predicts future sales using a trained ML model, and
visualizes business insights through an interactive Power BI dashboard.

This project demonstrates the **end-to-end data science workflow** including
data preprocessing, exploratory analysis, predictive modeling, deployment using
Streamlit, and business intelligence reporting with Power BI.

---

## 📌 Project Overview

### ✅ Data Cleaning & Preprocessing
- Removed missing values and handled inconsistencies in sales data
- Converted categorical variables into analysis-ready formats
- Prepared clean datasets for both machine learning and Power BI
- Engineered features relevant to sales prediction

---

### ✅ Exploratory Data Analysis (EDA)
- Sales distribution analysis across categories and sub-categories
- Profit and discount impact analysis
- Regional and shipping mode performance evaluation
- Identification of high-performing and loss-making segments

---

### ✅ Machine Learning – Sales Prediction
- Algorithm: Regression-based sales prediction model
- Model trained on historical Superstore data
- Evaluated using standard regression performance metrics
- Model serialized and reused for deployment

---

### ✅ Deployment (Streamlit Web App)
- Built an interactive web application using Streamlit
- Users can input sales-related parameters and receive real-time predictions
- Integrated trained ML model for instant inference
- Deployed on Streamlit Cloud for public access

🖥️ **Live Sales Prediction App**  
👉 https://superstore-sales-prediction.streamlit.app/

---

### ✅ Business Intelligence – Power BI Dashboard
- Designed an interactive Power BI dashboard for business stakeholders
- Visualized KPIs such as Sales, Profit, Quantity, and Profit Margin
- Created category, sub-category, regional, and discount-based insights
- Included geographic sales analysis using map visualizations

📊 **Live Power BI Dashboard**  
👉 https://app.powerbi.com/groups/me/reports/2ee77fe7-ef4e-4ff8-8935-d297078fe576/8fa52824170ad40d615a?experience=power-bi

📁 Power BI File Location:  
`PowerBI/Superstore_Sales_Performance_Dashboard.pbix`

---

## 📊 Key Features

### 🔹 Sales Prediction Web App
- User-friendly input form
- Real-time sales prediction
- Dataset-based comparison insights
- Clean and intuitive UI

### 🔹 Power BI Dashboard
- KPI cards for overall performance
- Sales & profit analysis by category and sub-category
- Discount impact on profitability
- Regional sales distribution using maps
- Shipping mode contribution analysis

---

## ⚙️ Tech Stack

**Language**
- Python

**Libraries & Tools**
- Pandas, NumPy
- Scikit-learn
- Streamlit
- Power BI
- Matplotlib
- Joblib
- Git & GitHub

---

## 📂 Repository Structure

```

Store-sales-prediction/
│
├── data/                         # Dataset files
├── notebooks/                    # Jupyter notebooks for analysis & modeling
├── models/                       # Trained ML models
├── app.py                        # Streamlit sales prediction app
├── PowerBI/                      # Power BI reports and visuals
│   ├── Superstore_Sales_Performance_Dashboard.pbix
│   └── dashboard_preview.png
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation

````

---

## 🚀 How to Run Locally

Clone the repository:
```bash
git clone https://github.com/<your-username>/Store-sales-prediction.git
cd Store-sales-prediction
````

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

---

## 🎯 Project Objective

To help business stakeholders:

* Understand Superstore sales and profit performance
* Identify key trends and regional strengths
* Predict future sales using machine learning
* Make data-driven decisions using interactive dashboards

---

## 📈 Future Improvements

* Compare multiple ML models (Random Forest, XGBoost)
* Add time-series forecasting for future sales
* Integrate database for storing predictions
* Enhance Streamlit UI with advanced analytics
* Add automated Power BI refresh using cloud data sources

---

## 👤 Author

**Ajaniya Kamalanathan**
Undergraduate | Data Analytics / AI Intern Aspirant

---

## 🙌 Acknowledgements

* Superstore sales dataset for analysis
* Streamlit Cloud for deployment
* Power BI for business intelligence reporting
* Academic guidance and learning resources

```

