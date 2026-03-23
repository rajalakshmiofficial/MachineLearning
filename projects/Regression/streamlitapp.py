import streamlit as st
import pandas as pd
import joblib

# Load Model
pipeline = joblib.load("insurance_pipeline.pkl")

# -------------------------------------------------------------
# ✅ Custom CSS for Styling (FIXED)
# -------------------------------------------------------------
page_bg = """
<style>
body {
    background-color: #f7f9fc;
}

.main-header {
    font-size: 40px;
    color: #3f72af;
    font-weight: 700;
    text-align: center;
    margin-bottom: -10px;
}

.sub-header {
    font-size: 18px;
    color: #112d4e;
    text-align: center;
    margin-bottom: 30px;
}

.stButton > button {
    background-color: #3f72af;
    color: white;
    border-radius: 8px;
    padding: 10px 30px;
    font-size: 18px;
    border: none;
}

.result-box {
    background-color: #dbe2ef;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    font-size: 24px;
    color: #112d4e;
    margin-top: 20px;
    border: 2px solid #3f72af;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# -------------------------------------------------------------
# ✅ App Header
# -------------------------------------------------------------
st.markdown('<div class="main-header">💰 Health Insurance Cost Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Enter your details below and get an instant prediction</div>', unsafe_allow_html=True)

# -------------------------------------------------------------
# ✅ Sidebar
# -------------------------------------------------------------
st.sidebar.title("📌 About")
st.sidebar.info(
    "This app uses a **Machine Learning model** to predict health insurance charges "
    "based on personal details such as age, BMI, smoking status, etc."
)

st.sidebar.write("✅ Built with Streamlit, sklearn Pipeline, Linear Regression.")
st.sidebar.write("✅ Fully Automated Preprocessing (Encoding + Scaling).")

# -------------------------------------------------------------
# ✅ Input Layout
# -------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("🎂 Age", min_value=18, max_value=100, value=25, step=1)
    sex = st.selectbox("⚧ Sex", ["male", "female"])
    bmi = st.number_input("⚖️ BMI", min_value=10.0, max_value=60.0, value=26.5)

with col2:
    children = st.number_input("👶 Number of Children", min_value=0, max_value=5, value=0, step=1)
    smoker = st.selectbox("🚬 Smoker", ["yes", "no"])
    region = st.selectbox("🌎 Region", ["northeast", "northwest", "southeast", "southwest"])

# -------------------------------------------------------------
# ✅ Prepare Input Data
# -------------------------------------------------------------
input_df = pd.DataFrame({
    "age": [age],
    "sex": [sex],
    "bmi": [bmi],
    "children": [children],
    "smoker": [smoker],
    "region": [region]
})

# -------------------------------------------------------------
# ✅ Predict
# -------------------------------------------------------------
st.write(" ")
predict_btn = st.button("🔮 Predict Your Insurance Cost")

if predict_btn:
    predicted = pipeline.predict(input_df)[0]

    st.markdown(
        f'''<div class="result-box">
        💵 <b>Estimated Insurance Charge:</b><br>
        <span style="color:#3f72af; font-size:32px;">${predicted:,.2f}</span>
        </div>''',
        unsafe_allow_html=True
    )
