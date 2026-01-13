import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import re

def prediction_app():
    st.title("Prediksi Pasien Prioritas")
    st.write("Masukkan data pasien untuk memprediksi Prioritas Pasien.")
    
    # 1. Load Model & Metadata
    kmeans = joblib.load('kmeans_model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_names = joblib.load('feature_names.pkl')
   
    # 2. Form Input Pengguna
    st.write("### 📝 Masukkan Data Pasien")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=17)
    with col2:
        hypertension = st.selectbox("Hypertension", options=[0, 1], index=0)
    with col3:
        heart_disease = st.selectbox("Heart Disease", options=[0, 1], index=0)
    with col4:
        cholesterol = st.number_input("Cholesterol", min_value=0, max_value=400, value=200)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        Residence_type = st.selectbox("Residence Type", options=['Urban', 'Rural'], index=0)
    with col2:
        insulin = st.number_input("Insulin", min_value=0, value=100)
    with col3:
        bmi = st.number_input("BMI", 0.0, 60.0, 25.0)
    with col4:
        plasma_glucose = st.number_input("Plasma Glucose", min_value=0, value=120)
    with col5:
        smoking_status = st.selectbox("Smoking Status", options=['never smoked', 'formerly smoked', 'smokes'], index=0) 
    
    # Form Input Pengguna
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        chest_pain_type = st.selectbox("Chest Pain Type", options=[0, 1], index=0)
    with col2:
        max_heart_rate = st.number_input("Max Heart Rate", min_value=0, value=200)
    with col3:
        exercise_angine = st.selectbox("Exercise Angine", options=[0, 1], index=0)
    with col4:
        skin_thickness = st.number_input("Skin Thickness", min_value=0, value=100)
    with col5:
        blood_pressure = st.number_input("Blood Pressure", min_value=0, value=200)
        
    
    # 3. Kategori Otomatis
    col1, col2 = st.columns([5,5])
    # Age Category
    with col1: 
        if age <= 16:
            Detail_age = "Remaja"
            color = "#4DA6FF"
        elif age <= 27:
            Detail_age = "Anak Muda"
            color = "#4CAF50"
        elif age <= 47:
            Detail_age = "Dewasa"
            color = "#EBD300"
        elif age <= 55:
            Detail_age = "Paruh Baya"
            color = "#F44336"
        else:
            Detail_age = "Tua"
            color = "#8C007E"
        st.markdown(
                f"""
                <div style="background:{color}; padding:10px; border-radius:10px; text-align:center; color:#fff;">
                    <div style="font-size:13px;">Age Category</div>
                    <div style="font-size:16px;font-weight:bold;">{Detail_age}</div>
                </div>
                """,
                unsafe_allow_html=True
        )
    
    #BMI Category
    with col2:
        if bmi < 18.5:
            bmi_category = "Underweight"
            color = "#4DA6FF"
        elif bmi < 24.9:
            bmi_category = "Normal Weight"
            color = "#4CAF50"
        elif bmi < 29.9:
            bmi_category = "Overweight"
            color = "#EBD300"
        else:
            bmi_category = "Obesity"
            color = "#F44336"
        st.markdown(
                f"""
                <div style="background:{color}; padding:10px; border-radius:10px; text-align:center; color:#fff;">
                    <div style="font-size:13px;">BMI Category</div>
                    <div style="font-size:16px;font-weight:bold;">{bmi_category}</div>
                </div>
                """,
                unsafe_allow_html=True
        )

    st.write("")

    col3, col4 = st.columns(2)
    #Glucose Category
    with col3:
        if plasma_glucose < 70:
            glucose_category = "Low"
            color = "#4DA6FF"
        elif plasma_glucose < 140:
            glucose_category = "Normal"
            color = "#4CAF50"
        elif plasma_glucose < 200:
            glucose_category = "Prediabetes"
            color = "#EBD300"
        else:
            glucose_category = "Diabetes"
            color = "#F44336"
        st.markdown(
                f"""
                <div style="background:{color}; padding:10px; border-radius:10px; text-align:center; color:#fff;">
                    <div style="font-size:13px;">Glucose Category</div>
                    <div style="font-size:16px;font-weight:bold;">{glucose_category}</div>
                </div>
                """,
                unsafe_allow_html=True
        )
    #Blood Pressure Category
    with col4:
        if blood_pressure < 80:
            bp_category = "Low"
            color = "#4DA6FF"
        elif blood_pressure < 120:
            bp_category = "Normal"
            color = "#4CAF50"
        elif blood_pressure < 130:
            bp_category = "Elevated"
            color = "#EBD300"
        elif blood_pressure < 140:
            bp_category = "Hypertension Stage 1"
            color = "#F44336"
        else:
            bp_category = "Hypertension Stage 2"
            color = "#8C007E"
        st.markdown(
            f"""
            <div style="background:{color}; padding:10px; border-radius:10px; text-align:center; color:#fff;">
                <div style="font-size:13px;">Blood Pressure Category</div>
                <div style="font-size:16px;font-weight:bold;">{bp_category}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.write("")

    # 4. Masukkan Data ke DataFrame
    input_data = pd.DataFrame({
        'Age': [age],
        'Hypertension': [hypertension],
        'Heart_Disease': [heart_disease],
        'Cholesterol': [cholesterol],
        'Blood_Pressure': [blood_pressure],
        'Residence_type': [Residence_type],
        'Insulin': [insulin],
        'BMI': [bmi],
        'Plasma_Glucose': [plasma_glucose],
        'Smoking_Status': [smoking_status]
    })

    # Lakukan Prediksi
    # Map/align input columns to the feature names used during training.
    def _norm(s):
        return re.sub(r"[^a-z0-9]", "", str(s).lower())

    feat_list = list(feature_names)
    input_norm_map = {_norm(c): c for c in input_data.columns}
    aligned_row = {}
    missing = []
    for f in feat_list:
        nf = _norm(f)
        if nf in input_norm_map:
            aligned_row[f] = input_data.loc[0, input_norm_map[nf]]
        else:
            missing.append(f)
            aligned_row[f] = 0

    if missing:
        st.warning("Beberapa fitur tidak ditemukan dan akan diisi 0: " + ", ".join(missing))

    input_aligned = pd.DataFrame([aligned_row], columns=feat_list)

    input_data_scaled = scaler.transform(input_aligned)
    cluster = kmeans.predict(input_data_scaled)[0]

    # Tampilkan hasil prediksi
    priority_mapping = {
        0: "Prioritas Sedang",
        1: "Prioritas Rendah",
        2: "Prioritas Tinggi"
    }
    priority_result = priority_mapping.get(cluster, "Tidak Diketahui")
    st.write("### 🏷️ Hasil Prediksi Prioritas Pasien")
    st.markdown(
        f"""
        <div style="background:#2196F3; padding:15px; border-radius:10px; text-align:center; color:#fff;">
            <div style="font-size:20px;font-weight:bold;">{priority_result}</div>
        </div>
        """,
        unsafe_allow_html=True
    ) 

    

# =========================
# 6. Panggil Fungsi Jika File Ini Dibuka
# =========================
if __name__ == "__main__":
    prediction_app()