import streamlit as st
import pandas as pd
import plotly.express as px
import re

def chart():
    df = pd.read_excel('Final Project DS.xlsx')
    pasien_count = df.shape[0]

    # Normalize column names map for robust lookup
    def _norm(s):
        return re.sub(r"[^a-z0-9]", "", str(s).lower())

    cols_map = {_norm(c): c for c in df.columns}

    # Helper: cari nama kolom dari beberapa kandidat (normalized keys)
    def get_col(candidates):
        """Return first matching column name from cols_map.
        candidates: iterable of normalized column-name keys (no spaces, lowercase) or substrings.
        """
        for cand in candidates:
            if cols_map.get(cand):
                return cols_map[cand]
        # fallback: substring match on normalized key
        for k, v in cols_map.items():
            for cand in candidates:
                if cand in k:
                    return v
        return None

    # Ensure an age column exists and create `Detail age` categorization
    age_col = cols_map.get('age')
    if age_col is None:
        st.warning('Kolom usia tidak ditemukan di dataset; beberapa metrik mungkin tidak tersedia.')
        avg_age = float('nan')
    else:
        avg_age = df[age_col].mean()
        def _age_cat(a):
            try:
                a = float(a)
            except Exception:
                return None
            if a <= 16:
                return "Remaja"
            elif a <= 27:
                return "Anak Muda"
            elif a <= 47:
                return "Dewasa"
            elif a <= 55:
                return "Paruh Baya"
            else:
                return "Tua"
        df['Detail age'] = df[age_col].apply(_age_cat)

    # Card Metrics dan Button Filter
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Pasien", f"{pasien_count}")
    with col2:
        st.metric("Rata-rata Usia Pasien", f"{avg_age:.2f} Tahun")

    # Ensure Residence_type exists (normalize possible column names)
    res_col = cols_map.get('residencetype') or cols_map.get('residence_type') or cols_map.get('residencetype')
    if res_col:
        df['Residence_type'] = df[res_col]
    else:
        # leave as-is if already exists, or fill with None
        if 'Residence_type' not in df.columns:
            df['Residence_type'] = None

    # Ensure triage exists; if not but Cluster exists, derive triage from Cluster
    if 'triage' not in df.columns:
        if 'Cluster' in df.columns:
            df['triage'] = df['Cluster'].map({0: 'Medium', 1: 'Low', 2: 'High'})
        else:
            df['triage'] = None

    # Initialize session state untuk filter
    if 'detail_age' not in st.session_state:
        st.session_state.detail_age = None
    if 'triage' not in st.session_state:
        st.session_state.triage = None
    if 'residence_type' not in st.session_state:
        st.session_state.residence_type = None
    
    # Filter Buttons
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.write("Filter Detail Age")
        if st.button("Remaja"):
            st.session_state.detail_age = "Remaja"
        if st.button("Anak Muda"):
            st.session_state.detail_age = "Anak Muda"
        if st.button("Dewasa"):
            st.session_state.detail_age = "Dewasa"
        if st.button("Paruh Baya"):
            st.session_state.detail_age = "Paruh Baya"
        if st.button("Tua"):
            st.session_state.detail_age = "Tua"
    with col2:
        st.write("Filter Triage")
        if st.button("Low"):
            st.session_state.triage = "Low"
        if st.button("Medium"):
            st.session_state.triage = "Medium"
        if st.button("High"):
            st.session_state.triage = "High"
    with col3:
        st.write("Filter Residence Type")
        if st.button("Urban"):
            st.session_state.residence_type = "Urban"
        if st.button("Rural"):
            st.session_state.residence_type = "Rural"
    with col4:
        if st.button("Reset Filters"):
            st.session_state.detail_age = None
            st.session_state.triage = None
            st.session_state.residence_type = None
    
    # Apply Filters
    filtered_df = df.copy()
    if st.session_state.detail_age:
        filtered_df = filtered_df[filtered_df['Detail age'] == st.session_state.detail_age]
    if st.session_state.triage:
        filtered_df = filtered_df[filtered_df['triage'] == st.session_state.triage]
    if st.session_state.residence_type:
        filtered_df = filtered_df[filtered_df['Residence_type'] == st.session_state.residence_type]
    
    st.header('**Data Pasien:**')
    st.dataframe(df.head())

    st.title('**Analisis dan Visualisasi Data Pasien Prioritas**')
    st.write('Analisis dan visualisasi data pasien prioritas berdasarkan hasil pemodelan machine learning.')
    st.write('Berikut adalah beberapa visualisasi yang menggambarkan hasil analisis data pasien prioritas.')

    # Jumlah hypertension
    hyper_stroke_count = filtered_df['hypertension'].value_counts().reset_index()
    hyper_stroke_count.columns = ['hypertension', 'count']
    hyper_stroke_count['hypertension_label'] = hyper_stroke_count['hypertension'].map({0: 'Tidak', 1: 'Hipertensi'})
    fig = px.pie(hyper_stroke_count, names='hypertension_label', values='count', title='Hipertensi pada Pasien')
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns([5,5])
    with col1:
    #Persentase Riwayat Merokok
        smoke_stroke_count = filtered_df['smoking_status'].value_counts(dropna=False).reset_index()
        smoke_stroke_count.columns = ['smoking_status', 'count']
        fig = px.pie(smoke_stroke_count, names='smoking_status', values='count', title='Riwayat Merokok pada Pasien')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
    #Persentase Tempat Tinggal
        residence_stroke_count = filtered_df['Residence_type'].value_counts().reset_index()
        residence_stroke_count.columns = ['Residence_type', 'count']
        fig = px.pie(residence_stroke_count, names='Residence_type', values='count', title='Tipe Tempat Tinggal Pasien')
        st.plotly_chart(fig, use_container_width=True)  
    
    st.subheader("Rata-rata BMI Berdasarkan Status Hipertensi")

    if 'bmi' in filtered_df.columns:
        bmi_hyper = filtered_df.groupby('hypertension')['bmi'].mean().reset_index()
        bmi_hyper['hypertension_label'] = bmi_hyper['hypertension'].map({0: 'Tidak', 1: 'Hipertensi'})
        fig = px.bar(bmi_hyper, x='hypertension_label', y='bmi',
                     title='Rata-rata BMI Berdasarkan Status Hipertensi',
                     labels={'hypertension_label': 'Hipertensi', 'bmi': 'Rata-rata BMI'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('Kolom BMI tidak ditemukan; melewatkan plot BMI vs Hipertensi.')

    st.subheader("Scatter Plot Usia vs Tekanan Darah")

    bp_col = get_col(['bloodpressure','blood_pressure','blood pressure','bloodpress'])
    if age_col and bp_col:
        filtered_df[age_col] = pd.to_numeric(filtered_df[age_col], errors='coerce')
        filtered_df[bp_col] = pd.to_numeric(filtered_df[bp_col], errors='coerce')
        fig = px.scatter(filtered_df, x=age_col, y=bp_col,
                         title='Usia vs Tekanan Darah',
                         labels={age_col: 'Usia', bp_col: 'Tekanan Darah'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('Kolom Usia atau Tekanan Darah tidak tersedia; melewatkan scatter Usia vs Tekanan Darah.')

        st.subheader("Distribusi Kategori Usia")

    # Cari kolom kategori usia yang sudah ada (robust lookup)
    age_cat_col = get_col(['detailage','detail_age','detail age','detailage']) or next((c for c in df.columns if 'detail' in c.lower() and 'age' in c.lower()), None)

    # Jika tidak ada, coba buat dari kolom usia jika tersedia
    if age_cat_col is None and age_col:
        def _age_cat_simple(a):
            try:
                a = float(a)
            except Exception:
                return 'Unknown'
            if a <= 16:
                return 'Remaja'
            elif a <= 27:
                return 'Anak Muda'
            elif a <= 47:
                return 'Dewasa'
            elif a <= 55:
                return 'Paruh Baya'
            else:
                return 'Tua'
        filtered_df['Detail age'] = filtered_df[age_col].apply(_age_cat_simple)
        age_cat_col = 'Detail age'

    if age_cat_col and age_cat_col in filtered_df.columns:
        counts = filtered_df[age_cat_col].value_counts(dropna=False).reset_index()
        counts.columns = ['kategori','count']
        fig = px.bar(counts, x='kategori', y='count', title='Distribusi Kategori Usia', labels={'kategori':'Kategori Usia','count':'Jumlah Pasien'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('Tidak ada kategori usia untuk ditampilkan; pastikan dataset memiliki kolom usia atau kolom kategori usia.')
