import streamlit as st
import pandas as pd
import plotly.express as px
import math
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def ml_model():
    df = pd.read_excel('Final Project DS.xlsx')
    df = df.drop(columns=['gender'])

   # 1. Menampilkan informasi tentang dataset
    st.title('**Machine Learning Model ini untuk Prediksi Pasien Prioritas menggunakan Clusterisasi**')
    st.write('Dataset yang digunakan memiliki {} baris dan {} kolom.'.format(df.shape[0], df.shape[1]))
    st.write('**Berikut adalah beberapa baris pertama dari dataset:**')
    st.dataframe(df.head())

    # 2. Memisahkan fitur numerik dan kategorikal
    numbers = df.select_dtypes(include = ['number']).columns
    categories = df.select_dtypes(exclude = ['number']).columns

    # 3. Pemilihan Feature, Scaling, dan Encoding
    df_select = df[numbers]
    st.write('**Fitur Numerik yang Digunakan untuk PCA dan KMeans:**')  
    st.dataframe(df_select.head())

    # 4. Standarisasi
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_select)

    # 5. Pemodelan menggunakan PCA
    pca = PCA(n_components=2)
    pca.fit(X_scaled)
    X_pca = pca.fit_transform(X_scaled)

    pcs = pca.transform(X_scaled)

    df_pca = pd.DataFrame(data = pcs, columns = [ 'PC 1', 'PC 2'])
    st.header('**1. PCA**')
    st.write('**Hasil reduksi dimensi menggunakan PCA:**')
    st.dataframe(df_pca.head())
    
    # Deskripsi Hasil PC1
    st.write('Dari hasil PCA, Didapatkan dua komponen utama (PC 1 dan PC 2) yang\
            merupakan kombinasi linier dari fitur-fitur asli dalam dataset.\
            **Pada PC 1 didapat Nilai PC 1 tinggi (Positif besar) yaitu 2.6919, 2.8119, 3.7369**. Menujukkan Observasi\
            memiliki karakteristik dominan pada fitur-fitur utama (misalnya: usia lebih tua,\
            tekanan darah tinggi, BMI tinggi, kolesterol tinggi, dsb.) **PC1 sering diinterpretasikan\
            sebagai indeks tingkat risiko atau tingkat keparahan umum pasien.**')
    # Deskripsi Hasil PC2
    st.write('Sedangkan pada PC 2, Didapat **Nilai PC 2 Positif yaitu : 0.4092, 1.1009**.\
            Menunjukkan Pasien memiliki pola karakteristik tertentu yang berbeda arah dari PC1\
            (misalnya: metabolik vs kardiovaskular, gaya hidup vs faktor genetik).\
            Dan PC 2 ini juga memiliki Nilai Negatif yaitu :-0.2256, -0.7572, -0.8585.\
            Menunjukkan Pasien memiliki pola kebalikan dari kelompok PC2 positif.\
            **PC2 berfungsi membedakan pasien yang mungkin sama-sama “berisiko”,\
            tetapi dengan karakteristik yang berbeda.**')

    # Menampilkan explained_variance ratio
    st.header('**2. Explained Variance Ratio**')
    st.write('Explained Variance Ratio adalah menunjukkan proporsi variansi (informasi)\
              data asli yang mampu dijelaskan oleh setiap Principal Component (PC).')
    st.write('Explained Variance Ratio untuk PC 1: {:.2f}%'.format(pca.explained_variance_ratio_[0] * 100))
    st.write('Explained Variance Ratio untuk PC 2: {:.2f}%'.format(pca.explained_variance_ratio_[1] * 100))
    st.write('Total Explained Variance Ratio oleh kedua komponen utama: {:.2f}%'.format(\
        (pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1]) * 100))
    st.write('Dari hasil di atas, Didapatkan bahwa nilai keduanya relatif rendah, artinya\
             Data memiliki dimensi tinggi dan informasi tersebar di banyak variabel.')
    
    # Visualisasi PCA
    st.write('**Visualisasi PCA**')
    pca_full = PCA()
    pca_full.fit(X_scaled)
    expl_var = pca_full.explained_variance_ratio_
    fig_pca = px.line(
        x=[f'PC {i+1}' for i in range(len(expl_var))],
        y=expl_var,
        labels={'x': 'Principal Components', 'y': 'Explained Variance Ratio'},
        title='Explained Variance Ratio by Principal Components'
    )
    st.plotly_chart(fig_pca)

    # Tuning Hyperparameter KMeans
    st.header('**3. Tuning Hyperparameter KMeans**')
    st.write('Tuning hyperparameter KMeans untuk menentukan jumlah cluster (k) yang optimal\
            berdasarkan silhouette score. Silhouette score mengukur seberapa baik setiap titik data\
            cocok dengan cluster-nya sendiri dibandingkan dengan cluster tetangga terdekat.\
            Nilai silhouette score berkisar dari -1 hingga 1, di mana nilai yang lebih tinggi menunjukkan\
            bahwa titik data lebih cocok dengan cluster-nya sendiri. **Dan Cluster yang cocok pada dataset\
             ini adalah k = 3 dengan silhouette score 0.24**.')
    k_values = []
    silhouette_values = []

    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, n_init=10, random_state = 42)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)

        k_values.append(k)
        silhouette_values.append(score)

    #Membuat tabel hyperparameter
    st.write('**Tabel Hyperparameter KMeans**')
    hyperparameter_df = pd.DataFrame({
        'Jumlah Cluster (k)': k_values,
        'Silhouette Score': silhouette_values
    })
    st.dataframe(hyperparameter_df)

    # Pemodelan 
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    # Visualisasi Clustering
    st.header('**4. Visualisasi Clustering dengan KMeans**')
    fig_cluster = px.scatter(
        df_pca, x='PC 1', y='PC 2',
        color=df['Cluster'].astype(str),
        title='Hasil Clustering KMeans pada Data Pasien Prioritas',
        labels={'color': 'Cluster'}
    )
    st.plotly_chart(fig_cluster)

     # Menampilkan interpretasi cluster
    st.write('**Interpretasi Cluster**')
    st.write('**Cluster 0:** Pasien dengan karakteristik risiko sedang hingga tinggi,\
            mungkin memerlukan pemantauan lebih lanjut dan intervensi medis.\
            **Cluster 1:** Pasien dengan karakteristik risiko rendah, kemungkinan besar\
            memiliki kondisi kesehatan yang lebih baik dan memerlukan perawatan rutin.\
            **Cluster 2:** Pasien dengan karakteristik risiko sangat tinggi, memerlukan\
            perhatian medis segera dan intervensi intensif untuk mengelola kondisi kesehatan mereka.')

    # Evaluasi Model
    st.header('**5. Evaluasi Model KMeans**')
    numbers = df.select_dtypes(include = ['number']).columns
    st.write('Evaluasi model KMeans dapat dilakukan dengan melihat rata-rata fitur numerik\
             untuk setiap cluster. Hal ini membantu dalam memahami karakteristik masing-masing\
             cluster dan bagaimana mereka berbeda satu sama lain.')
    cluster_means = df.groupby('Cluster')[numbers].mean()
    st.dataframe(cluster_means)

    categories = df.select_dtypes(exclude = ['number']).columns
    df2 = df[categories]
    df2['Cluster'] = kmeans.fit_predict(X_scaled)
    cluster_modes = df2.groupby('Cluster').agg(lambda x: x.mode().iloc[0])
    st.dataframe(cluster_modes)
    
    # Menampilkan banyaknya data pada tiap cluster
    st.write('**Jumlah Data pada Setiap Cluster**')
    cluster_counts = df['Cluster'].value_counts().sort_index()
    st.dataframe(cluster_counts)

    import joblib

    # Simpan model dan fitur kolom agar prediction.py bisa pakai
    joblib.dump(kmeans, 'kmeans_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    # Save the feature names that were used for scaling (numeric features before adding 'Cluster')
    joblib.dump(list(df_select.columns), 'feature_names.pkl')
