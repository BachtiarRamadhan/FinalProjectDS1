import streamlit as st

def about_dataset():
    st.write('**Tentang Dataset Pasien Prioritas**')
    col1, col2= st.columns([5,5])

    with col1:
        link = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSWFeY54-gRjiRiCXIJNdCUF6qZ6x4Gz2-sOQ&s"
        st.image(link, caption="Pasien Prioritas")

    with col2:
        st.write('Dataset ini berisi informasi mengenai pasien prioritas,\
                Data ini juga mencakup berbagai atribut\
                seperti usia, kondisi medis, dan tingkat keparahan penyakit, dll.\
                Dataset ini digunakan untuk menganalisis dan memprediksi kebutuhan perawatan\
                medis bagi pasien prioritas, serta membantu dalam pengambilan keputusan\
                di rumah sakit untuk meningkatkan pelayanan kesehatan.')