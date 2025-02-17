# Import Package
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# Fungsi untuk mengubah array numpy menjadi DataFrame dengan kolom yang sesuai
def to_dataframe(X, columns):
    return pd.DataFrame(X, columns=columns)

# Daftar kolom yang diharapkan setelah transformasi
transformed_columns = ['income', 'age', 'experience', 'current_job_yrs', 'current_house_yrs', 'profession', 'city', 'married/single', 'house_ownership', 'car_ownership']

# FunctionTransformer untuk mengubah array menjadi DataFrame
to_df_transformer = FunctionTransformer(to_dataframe, kw_args={'columns': transformed_columns})

# Load Preprocessor yang telah disimpan
preprocessor = joblib.load('preprocessor.pkl')

# Load best_model yang telah disimpan
best_model = joblib.load('best_model.pkl')

# Final pipeline dengan tambahan langkah untuk mengubah menjadi DataFrame
final_pipeline = Pipeline([
    ('preprocessor', preprocessor),  # Menggunakan preprocessor yang sudah didefinisikan dan difit
    ('to_dataframe', to_df_transformer),  # Mengubah array menjadi DataFrame
    ('classifier', best_model)  # Menggunakan model terbaik yang telah disimpan
])

# Judul Website
st.title('Model Prediksi Resiko Gagal Bayar (Pipeline)')

# Input data untuk prediksi
income = st.number_input('Income')
age = st.number_input('Age')
experience = st.number_input('Experience')
profession = st.text_input('Profession')
city = st.text_input('City')
current_job_yrs = st.number_input('Current Job (years)')
current_house_yrs = st.number_input('Current House (years)')
married_single = st.text_input('Married/Single')
house_ownership = st.text_input('House Ownership')
car_ownership = st.text_input('Car Ownership')

# Tombol untuk memproses input
input_data = pd.DataFrame({
    'income': [income],
    'age': [age],
    'experience': [experience],
    'profession': [profession],
    'city': [city],
    'married/single': [married_single],
    'house_ownership': [house_ownership],
    'car_ownership': [car_ownership],
    'current_job_yrs': [current_job_yrs],
    'current_house_yrs': [current_house_yrs]
})

if st.button('Prediksi Input Data'):
    # Melakukan prediksi menggunakan model yang sudah diload
    prediction = final_pipeline.predict(input_data)
    if prediction[0] == 0:
        st.write('Tidak Gagal Bayar')
    else:
        st.write('Gagal Bayar')

# **Upload File CSV**
uploaded_file = st.file_uploader('Masukkan file CSV', type=['csv'])

# Pastikan file tersimpan di session_state agar tidak hilang setelah klik tombol
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Cek apakah kolom sudah sesuai
    expected_columns = ['income', 'age', 'experience', 'married/single', 'house_ownership', 'car_ownership', 'profession', 'city', 'current_job_yrs', 'current_house_yrs']
    if list(df.columns) != expected_columns:
        st.error(f"Kolom dalam CSV tidak sesuai! Diharapkan: {expected_columns}, tetapi ditemukan: {list(df.columns)}")
    else:
        st.session_state.df = df
        st.write('Upload Sukses!')

# **Tombol Prediksi**
if 'df' in st.session_state:  # Pastikan ada file yang diunggah
    if st.button('Prediksi File'):
        df = st.session_state.df  # Ambil DataFrame dari session_state
        
        # Prediksi menggunakan seluruh DataFrame
        prediction = final_pipeline.predict(df)

        # Tampilkan hasil prediksi
        for i, pred in enumerate(prediction):
            if pred == 0:
                st.write(f'Baris {i+1}: Tidak Gagal Bayar')
            else:
                st.write(f'Baris {i+1}: Gagal Bayar')
