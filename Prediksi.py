import streamlit as st
import pickle
import pandas as pd

# Memuat model yang telah dilatih
@st.cache_data
def load_model():
    with open('HasilModel_air.sav', 'rb') as file:
        model = pickle.load(file)
    return model

# Validasi input terhadap standar internasional
def check_standards(input_data):
    standards = {
        'ph': (6.5, 8.5),
        'Hardness': (0, 300),
        'Solids': (0, 500),
        'Chloramines': (0, 4),
        'Sulfate': (0, 250),
        'Conductivity': (0, 800),
        'Organic_carbon': (0, 5),
        'Trihalomethanes': (0, 80),
        'Turbidity': (0, 1),
    }
    results = {"valid": True, "messages": []}
    for feature, (min_val, max_val) in standards.items():
        if not (min_val <= input_data[feature][0] <= max_val):
            results["valid"] = False
            results["messages"].append(f"{feature} berada di luar batas standar ({min_val} - {max_val}).")
    return results

# Input data pengguna
def user_input_features():
    ph = st.sidebar.slider('pH', 0.0, 14.0, 7.0)
    hardness = st.sidebar.slider('Kekerasan (Hardness)', 0, 500, 150)
    solids = st.sidebar.slider('Zat Padat (Solids)', 0, 1000, 500)
    chloramines = st.sidebar.slider('Kloramin (Chloramines)', 0.0, 10.0, 4.0)
    sulfate = st.sidebar.slider('Sulfat (Sulfate)', 0, 400, 200)
    conductivity = st.sidebar.slider('Konduktivitas (Conductivity)', 0, 1500, 500)
    organic_carbon = st.sidebar.slider('Karbon Organik (Organic Carbon)', 0.0, 10.0, 5.0)
    trihalomethanes = st.sidebar.slider('Trihalometan (Trihalomethanes)', 0.0, 200.0, 80.0)
    turbidity = st.sidebar.slider('Kekeruhan (Turbidity)', 0.0, 5.0, 1.0)

    data = {
        'ph': ph,
        'Hardness': hardness,
        'Solids': solids,
        'Chloramines': chloramines,
        'Sulfate': sulfate,
        'Conductivity': conductivity,
        'Organic_carbon': organic_carbon,
        'Trihalomethanes': trihalomethanes,
        'Turbidity': turbidity
    }
    return pd.DataFrame(data, index=[0])

# Faktor penentu layak minum
def determine_factors(input_data):
    factors = []
    standards = {
        'ph': (6.5, 8.5),
        'Hardness': (0, 300),
        'Solids': (0, 500),
        'Chloramines': (0, 4),
        'Sulfate': (0, 250),
        'Conductivity': (0, 800),
        'Organic_carbon': (0, 5),
        'Trihalomethanes': (0, 80),
        'Turbidity': (0, 1),
    }
    for feature, (min_val, max_val) in standards.items():
        if not (min_val <= input_data[feature][0] <= max_val):
            factors.append(feature)
    return factors

# Streamlit Antarmuka
st.title("Prediksi Kualitas Air")

# Memuat model
model = load_model()

# Input data pengguna
input_df = user_input_features()

# Validasi standar internasional
validation = check_standards(input_df)

st.write("### Parameter Data")
st.write(input_df)

if validation["valid"]:
    st.write("### Semua parameter memenuhi standar internasional!")
    st.write("Air **layak diminum** berdasarkan standar internasional.")
else:
    st.write("### Validasi Standar Kualitas Air:")
    for message in validation["messages"]:
        st.write(f"- {message}")
    st.write("Air **tidak layak diminum** karena parameter di atas tidak memenuhi standar.")

# Menentukan faktor yang berpengaruh
factors = determine_factors(input_df)
if factors:
    st.write("### Faktor yang Mempengaruhi Keputusan Tidak Layak Minum:")
    st.write(", ".join(factors))
else:
    st.write("### Semua parameter sudah optimal.")

# Prediksi menggunakan model
st.write("### Prediksi Berdasarkan Model:")
if not validation["valid"]:
    # Jika validasi standar gagal, otomatis tidak layak
    st.write("Prediksi: Air **tidak layak diminum** karena tidak memenuhi standar internasional.")
    st.write("### Probabilitas Prediksi:")
    st.write("Tidak Layak Minum: 1.00, Layak Minum: 0.00")
else:
    # Jika validasi berhasil, gunakan model untuk prediksi
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    if prediction[0] == 1:
        st.write("Prediksi: Air **layak diminum**.")
    else:
        st.write("Prediksi: Air **tidak layak diminum**.")

    st.write("### Probabilitas Prediksi:")
    st.write(f"Tidak Layak Minum: {prediction_proba[0][0]:.2f}, Layak Minum: {prediction_proba[0][1]:.2f}")
