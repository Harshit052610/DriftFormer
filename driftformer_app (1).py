import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import warnings
from streamlit_lottie import st_lottie
import requests
import os
import threading
import io
warnings.filterwarnings("ignore")

# Load Lottie animation
@st.cache_data
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code == 200:
        return r.json()
    return None

lottie_bg = load_lottieurl("https://assets9.lottiefiles.com/private_files/lf30_jcikwtux.json")
st_lottie(lottie_bg, height=200, key="background")

st.set_page_config(page_title="DriftFormer", layout="wide")
st.title("DriftFormer: Real-Time Concept Drift Detection")

uploaded_file = st.file_uploader("Upload Cleaned Airline Data (.csv)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("1. Data Preprocessing")
    categorical_cols = ['Airline', 'Departure_Airport', 'Arrival_Airport', 'Flight_Status',
                        'Gender', 'Income_Level', 'Travel_Purpose', 'Seat_Class',
                        'Frequent_Flyer_Status', 'Check_in_Method', 'Seat_Selected']
    for col in categorical_cols:
        df[col] = df[col].str.lower()

    df = df[~((df['Departure_Airport'] == df['Arrival_Airport']) & (df['Flight_Duration_Minutes'] > 0))]
    median_distance = df['Distance_Miles'].median()
    df.loc[df['Distance_Miles'] < 100, 'Distance_Miles'] = median_distance

    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    numerical_cols = ['Flight_Duration_Minutes', 'Distance_Miles', 'Price_USD', 'Age',
                      'Bags_Checked', 'Flight_Satisfaction_Score', 'Delay_Minutes',
                      'Booking_Days_In_Advance', 'Departure_Hour', 'Departure_Day',
                      'Departure_Weekday', 'Departure_Month']
    scaler = StandardScaler()
    df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])

    df_encoded['timestamp'] = df['Departure_Month'] * 10000 + df['Departure_Day'] * 100 + df['Departure_Hour']
    df_encoded = df_encoded.sort_values('timestamp')
    batches = np.array_split(df_encoded, 10)

    for i in range(5, len(batches)):
        batches[i]['Price_USD'] = np.random.permutation(batches[i]['Price_USD'].values)

    X = df_encoded.drop(['Flight_Status_delayed', 'Flight_Status_on-time'], axis=1, errors='ignore')
    y = df['Flight_Status'].map({'on-time': 0, 'delayed': 1, 'cancelled': 2})
    X_train = X[:int(0.8*len(X))]
    y_train = y[:int(0.8*len(y))]

    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    input_dim = X.shape[1]
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = Dense(32, activation='relu')(encoded)
    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(input_dim, activation='linear')(decoded)
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(X_train, X_train, epochs=5, batch_size=32, verbose=0)

    st.subheader("2. Drift and Trust Score Analysis")
    results = []
    label_encoder = LabelEncoder()
    label_encoder.fit(['on-time', 'delayed', 'cancelled'])

    def calculate_psi(expected, actual, bins=10):
        bins = np.histogram_bin_edges(np.concatenate([expected, actual]), bins=bins)
        expected_hist, _ = np.histogram(expected, bins=bins, density=True)
        actual_hist, _ = np.histogram(actual, bins=bins, density=True)
        expected_hist = np.where(expected_hist == 0, 1e-10, expected_hist)
        actual_hist = np.where(actual_hist == 0, 1e-10, actual_hist)
        psi = np.sum((expected_hist - actual_hist) * np.log(expected_hist / actual_hist))
        return psi

    for i, batch in enumerate(batches):
        X_batch = batch.drop(['Flight_Status_delayed', 'Flight_Status_on-time'], axis=1, errors='ignore')
        y_batch_labels = df.loc[batch.index, 'Flight_Status']
        y_batch = label_encoder.transform(y_batch_labels)
        y_pred = clf.predict(X_batch)
        y_proba = clf.predict_proba(X_batch)
        accuracy = accuracy_score(y_batch, y_pred)

        psi_score = calculate_psi(X_train['Price_USD'], X_batch['Price_USD'])
        ae_error = np.mean((autoencoder.predict(X_batch, verbose=0) - X_batch)**2)
        uncertainty = np.mean(np.max(y_proba, axis=1) - np.sort(y_proba, axis=1)[:, -2])

        results.append({
            'Batch': i+1,
            'PSI': psi_score,
            'AE_Error': ae_error,
            'Uncertainty': uncertainty,
            'Error_Rate': 1 - accuracy
        })

    results_df = pd.DataFrame(results)

    st.sidebar.header("Drift Metrics Summary")
    st.sidebar.metric("PSI (Batch 10)", f"{results_df.PSI.iloc[-1]:.3f}")
    st.sidebar.metric("AE Error", f"{results_df.AE_Error.iloc[-1]:.3f}")
    st.sidebar.metric("Uncertainty", f"{results_df.Uncertainty.iloc[-1]:.3f}")
    st.sidebar.metric("Error Rate", f"{results_df.Error_Rate.iloc[-1]*100:.2f}%")

    col1, col2 = st.columns(2)
    with col1:
        st.line_chart(results_df.set_index("Batch")[['PSI', 'AE_Error']])
    with col2:
        st.bar_chart(results_df.set_index("Batch")[['Uncertainty', 'Error_Rate']])

    with st.expander("📋 Dataset Overview"):
        st.write(df.describe())
        st.table(df[categorical_cols].nunique().to_frame("Unique Values"))

    st.markdown("""
    **Interpretation Guide:**

    - **PSI** indicates shift in pricing patterns across time.
    - **AE Error** reveals anomaly in pattern reconstruction.
    - **Uncertainty** quantifies model confidence drop.
    - **Error Rate** shows predictive failure spikes.

    Consistent rise in all → Trigger drift alert.
    """)

    st.success("✅ Drift and Trust Score Analysis Completed")

else:
    st.info("👆 Please upload your dataset to begin.")
