import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="IDS Dashboard", page_icon="🛡️", layout="wide")

# ---------------- CUSTOM UI ----------------
st.markdown("""
<style>
body {background-color: #0e1117;}
.main-title {text-align:center; color:#00FFAA; font-size:42px;}
.sub-text {text-align:center; color:#CCCCCC;}
.card {
    padding:20px;
    border-radius:15px;
    background-color:#1c1f26;
    box-shadow: 0 4px 10px rgba(0,0,0,0.4);
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown('<div class="main-title">🛡️ Intrusion Detection Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">AI-based Network Threat Detection & Analysis</div>', unsafe_allow_html=True)
st.markdown("---")

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Controls")
mode = st.sidebar.radio("Choose Input:", ["Upload Dataset", "Use Sample Dataset"])

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl", "rb"))

# ---------------- LOAD DATA ----------------
# ---------------- LOAD DATA ----------------
data = None

if mode == "Upload Dataset":
    uploaded = st.file_uploader("📂 Upload CSV", type=["csv"])
    if uploaded:
        data = pd.read_csv(uploaded)

else:
    full_data = pd.read_csv("final_dataset.csv")
    data = full_data.sample(n=100, random_state=42)   

    st.info(f"Using sample dataset (100 rows out of {full_data.shape[0]})")

# ---------------- MAIN ----------------
if data is not None:

    st.subheader("📊 Dataset Preview")
    st.dataframe(data.head(), use_container_width=True)

    # Timestamp
    if 'Timestamp' in data.columns:
        data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce')

    # ---------------- PREPROCESS ----------------
    data_numeric = data.select_dtypes(include=[np.number])

    if 'Label' in data_numeric.columns:
        data_numeric = data_numeric.drop(columns=['Label'])

    data_numeric = data_numeric.replace([np.inf, -np.inf], np.nan)
    data_numeric = data_numeric.fillna(0)

    # Feature alignment
    EXPECTED = model.n_features_in_

    if data_numeric.shape[1] > EXPECTED:
        data_numeric = data_numeric.iloc[:, :EXPECTED]
    elif data_numeric.shape[1] < EXPECTED:
        for i in range(EXPECTED - data_numeric.shape[1]):
            data_numeric[f"missing_{i}"] = 0

    st.sidebar.success(f"✅ Features: {data_numeric.shape[1]}")

    # ---------------- RUN ----------------
    if st.button("🚀 Run Detection"):

        pred = model.predict(data_numeric.values)
        data['Prediction'] = pred

        # -------- Attack Mapping --------
        attack_map = {
            0: "Normal",
            1: "DoS",
            2: "DDoS",
            3: "Brute Force",
            4: "Port Scan"
        }

        data['Attack_Type'] = data['Prediction'].map(attack_map).fillna("Other")

        st.success("✅ Detection Completed")

        # ---------------- SUMMARY CARDS ----------------
        total = len(data)
        attacks = len(data[data['Prediction'] != 0])
        normal = total - attacks

        col1, col2, col3 = st.columns(3)

        col1.metric("📦 Total Records", total)
        col2.metric("🟢 Normal Traffic", normal)
        col3.metric("🔴 Intrusions", attacks)

        st.markdown("---")

        # ---------------- FORCE ALL TYPES ----------------
        # -------- FORCE ALL ATTACK TYPES --------
        all_classes = ["Normal", "DoS"]

        counts = data['Attack_Type'].value_counts()

        for cls in all_classes:
            if cls not in counts:
                counts[cls] = 0
                counts = counts[all_classes]

        # ---------------- VISUALS ----------------
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Attack Count")
            st.bar_chart(counts)

        with col2:
            fig = px.pie(
                names=counts.index,
                values=counts.values,
                title="Attack Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

        # ---------------- TIME ANALYSIS ----------------
        if 'Timestamp' in data.columns:

            st.subheader("⏳ Intrusions Over Time")

            intrusions = data[data['Prediction'] != 0]

            if not intrusions.empty:

                intrusions['Hour'] = intrusions['Timestamp'].dt.hour
                time_counts = intrusions.groupby('Hour').size()

                fig_time = px.line(
                    x=time_counts.index,
                    y=time_counts.values,
                    labels={'x': 'Hour', 'y': 'Intrusions'},
                    title="Intrusions per Hour"
                )

                st.plotly_chart(fig_time, use_container_width=True)

        # ---------------- SAMPLE RESULTS ----------------
        st.subheader("📋 Sample Predictions")
        st.dataframe(data[['Attack_Type']].head(20), use_container_width=True)

        # ---------------- FINAL STATUS ----------------
        if attacks == 0:
            st.success("🟢 No Intrusion Detected")
        else:
            st.error("🔴 Intrusion Detected!")

        # ---------------- DOWNLOAD ----------------
        csv = data.to_csv(index=False).encode('utf-8')

        st.download_button(
            "⬇️ Download Results",
            csv,
            "results.csv",
            "text/csv"
        )