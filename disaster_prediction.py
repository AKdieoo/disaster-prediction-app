import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(page_title="Disaster Prediction App", layout="wide")
st.title("🌧️🌋 Disaster Prediction System")

# --- Load Dataset ---
st.subheader("Uploaded Dataset")

uploaded_file = st.file_uploader("Upload disaster_data.csv", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data)

    features = data[['Rainfall', 'SoilMoisture', 'Slope', 'VegetationIndex', 'DistanceToRiver']]
    flood_labels = data['Flood']
    landslide_labels = data['Landslide']

    X_train, X_test, yf_train, yf_test, yl_train, yl_test = train_test_split(
        features, flood_labels, landslide_labels, test_size=0.3, random_state=42
    )

    # --- Train Models ---
    flood_model = RandomForestClassifier(n_estimators=100, random_state=42)
    flood_model.fit(X_train, yf_train)

    landslide_model = RandomForestClassifier(n_estimators=100, random_state=42)
    landslide_model.fit(X_train, yl_train)

    # --- Prediction Samples ---
    st.subheader("Flood Prediction Samples")
    for i in range(min(4, len(X_test))):
        sample = X_test.iloc[i:i+1]
        actual = yf_test.iloc[i]
        pred = flood_model.predict(sample)[0]

        st.markdown(f"**Sample {i+1}**")
        st.write("Features:", sample)

        fig, ax = plt.subplots()
        ax.bar(["Actual", "Predicted"], [actual, pred], color=['green', 'blue'])
        ax.set_ylim(0, 1.1)
        st.pyplot(fig)

    st.subheader("Landslide Prediction Samples")
    for i in range(min(4, len(X_test))):
        sample = X_test.iloc[i:i+1]
        actual = yl_test.iloc[i]
        pred = landslide_model.predict(sample)[0]

        st.markdown(f"**Sample {i+1}**")
        st.write("Features:", sample)

        fig, ax = plt.subplots()
        ax.bar(["Actual", "Predicted"], [actual, pred], color=['purple', 'orange'])
        ax.set_ylim(0, 1.1)
        st.pyplot(fig)

    # --- Feature Importance ---
    st.subheader("Feature Importance")
    importance_flood = flood_model.feature_importances_
    importance_landslide = landslide_model.feature_importances_

    flood_imp_df = pd.DataFrame({"Feature": features.columns, "Importance": importance_flood})
    landslide_imp_df = pd.DataFrame({"Feature": features.columns, "Importance": importance_landslide})

    col1, col2 = st.columns(2)
    with col1:
        st.write("Flood Model Importance")
        fig1 = plt.figure()
        sns.barplot(x="Importance", y="Feature", data=flood_imp_df.sort_values(by="Importance", ascending=False))
        st.pyplot(fig1)

    with col2:
        st.write("Landslide Model Importance")
        fig2 = plt.figure()
        sns.barplot(x="Importance", y="Feature", data=landslide_imp_df.sort_values(by="Importance", ascending=False))
        st.pyplot(fig2)

else:
    st.warning("Please upload a CSV file with the required format.")

