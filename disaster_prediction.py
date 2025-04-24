import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv('disaster_data.csv')  # Assuming the file is in the same directory as your Python script

# --- Step 1: Select Features and Labels ---
features = data[['Rainfall', 'SoilMoisture', 'Slope', 'VegetationIndex', 'DistanceToRiver']]
flood_labels = data['Flood']  # 0 or 1
landslide_labels = data['Landslide']  # 0 or 1

# --- Step 2: Split Data ---
X_train, X_test, yf_train, yf_test, yl_train, yl_test = train_test_split(
    features, flood_labels, landslide_labels, test_size=0.3, random_state=42
)

# --- Step 3: Train Models ---
flood_model = RandomForestClassifier(n_estimators=100, random_state=42)
flood_model.fit(X_train, yf_train)

landslide_model = RandomForestClassifier(n_estimators=100, random_state=42)
landslide_model.fit(X_train, yl_train)

# --- Step 4: Evaluate Models ---
def evaluate_model(model, X_test, y_test, label):
    predictions = model.predict(X_test)
    st.write(f"--- {label} Prediction Report ---")
    st.write(confusion_matrix(y_test, predictions))
    st.write(classification_report(y_test, predictions))
    
    # Plot confusion matrix
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, predictions), annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f'{label} Confusion Matrix')
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)  # Display the plot in Streamlit

evaluate_model(flood_model, X_test, yf_test, 'Flood')
evaluate_model(landslide_model, X_test, yl_test, 'Landslide')
