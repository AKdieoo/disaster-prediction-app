import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

st.title("Disaster Prediction App")

# List of dataset files
dataset_files = [
    'data1.csv',
    'data2.csv',
    'data3.csv',
    'data4.csv'
]

def evaluate_and_plot(X, y, model, label_title):
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred, output_dict=True)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f'{label_title} Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    return fig

for i, file in enumerate(dataset_files):
    st.header(f"Dataset {i+1}")
    data = pd.read_csv(file)

    features = data[['Rainfall', 'SoilMoisture', 'Slope', 'VegetationIndex', 'DistanceToRiver']]
    flood_labels = data['Flood']
    landslide_labels = data['Landslide']

    X_train, X_test, yf_train, yf_test, yl_train, yl_test = train_test_split(
        features, flood_labels, landslide_labels, test_size=0.3, random_state=42
    )

    # Train flood model
    flood_model = RandomForestClassifier(n_estimators=100, random_state=42)
    flood_model.fit(X_train, yf_train)
    flood_fig = evaluate_and_plot(X_test, yf_test, flood_model, f'Dataset {i+1} - Flood')
    st.pyplot(flood_fig)

    # Train landslide model
    landslide_model = RandomForestClassifier(n_estimators=100, random_state=42)
    landslide_model.fit(X_train, yl_train)
    landslide_fig = evaluate_and_plot(X_test, yl_test, landslide_model, f'Dataset {i+1} - Landslide')
    st.pyplot(landslide_fig)


