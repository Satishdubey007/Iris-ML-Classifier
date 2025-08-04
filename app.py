
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

model = RandomForestClassifier()
model.fit(X, y)




st.markdown("""
    <style>
        body {
            background-color: #f2f2f2;
        }
        .main {
            background: linear-gradient(to right, #f8f9fa, #e0f7fa);
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            max-width: 600px;
            margin: auto;
        }
        h1 {
            color: #00695c;
            text-align: center;
        }
        .stButton>button {
            background-color: #00796b;
            color: white;
            font-weight: bold;
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)


st.set_page_config(page_title="Iris Classifier", layout="centered")
st.title("🌸 Iris Flower Classification")
st.markdown("Enter flower measurements below to predict the species:")


sepal_length = st.number_input("Sepal Length (cm)", step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", step=0.1)
petal_length = st.number_input("Petal Length (cm)", step=0.1)
petal_width = st.number_input("Petal Width (cm)", step=0.1)


if st.button("Predict"):
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(input_data)[0]
    species = iris.target_names[prediction]
    st.success(f"🌼 Predicted Species: **{species}**")
