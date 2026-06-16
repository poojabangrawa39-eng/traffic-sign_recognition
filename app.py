import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Medical Insurance Prediction", page_icon="🏥")

st.title("🏥 Medical Insurance Cost Prediction")
st.write("Predict insurance charges using Machine Learning Regression")

# Load dataset
df = pd.read_csv("insurance.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Missing Values")
st.write(df.isnull().sum())

# Encoding
le_sex = LabelEncoder()
le_smoker = LabelEncoder()
le_region = LabelEncoder()

df['sex'] = le_sex.fit_transform(df['sex'])
df['smoker'] = le_smoker.fit_transform(df['smoker'])
df['region'] = le_region.fit_transform(df['region'])


X = df[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
y = df['charges']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

st.subheader("Model Evaluation")
st.write("MAE:", mean_absolute_error(y_test, y_pred))
st.write("MSE:", mean_squared_error(y_test, y_pred))
st.write("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
st.write("R2 Score:", r2_score(y_test, y_pred))

st.subheader("Predict New Insurance Cost")

age = st.number_input("Enter Age", min_value=1, max_value=100, value=25)
sex = st.selectbox("Select Gender", ["female", "male"])
bmi = st.number_input("Enter BMI", min_value=10.0, max_value=60.0, value=28.5)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=2)
smoker = st.selectbox("Smoker", ["no", "yes"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

if st.button("Predict Insurance Cost"):
    sex_encoded = le_sex.transform([sex])[0]
    smoker_encoded = le_smoker.transform([smoker])[0]
    region_encoded = le_region.transform([region])[0]

    new_person = [[age, sex_encoded, bmi, children, smoker_encoded, region_encoded]]

    prediction = model.predict(new_person)

    st.success(f"Predicted Insurance Cost: ₹{prediction[0]:,.2f}")
    