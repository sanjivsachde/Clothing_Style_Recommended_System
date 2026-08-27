import streamlit as st
import pandas as pd
import joblib
import os

base_path = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(base_path, "clothing_recommendation_system.pkl"))
feature_columns = joblib.load(os.path.join(base_path, "feature_columns.pkl"))
encoders = joblib.load(os.path.join(base_path, "encoders.pkl"))

st.title("Clothing Style Recommendation System")

st.write("Enter customer details to get clothing recommendations.")

st.subheader("Customer Details")

Fashion_Era = st.number_input(
    "Fashion Era",
    min_value=1900,
    max_value=2030,
    value=2020
)

Age = st.number_input(
    "Age",
    min_value=10,
    max_value=100,
    value=25
)

Gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

Gender = 0 if Gender == "Male" else 1

Height_cm = st.number_input(
    "Height (cm)",
    min_value=100.0,
    max_value=250.0,
    value=170.0
)

Weight_kg = st.number_input(
    "Weight (kg)",
    min_value=30.0,
    max_value=200.0,
    value=70.0
)

BMI = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=23.0
)
Body_Type = st.selectbox(
    "Body Type",
    encoders["Body_Type"].classes_
)

Occasion = st.selectbox(
    "Occasion",
    encoders["Occasion"].classes_
)

Season = st.selectbox(
    "Season",
    encoders["Season"].classes_
)

Budget_INR = st.number_input(
    "Budget (INR)",
    min_value=500,
    max_value=100000,
    value=5000
)

Preferred_Style = st.selectbox(
    "Preferred Style",
    encoders["Preferred_Style"].classes_
)

Favorite_Color = st.selectbox(
    "Favorite Color",
    encoders["Favorite_Color"].classes_
)

Fashion_Trend = st.selectbox(
    "Fashion Trend",
    encoders["Fashion_Trend"].classes_
)


if st.button("Get Recommendation"):

    input_data = pd.DataFrame([{
    "Fashion_Era": Fashion_Era,
    "Age": Age,
    "Gender": Gender,
    "Height_cm": Height_cm,
    "Weight_kg": Weight_kg,
    "BMI": BMI,
    "Body_Type": encoders["Body_Type"].transform([Body_Type])[0],
    "Occasion": encoders["Occasion"].transform([Occasion])[0],
    "Season": encoders["Season"].transform([Season])[0],
    "Budget_INR": Budget_INR,
    "Preferred_Style": encoders["Preferred_Style"].transform([Preferred_Style])[0],
    "Favorite_Color": encoders["Favorite_Color"].transform([Favorite_Color])[0],
    "Fashion_Trend": encoders["Fashion_Trend"].transform([Fashion_Trend])[0],
    }])

    input_data = input_data[feature_columns]

    prediction = model.predict(input_data)

    st.success("Recommendation Generated!")

    st.subheader("Your Recommendations")

    top = encoders["Recommended_Top"].inverse_transform([prediction[0][0]])[0]
    bottom = encoders["Recommended_Bottom"].inverse_transform([prediction[0][1]])[0]
    footwear = encoders["Recommended_Footwear"].inverse_transform([prediction[0][2]])[0]
    dress = encoders["Dress_Combination"].inverse_transform([prediction[0][3]])[0]

    st.write("Recommended Top:", top)
    st.write("Recommended Bottom:", bottom)
    st.write("Recommended Footwear:", footwear)
    st.write("Dress Combination:", dress)


    #  "E:\Python\Project_Machine_Learning"
    #  Streanlit run app.py