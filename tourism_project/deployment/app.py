import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# TODO: replace <your-hf-username> below
# Download and load the trained model from the Hugging Face model hub
model_path = hf_hub_download(
    repo_id="Kman42/wellness-tourism-model",
    filename="best_tourism_model_v1.joblib"
)
model = joblib.load(model_path)

# Streamlit UI
st.title("Wellness Tourism Package Purchase Prediction")
st.write("""
This application predicts whether a customer is likely to **purchase the Wellness
Tourism Package**, based on their profile and interaction details.
Enter the customer details below to get a prediction.
""")

# ----- Categorical inputs -----
typeofcontact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
gender = st.selectbox("Gender", ["Male", "Female"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])

# ----- Numeric inputs -----
age = st.number_input("Age", min_value=18, max_value=100, value=35)
city_tier = st.selectbox("City Tier", [1, 2, 3])
duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=200, value=15)
num_persons = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
num_followups = st.number_input("Number of Followups", min_value=0, max_value=10, value=3)
preferred_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
num_trips = st.number_input("Number of Trips (per year)", min_value=0, max_value=50, value=2)
passport = st.selectbox("Has Passport", [0, 1])
pitch_satisfaction = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
own_car = st.selectbox("Owns a Car", [0, 1])
num_children = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)
monthly_income = st.number_input("Monthly Income", min_value=0, max_value=1000000, value=20000, step=1000)

# Assemble inputs into a single-row DataFrame (column names must match training data)
input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": typeofcontact,
    "CityTier": city_tier,
    "DurationOfPitch": duration_of_pitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": num_persons,
    "NumberOfFollowups": num_followups,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": preferred_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": num_trips,
    "Passport": passport,
    "PitchSatisfactionScore": pitch_satisfaction,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": num_children,
    "Designation": designation,
    "MonthlyIncome": monthly_income,
}])

if st.button("Predict Purchase"):
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]
    st.subheader("Prediction Result:")
    if int(prediction) == 1:
        st.success(f"Likely to PURCHASE the package (probability: {proba:.2%})")
    else:
        st.warning(f"Unlikely to purchase the package (probability: {proba:.2%})")
