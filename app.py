import streamlit as st
import pandas as pd
import joblib


# Load model

model = joblib.load("promotion_model.pkl")


st.title("Employee Promotion Prediction")


st.write("Enter Employee Details")


employee_id = st.number_input(
    "Employee ID"
)

department = st.number_input(
    "Department"
)

region = st.number_input(
    "Region"
)

education = st.number_input(
    "Education"
)

gender = st.number_input(
    "Gender"
)

recruitment_channel = st.number_input(
    "Recruitment Channel"
)

no_of_trainings = st.number_input(
    "Number of Trainings"
)

age = st.number_input(
    "Age"
)

previous_year_rating = st.number_input(
    "Previous Year Rating"
)

length_of_service = st.number_input(
    "Length of Service"
)


awards_won = st.number_input(
    "Awards Won"
)

avg_training_score = st.number_input(
    "Average Training Score"
)



if st.button("Predict"):


    data = pd.DataFrame(
        [[
        employee_id,
        department,
        region,
        education,
        gender,
        recruitment_channel,
        no_of_trainings,
        age,
        previous_year_rating,
        length_of_service,
        awards_won,
        avg_training_score
        ]],

        columns=[
        "employee_id",
        "department",
        "region",
        "education",
        "gender",
        "recruitment_channel",
        "no_of_trainings",
        "age",
        "previous_year_rating",
        "length_of_service",
        "awards_won?",
        "avg_training_score"
        ]
    )


    prediction = model.predict(data)


    if prediction[0] == 1:

        st.success(
            "Employee will be promoted 🎉"
        )

    else:

        st.error(
            "Employee will not be promoted"
        )
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <h4>Employee Promotion Prediction</h4>
        <p>Developed by <b>Abhay Mahajan</b></p>
    </div>
    """,
    unsafe_allow_html=True
)
with st.expander("ℹ️ About Project"):
    st.write("""
    **Project:** Employee Promotion Prediction

    **Model:** Gradient Boosting Classifier

    **Dataset:** HR Analytics - Employee Promotion Prediction Dataset

    **Description:**
    This application predicts whether an employee is likely to be promoted based on
    factors such as department, education, age, previous year rating, length of
    service, awards won, and average training score.

    **Developed by:** Abhay Mahajan
    """)