import streamlit as st
import pandas as pd
from predict import predict_score


# --- Placeholder Score Calculation Function ---
# In a real application, you would load your trained machine learning model
# (e.g., using pickle, joblib, or torch) and use this function to preprocess
# the user inputs and predict the score.



# --- Streamlit App Configuration ---
st.set_page_config(
    page_title="Credit Risk Score Calculator",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("Credit Risk Score Prediction App")

# --- Define Feature Layout and Widgets ---

# Use a container to keep the inputs visually grouped
with st.container(border=True):

    # 1. Row 1 (Col 1, Col 2, Col 3)
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider(
            'Age',
            min_value=0, max_value=100, value=30, step=1,
            help="Age of the applicant in years."
        )
    with col2:
        income = st.number_input(
            'Income',
            min_value=0, max_value=1200000, value=75000, step=1000,
            help="Annual income of the applicant.",
            format="%d"
        )
    with col3:
        loan_amount = st.number_input(
            'Loan Amount',
            min_value=0, max_value=2600000, value=150000, step=5000,
            help="Total amount of the loan requested.",
            format="%d"
        )

    # 2. Row 2
    col4, col5, col6 = st.columns(3)
    with col4:
        loan_tenure_months = st.number_input(
            'Loan Tenure (Months)',
            min_value=1, value=60, step=12,
            help="Duration of the loan in months.",
            format="%d"
        )
    with col5:
        number_of_open_accounts = st.number_input(
            'Number of Open Accounts',
            min_value=0, value=4, step=1,
            help="Total number of active credit accounts.",
            format="%d"
        )
    with col6:
        if income==0:
            l_2_i=0
        else:
            l_2_i=loan_amount/income
        st.markdown('loan_amount/income', help="Ratio of total loan amount to annual income.")

        loan_to_income=st.markdown(f""" <div style = 'background-color: gray;'>
        {l_2_i}
                                                 </div>""" ,unsafe_allow_html=True,

        )


    # 3. Row 3
    col7, col8, col9 = st.columns(3)
    with col7:
        credit_utilization_ratio = st.slider(
            'Credit Utilization Ratio (%)',
            min_value=0.0, max_value=100.0, value=25.5, step=0.1,
            help="Percentage of available credit currently being used."
        )
    with col8:
        delinquency_ratio = st.slider(
            'Delinquency Ratio (%)',
            min_value=0.0, max_value=100.0, value=0.5, step=0.1,
            help="Percentage of payments that were missed or late."
        )
    with col9:
        avg_dpd_per_delinquency = st.number_input(
            'Avg DPD per Delinquency',
            min_value=0.0, value=15.0, step=1.0,
            help="Average Days Past Due (DPD) for each instance of delinquency."
        )

    # 4. Row 4 (Categorical Features)
    col10, col11, col12 = st.columns(3)
    with col10:
        residence_type = st.selectbox(
            'Residence Type',
            options=['Owned', 'Mortgage', 'Rented'],
            help="Applicant's residence status."
        )
    with col11:
        # Note: I'm using the unique, sensible options based on your list.
        loan_purpose = st.selectbox(
            'Loan Purpose',
            options=['Home', 'Education', 'Personal', 'Auto'],
            help="The primary reason for applying for the loan."
        )
    with col12:
        loan_type = st.selectbox(
            'Loan Type',
            options=['Secured', 'Unsecured'],
            help="Whether the loan is backed by collateral (Secured) or not (Unsecured)."
        )

# --- Collect all inputs into a dictionary ---
input_features = {
    'age': age,
    'income': income,
    'loan_amount': loan_amount,
    'loan_tenure_months': loan_tenure_months,
    'number_of_open_accounts': number_of_open_accounts,
    'credit_utilization_ratio': credit_utilization_ratio,
    'loan_to_income': loan_to_income,
    'delinquency_ratio': delinquency_ratio,
    'avg_dpd_per_delinquency': avg_dpd_per_delinquency,
    'residence_type': residence_type,
    'loan_purpose': loan_purpose,
    'loan_type': loan_type
}

# --- Calculation Button ---
st.markdown("---")
# Center the button using columns
btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])

with btn_col2:
    calculate_button = st.button(
        'Calculate Score',
        type="primary",
        use_container_width=True
    )

# --- Display Results ---
if calculate_button:
    with st.spinner('Calculating score...'):

        # calculate  probability, score, risk_level
        probability, score, risk_level, color = predict_score(input_features)
        # try:
        #     probability, score, risk_level, color = predict_score(input_features)
        # except Exception as e:
        #     st.error(f"An error occurred during SCORE calculation: {e}")

        if score is not None:
            # Display results in a visually appealing way
            st.markdown("---")
            st.subheader("Prediction Result")

            result_col1, result_col2, result_col3 = st.columns(3)
            st.markdown(
                f"**Probability:** <span style='color:{color}; font-weight:bold;'>{probability:0.0f}%</span>",
                unsafe_allow_html=True)

            with result_col2:
                # Use markdown to style the score with a large font and color

                st.markdown(
                    f"""
                    <div style='text-align: center; padding: 20px; border: 3px solid {color}; border-radius: 10px;'>
                        <h2 style='color: #4a4a4a; margin-bottom: 5px;'>Predicted Score</h2>
                        <h1 style='color: {color}; font-size: 4em; margin-top: 0px;'>{score}</h1>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown(f"**Risk Classification:** <span style='color:{color}; font-weight:bold;'>{risk_level}</span>",
                        unsafe_allow_html=True)

            # Show the raw inputs (optional, good for debugging)
            with st.expander("Show Detailed Input Data"):
                st.dataframe(pd.DataFrame(input_features, index=["Value"]).T)