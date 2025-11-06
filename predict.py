import joblib
import pandas as pd
import numpy as np

MODEL_PATH='artifacts/model_data.joblib'
# Load the model and its components
model_data = joblib. load(MODEL_PATH)
model = model_data [ 'model']
scaler = model_data['scaler' ]
features = model_data[ 'features' ]
cols_to_scale = model_data['cols_to_scale']


def prepare_df(features):

    input_features = {
        'age': features['age'],
        'income': features['income'],
        'loan_amount': features['loan_amount'],
        'loan_tenure_months': features['loan_tenure_months'],
        'number_of_open_accounts': features['number_of_open_accounts'],
        'credit_utilization_ratio': features['credit_utilization_ratio'],
        'loan_to_income': features['loan_to_income'],
        'delinquency_ratio': features['delinquency_ratio'],
        'avg_dpd_per_delinquency': features['avg_dpd_per_delinquency'],
        'residence_type_Owned': 1 if features['residence_type']=='Owned' else 0,
        'residence_type_Rented': 1 if features['residence_type']=='Rented' else 0,
        'loan_purpose_Personal': 1 if features['loan_purpose']=='Personal' else 0,
        'loan_purpose_Education': 1 if features['loan_purpose']=='Education' else 0,
        'loan_type_Unsecured': 1 if features['loan_type']=='Unsecured' else 0,
        # additional features (Dummy)
        'gst': 1,
        'zipcode': 1,
        'enquiry_count': 1,
        'processing_fee': 1,
        'sanction_amount': 1,
        'net_disbursement': 1,
        'number_of_dependants': 1,
        'principal_outstanding': 1,
        'years_at_current_address': 1,
        'number_of_closed_accounts': 1,
        'bank_balance_at_application': 1
    }
    df=pd.DataFrame([input_features])
    # perform scaling
    df[cols_to_scale]=scaler.transform(df[cols_to_scale])
    df=df[features]
    return df

def find_score(prepared_df, base_score=300, scale_length=600):
    x=prepared_df.values()*model.coef_.T+model.intercept_
    # Apply the logistic function to calculate the probability
    default_probability=1 / (1+np.exp(-x))
    non_default_probability=1-default_probability

    # Convert the probability to a credit score, scaled to fit within 300 to 900
    score= base_score+non_default_probability.flatten()*scale_length

    return  non_default_probability,score


def toget_risk_level(score):
    # Determine risk level based on score
    if 300 <= score < 500:
        return 'Poor' , "red"
    elif 500 <= score < 650:
        return 'Average' ,"orange"
    elif 650 <= score < 750:
        return 'Good' , "blue"
    elif 750 <= score <= 900:
        return 'Excellent' ,"green"
    else:
        return 'Undefined' ,"black" # in case of any unexpected score

def predict_score(features):
    # prepare input dataframe for model
    input_df=prepare_df(features)
    # find credit score
    probability,score=find_score(input_df)
    # find credit score level
    risk_level, color=toget_risk_level(score)
    return probability, score, risk_level, color
