from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
from predict import predict_score

app=FastAPI()

@app.get("/")
def bing():
    return {"Started": "Credit Risk Classification"}


class feature_data_types (BaseModel):
    age: int
    income: float
    loan_amount: float
    loan_tenure_months: int
    num_open_accounts: int
    credit_utilization_ratio: float
    loan_to_income: float
    delinquency_ratio: float
    avg_dpd_per_delinquency: float
    residence_type: str  # Typically a categorical string
    loan_purpose: str    # Typically a categorical string
    loan_type: str        # Typically a categorical string


class output_datatype(BaseModel):
    probability: float
    score : float
    risk_level:str
    color: str

@app.post("/predict",response_model=output_datatype)
def predict(input_features:feature_data_types):

    try:

        probability, score, risk_level,color = predict_score(dict(input_features))
        return output_datatype(probability=probability, score=score, risk_level=risk_level,color=color)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


    # print(probability, score, risk_level)
# {
#     "age": 30,
#     "income": 75000,
#     "loan_amount": 150000,
#     "loan_tenure_months": 60,
#     "num_open_accounts": 4,
#     "credit_utilization_ratio": 25.5,
#     "loan_to_income": 2.0,
#     "delinquency_ratio": 0.5,
#     "avg_dpd_per_delinquency": 15.0,
#     "residence_type": "Owned",
#     "loan_purpose":"Home",
#     "loan_type": "Secured"
# }