from fastapi import FastAPI
import uvicorn
from main import request_body
import loan 

app = FastAPI()
@app.post('/predict')
def prediction(data :request_body):
    test_data=[[data.Gender,data.Married,data.Dependents,data.Education,data.Self_Employed,data.ApplicantIncome,
    data.CoapplicantIncome,data.LoanAmount,data.Loan_Amount_Term,data.Credit_History,data.Property_Area]]
    res=loan.model.predict(test_data)[0]

    if res=='0':
        ans='NO'

    else:
        ans='YES'    


    return {"prediction" : ans}

    

    
   


