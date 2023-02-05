from pydantic import BaseModel

class request_body(BaseModel):
    
 Gender            :    int  
 Married           :    int  
 Dependents        :    int  
 Education         :    int  
 Self_Employed      :   int  
 ApplicantIncome   :    int  
 CoapplicantIncome  :   float
 LoanAmount        :    float
 Loan_Amount_Term  :    float
 Credit_History     :   float
 Property_Area     :    int  


