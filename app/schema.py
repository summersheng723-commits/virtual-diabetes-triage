from pydantic import BaseModel, Field

class DiabetesFeatures(BaseModel):
    age: float = Field(..., description="standardized age feature")
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float

class PredictionResponse(BaseModel):
    prediction: float

class HealthResponse(BaseModel):
    status: str
    model_version: str
