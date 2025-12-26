from fastapi import HTTPException
from pydantic import BaseModel, field_validator

class TrainRow(BaseModel):
    date: str
    close: float

class ContinueTrainRequest(BaseModel):
    model_name: str
    train_input: list[TrainRow]
    new_model_name: str

    @field_validator("train_input")
    @classmethod
    def check_train_input_length(cls, v):
        if len(v) < 6:
            raise HTTPException(status_code=400, detail="input must contain at least 6 rows of data!")
        return v

class ContinueTrainResponse(BaseModel):
    metrics: dict[str, float]

class PredictRequest(BaseModel):
    model_name: str
    input: list[float]

    @field_validator("input")
    @classmethod
    def check_input_length(cls, v):
        if len(v) != 5:
            raise HTTPException(status_code=400, detail="input must contain exactly 5 last closing prices!")
        return v

class PredictResponse(BaseModel):
    prediction: float

class ModelsResponse(BaseModel):
    models: list[str]