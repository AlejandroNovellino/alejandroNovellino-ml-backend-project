from pydantic import BaseModel
from fastapi import Form


class OnePredictionInputDto(BaseModel):
    """
    Input DTO features to make a prediction.
    """

    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

    @classmethod
    def as_form(
        cls,
        n: float = Form(...),
        p: float = Form(...),
        k: float = Form(...),
        temperature: float = Form(...),
        humidity: float = Form(...),
        ph: float = Form(...),
        rainfall: float = Form(...),
    ):
        return cls(N=n, P=p, K=k, temperature=temperature, humidity=humidity, ph=ph, rainfall=rainfall)


class OnePredictionOutputDto(BaseModel):
    """
    Output DTO for one prediction.
    """

    prediction: str

    def __init__(self, prediction: str):
        # call pydantic basemodel constructor
        super().__init__(prediction=prediction)
