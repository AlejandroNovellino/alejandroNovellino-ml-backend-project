from pydantic import BaseModel


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


class OnePredictionOutputDto(BaseModel):
    """
    Output DTO for one prediction.
    """

    prediction: str

    def __init__(self, prediction: str):
        # call pydantic basemodel constructor
        super().__init__(prediction=prediction)
