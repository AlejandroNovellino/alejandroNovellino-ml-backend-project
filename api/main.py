import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from wrapper import LogisticRegressionModelWrapper
from dtos import OnePredictionInputDto, OnePredictionOutputDto
from mappers import map_to_output_dto


# load environment variables
load_dotenv()
cors_url = os.getenv("CORS_URL")

# load the ML model
model_wrapper = LogisticRegressionModelWrapper(
    model_path="./models/model.pkl",
    label_encoder_path="./models/encoder.pkl"
)

app = FastAPI()

# CORS middleware configuration
origins = [
    cors_url,
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"status": "Hi there! I'm classification API. I can help you choose what crop to plant."}


@app.post("/predict")
def predict(features_dto: OnePredictionInputDto) -> OnePredictionOutputDto:
    """
    Endpoint for doing one prediction.

    Args:
        features_dto (FeaturesDto): The features to do a prediction.
    """

    try:
        # get the data from the request as a dictionary
        features: dict = features_dto.model_dump()
        # mapp the model wrapper to the output dto
        result = model_wrapper.predict_one(features)

        # map from the model result to the output dto
        output = map_to_output_dto(result)

        return output

    except Exception as e:

        print(e)

        raise HTTPException(
            status_code=500,
            detail="Error doing the prediction."
        )

# handler for the deployment in vercel
handler = app
