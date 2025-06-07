import os
from dotenv import load_dotenv
from pathlib import Path

from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from api.packages.wrapper import LogisticRegressionModelWrapper
from api.packages.dtos import OnePredictionInputDto


# load the ML model
model_wrapper = LogisticRegressionModelWrapper(
    model_path="./models/model.pkl",
    label_encoder_path="./models/encoder.pkl"
)

app = FastAPI()


BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

@app.get("/", response_class=HTMLResponse)
async def landing(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/predict", response_class=HTMLResponse)
async def get_predict_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict_crop(
    request: Request,
    form_data: OnePredictionInputDto = Depends(OnePredictionInputDto.as_form)
):
    try:
        # get the data from the request as a dictionary
        features: dict = form_data.model_dump()

        # map the model wrapper to the output dto
        result = model_wrapper.predict_one(features)

        return templates.TemplateResponse(
            "form.html", {
                "request": request,
                "prediction": result.prediction
            }
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Error doing the prediction."
        )
