# Crop Recommendation API

This project provides a machine learning API for crop recommendation based on soil and weather features. It includes data exploration notebooks, model training, and a FastAPI-based prediction service.

## Project Structure

- `notebooks/` — Jupyter notebooks for EDA and model training.
- `models/` — Trained model and encoder files.
- `api/` — FastAPI app and related code.
- `requirements.txt` — Python dependencies.

## Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd <your-project-directory>
   ```
   
2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Set up environment variables. Create a .env file in the root directory:**
    ```bash
    KAGGLEHUB_CACHE=../.
   ```

## Run the API:
    ```bash
    # for development
    fastapi run ./api/main.py --reload
    
    # for production
    uvicorn api.main:app
    ```



## API Usage

### Predict Crop Endpoint: POST /predict

#### Request Body:
```json
{
  "N": 90,
  "P": 42,
  "K": 43,
  "temperature": 20.87974371,
  "humidity": 82.00274423,
  "ph": 6.502985292000001,
  "rainfall": 202.9355362
}
```


#### Response 
```json
{
  "crop": "rice"
}
```

## Notes
- The model and encoder files must be present in the **models/** directory as model.pkl and encoder.pkl.
- For development, you can use the provided Jupyter notebooks in **notebooks/** for EDA and model retraining.
