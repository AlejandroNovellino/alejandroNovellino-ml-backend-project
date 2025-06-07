# Crop Recommendation API

This project provides a machine learning API for crop recommendation based on soil and weather features. It includes data exploration notebooks, model training, and a FastAPI-based prediction service.

## **IMPORTANT NOTE**

The project was developed on FastAPI because that was the tool we used for the final project.

## Link to the deployed API

The API is deployed in Render in the link: https://ml-backend-project.onrender.com/docs

The deployed project is the one on the following repository [AlejandroNovellino](https://github.com/AlejandroNovellino/alejandroNovellino-ml-backend-project/tree/main) this is because Render cannot have access to the projects in a Organization (or at least I couldn't connect it).

## Project Structure

- `notebooks/` — Jupyter notebooks for EDA and model training.
- `models/` — Trained model and encoder files.
- `api/` — FastAPI app and related code.
- `requirements-dev.txt` — Python dependencies for development.
- `requirements.txt` — Python dependencies for deployment.

## Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd <your-project-directory>
   ```
   
2. **Install dependencies:**
    ```bash
    # for development
    pip install -r requirements-dev.txt
    # for deployment
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

### Predict Crop values to test

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
