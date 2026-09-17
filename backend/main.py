from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional


app = FastAPI(
    title="OceanEmbed API",
    description="Backend API for subsurface ocean temperature reconstruction",
    version="1.0.0"
)


# --------------------------------------------------
# CORS
# Allows the React frontend to communicate with API
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Request model
# --------------------------------------------------

class PredictionRequest(BaseModel):
    lat: float
    lon: float
    date: str

    # Surface variables
    sst: Optional[float] = None
    sss: Optional[float] = None
    ssh: Optional[float] = None
    current_u: Optional[float] = None
    current_v: Optional[float] = None
    wind_u: Optional[float] = None
    wind_v: Optional[float] = None


# --------------------------------------------------
# Basic endpoints
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "OceanEmbed API is running",
        "status": "prototype"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# --------------------------------------------------
# Prediction endpoint
# TEMPORARY MOCK
# Will later connect to M2's trained model
# --------------------------------------------------

@app.post("/predict")
def predict(request: PredictionRequest):

    depths = [
        0, 5, 10, 20, 30,
        50, 75, 100, 125, 150,
        200, 300, 500, 700, 1000
    ]

    temperature = [
        28.4, 28.1, 27.7, 26.9, 25.8,
        24.1, 22.6, 20.8, 19.5, 18.4,
        16.7, 14.2, 10.8, 8.7, 6.2
    ]

    return {
        "location": {
            "lat": request.lat,
            "lon": request.lon
        },
        "date": request.date,
        "depths": depths,
        "temperature": temperature,
        "model_status": "mock"
    }


# --------------------------------------------------
# Validation
# TEMPORARY MOCK
# Will later connect to M3
# --------------------------------------------------

@app.get("/validation")
def validation():

    return {
        "status": "mock",
        "metrics": {
            "rmse": None,
            "mae": None,
            "correlation": None,
            "bias": None
        }
    }


# --------------------------------------------------
# Explainability
# TEMPORARY MOCK
# Will later connect to M4
# --------------------------------------------------

@app.get("/explanation")
def explanation(depth: Optional[int] = None):

    return {
        "depth": depth,
        "status": "mock",
        "feature_importance": {
            "sst": 0.40,
            "sss": 0.15,
            "ssh": 0.15,
            "current_u": 0.10,
            "current_v": 0.08,
            "wind_u": 0.07,
            "wind_v": 0.05
        },
        "insight": "Feature sensitivity will be provided by the explainability module."
    }