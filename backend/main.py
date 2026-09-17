from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="OceanEmbed API",
    description="Backend API for subsurface ocean temperature reconstruction",
    version="1.0.0"
)

# Allow the React frontend to communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "OceanEmbed API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.get("/predict")
def predict():
    # Temporary mock response.
    # This will later be replaced with the actual ML model.
    return {
        "depths": [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000],
        "temperature": [
            28.4, 28.1, 27.7, 26.9, 25.8,
            24.1, 22.6, 20.8, 19.5, 18.4,
            16.7, 14.2, 10.8, 8.7, 6.2
        ]
    }