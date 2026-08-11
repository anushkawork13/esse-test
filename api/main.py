from fastapi import FastAPI
from src.inference import predict, health_check

app = FastAPI()

@app.post("/predict")
async def predict_price(property_details: dict):
    return predict(property_details)

@app.get("/health")
async def health():
    return health_check()
