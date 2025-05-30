from fastapi import FastAPI
from fastapi.responses import JSONResponse
from src.pipeline.predict_pipeline import InputData, CustomData, PredictPipeline


app = FastAPI()

@app.post("/predict")
def predict(data:InputData):
    custom_data = CustomData(data)
    input_df = custom_data.get_data_as_dataframe()

    predict_pipeline = PredictPipeline()
    results = predict_pipeline.predict(input_df)

    return JSONResponse(status_code=200, content={"prediction":results[0]})