import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from pydantic import BaseModel, computed_field, Field
from typing import Annotated, Literal
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        model_path = "artifacts/model.pkl"
        preprocessor_path = "artifacts/preprocessor.pkl"

        model = load_object(file_path=model_path)
        preprocessor = load_object(file_path=preprocessor_path)

        scaled_data = preprocessor.transform(features)
        pred = model.predict(scaled_data)
        return pred

class InputData(BaseModel):
    gender: Annotated[Literal["male", "female"], Field(..., description="Gender")]
    race_ethnicity: Annotated[Literal["group A", "group B", "group C", "group D", "group E"], Field(..., description="Race of the person")]
    parental_level_of_education: Annotated[Literal["associate's degree", "bachelor's degree", "high school", "master's degree", "some college", "some high school"], Field(...,description="Parental education")]
    lunch: Annotated[Literal["free/reduced", "standard"], Field(...,description="Lunch")]
    test_preparation_course: Annotated[Literal["none", "completed"], Field(...,description="Test course")]
    reading_score: Annotated[int, Field(...,description="Reading score",ge=0, le=100)]
    writing_score: Annotated[int, Field(...,description="Writing score", ge=0, le=100)]


class CustomData:
    def __init__(self, data:InputData):
        self.data = data

    def get_data_as_dataframe(self):
        custom_data_input_dict = {
                "gender": [self.data.gender],
                "race_ethnicity": [self.data.race_ethnicity],
                "parental_level_of_education": [self.data.parental_level_of_education],
                "lunch": [self.data.lunch],
                "test_preparation_course": [self.data.test_preparation_course],
                "reading_score": [self.data.reading_score],
                "writing_score": [self.data.writing_score],
            }
        return pd.DataFrame(custom_data_input_dict)
    

