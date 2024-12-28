import joblib
import numpy as np
import pandas as pd
from pathlib import Path

class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path(r'C:\Users\Pulipati Aswhitha\Desktop\Poojitha\Kaggle\End_to_End_ML_Flow\End_to_End_ML_MLOPS_Project\artifacts\model_trainer\model.joblib'))

    def predict(self,data):
        prediction = self.model.predict(data)

        return prediction