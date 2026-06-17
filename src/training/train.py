from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from .data.preprocessing import DataLoader, DataPreprocessor
from .features.engineer import FeatureEngineer
from .config.config import DATA_PATH, MODEL_PATH

class ModelTrainer:
    def __init__(self):
        self.data_loader = DataLoader(DATA_PATH)
        self.data_preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()

    def train_models(self):
        data = self.data_loader.load_data()
        data = self.data_preprocessor.preprocess(data)
        data = self.feature_engineer.create_features(data)
        # Split, train models and evaluate metrics here

    def save_model(self, model, model_name):
        pass  # Implement model saving functionality
