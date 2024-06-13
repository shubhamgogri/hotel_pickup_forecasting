import pandas as pd
import os
from hotel_pickup_forecasting import logger
import joblib
from hotel_pickup_forecasting.entity.config_entity import ModelTrainerConfig
import xgboost as xgb


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    
    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)


        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[[self.config.target_column]]
        test_y = test_data[[self.config.target_column]]


        reg = xgb.XGBRegressor(base_score=self.config.base_score, 
                               booster=self.config.booster,
                            n_estimators=self.config.n_estimators,
                            early_stopping_rounds=self.config.early_stopping_rounds,
                            objective=self.config.objective,
                            max_depth=self.config.max_depth,
                            learning_rate=self.config.learning_rate)
        reg.fit(train_x, train_y,
                eval_set=[(train_x, train_y), (test_x,test_y)],
                verbose=100)

        logger.info(f"Model Parameters: {self.config}" )
        fi = pd.DataFrame(data=reg.feature_importances_,
             index=reg.feature_names_in_,
             columns=['importance'])
        fi = fi.sort_values('importance')

        logger.info(f"Feature Importance{fi.loc[(fi.importance > .005)]}")

        
        joblib.dump(reg, os.path.join(self.config.root_dir, self.config.model_name))

