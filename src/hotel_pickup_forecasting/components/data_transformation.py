from hotel_pickup_forecasting.utils.common import *
from hotel_pickup_forecasting.entity.config_entity import DataTransformationConfig

class DataTransformation: 
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def initiate_transformation(self):

        config = self.config
        
        # Cleaning the dataset i.e. necessary columns featuring relevant columns 
        df = pd.read_csv(self.config.data_path, index_col= 'report_date')
        df.index = pd.to_datetime(df.index)

        logger.info("Converting Stay_date and cancel_date to numerical values")

        df['stay_date'] = date_to_numeric(pd.to_datetime(df['stay_date']))
        df['cancel_date'] = date_to_numeric(pd.to_datetime(df['cancel_date']))

        df = df[config.all_schema] 
        logger.info("Added data with only relevant features")
        df = df[df[config.target].isna()==False]
        logger.info("checked for any null values in target variable and neglected them.")
        cutt_off_date = config.cutoff_date

        train = df.loc[df.index < cutt_off_date]
        test = df.loc[df.index >= cutt_off_date]

        train.to_csv(os.path.join(config.root_dir, "train.csv"), index= False)
        test.to_csv(os.path.join(config.root_dir, "test.csv"), index= False)

        logger.info(f"Sucessful train and test split on cut off date as {cutt_off_date}")
        logger.info(f"shape of training set is {train.shape}")
        logger.info(f"shape of testing set is {test.shape}")

        
        
        

