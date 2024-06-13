from hotel_pickup_forecasting import logger
from hotel_pickup_forecasting.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from hotel_pickup_forecasting.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from hotel_pickup_forecasting.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline


INGESTION = "Data Ingestion stage"
VALIDATION = "Data Validation stage"
TRANSFORMATION = "Data Transformation stage"
TRAINER = "MODEL TRAINER"
EVALUATION = "MODEL EVALUATION"


try:
   logger.info(f">>>>>> stage {INGESTION} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {INGESTION} completed <<<<<<\n\nx==========x")

   logger.info(f">>>>>> stage {VALIDATION} started <<<<<<")
   data_validation = DataValidationTrainingPipeline()
   data_validation.main()
   logger.info(f">>>>>> stage {VALIDATION} completed <<<<<<\n\nx==========x")

   logger.info(f">>>>>> stage {TRANSFORMATION} started <<<<<<")
   data_transformation = DataTransformationTrainingPipeline()
   data_transformation.main()
   logger.info(f">>>>>> stage {TRANSFORMATION} completed <<<<<<\n\nx==========x")

except Exception as e:
   logger.exception(e)
   raise e
