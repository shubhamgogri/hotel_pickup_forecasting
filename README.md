# hotel_pickup_forecasting

Machine Learning Pipeline Steps Involved ->
1. Data Ingestion
2. Data Validation
3. Data Transformation
4. Model Training
5. Model Evaluation

**Dataset Lifecycle**
Updated Dataset  is easily used to Re-Train the model
Data Ingestion Stage
1. Can fetch the .zip File from remote server i.e. using the HTTPS link.
2. Creates a local copy of the File and extracts the content accordingly.

**Data Validation**

Schema of the Data is validated to ensure the relevant Features are present.                  
Output file created for Validation containing Status of the schema of the table.   

**Data Transformation**

Important Feature Engineering steps are carried out while keeping relevant features to train the model.           
Train and Test split is carried out on the Dataset on a specific Date ->(01-01-2024) with each files stored separately. 

**Model Training & Evaluation**

Dependant and Independant Features are separated for feeding the data into XGBOOST.             
XGBOOST model is trained using modifiable parameters and record saved on the DAGSHUB integrated with MLFlow library. 
Subsequently, the evaluation is carried out and results stored on the platform as well. 

The Model Parameters, and Model file is stored both locally and on the DAGSHUB platform.

DataStore -> YAML files.
1. CONFIG.YAML -> Useful for handling the directories and path to the intermediate files generated. 
2. PARAMS.YAML -> The file consists of all the parameters with model name required to train the model. 
3. SCHEMA.YAML -> Consists of list of Relevant features with target features and the Cutt_off date for the train and test split. 

The Datastorage Storage facilitates easier experimentations i.e.  
1. updating new version of same dataset, 
2. Tweaking the Parameters for the model.
3. Ability to integrate multiple algorithms and facilitate ensemble learning if required. 
4. Altering the Relevant Independent Features.
 