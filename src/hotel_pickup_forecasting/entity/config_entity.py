from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    unzip_data_dir: Path
    STATUS_FILE: str
    all_schema: dict

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    all_schema: list
    target: str
    cutoff_date: str
    
@dataclass(frozen= True)
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path:Path
    model_name: str
    learning_rate: float
    n_estimators: float
    early_stopping_rounds: float
    target_column: str
    base_score: float
    booster:str
    objective:str
    max_depth: float
 

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    all_params: dict
    metric_file_name: Path
    target_column: str
    mlflow_uri: str