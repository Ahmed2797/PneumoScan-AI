from project.components.data_ingestion import Data_Ingestion

from project.configeration import Configeration_Manager
from project.exception import CustomException
from project.logger import logging
import sys


class Training_Pipeline:
    """
    Orchestrates the complete machine learning workflow including:
    - Data ingestion
    - Base model preparation
    - Callback creation
    - Model training
    - Model evaluation
    
    This class executes each pipeline step sequentially using configurations
    provided by the ConfigerationManager.
    """

    def __init__(self):
        """
        Initialize the TrainingPipeline with a single instance of
        ConfigerationManager to access all configuration sections.
        """
        self.config = Configeration_Manager()

    def run_data_ingestion(self):
        """
        Execute the data ingestion pipeline.
        
        Steps:
        - Download the dataset from the specified URL.
        - Extract the downloaded zip file.
        
        Raises:
            CustomException: If any part of ingestion fails.
        """
        try:
            logging.info(">>>>>>> Data Ingestion started <<<<<<<<<")
            data_ingestion_config = self.config.get_data_ingestion_config()
            data_ingestion = Data_Ingestion(data_ingestion_config)
            data_ingestion.download_data()
            data_ingestion.extract_zip_file()
            logging.info(">>>>>>> Data Ingestion completed <<<<<<<<<")
        except Exception as e:
            raise CustomException(e, sys)


    def run(self):
        """
        Execute the full ML pipeline in order:
        1. Data ingestion
        2. Base model preparation
        3. Model training
        4. Model evaluation
        
        Raises:
            CustomException: If any stage of the pipeline fails.
        """
        try:
            logging.info(">>>>>>> Training Pipeline started <<<<<<<<<")
            self.run_data_ingestion()
            # self.run_prepare_base_model()
            # self.run_model_training()
            # self.run_model_evaluation()
            logging.info(">>>>>>> Training Pipeline completed <<<<<<<<<")
        except Exception as e:
            raise CustomException(e, sys)
