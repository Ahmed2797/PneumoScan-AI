from project.pipeline import Training_Pipeline 
from project.exception import CustomException
import sys


# if __name__ == "__main__":
#     try:
#         training_pipeline = Training_Pipeline()
#         training_pipeline.run()
#     except Exception as e:
#         raise CustomException(e, sys)
    
from project.configeration import Configeration_Manager
from project.components.prepare_basemodel import Prepare_Segmentation_Model
from pathlib import Path
import sys

config_manager = Configeration_Manager()
prepare_base_model_config = config_manager.get_prepare_base_model_config()

model_preparer = Prepare_Segmentation_Model(config=prepare_base_model_config)
unet_model = model_preparer.build_resnet50_unet()

# Save to the 'updated_base_model' path defined in config
model_preparer.save_model(
    path=Path(prepare_base_model_config.update_base_model), 
    model=unet_model
)
