import os
import sys
import time
import tensorflow as tf
from project.entity.config import Prepare_Callback_Config
from project.exception import CustomException
from project.constants import *

class Call_Backs:
    """
    Manages the creation of Keras callbacks for model training.

    This class provides properties to generate individual callbacks and a 
    method to retrieve a compiled list of all callbacks for use in model.fit().

    Args:
        config (PrepareCall_backConfig): Configuration object containing paths 
                                         and parameters for callbacks.
    """

    def __init__(self, config: Prepare_Callback_Config):
        """Initializes the Call_Backs manager with the provided configuration."""
        try:
            self.config = config
        except Exception as e:
            raise CustomException(e, sys)

    @property
    def create_TensorBoard_callback(self) -> tf.keras.callbacks.TensorBoard:
        """
        Creates a TensorBoard callback with a timestamped log directory.

        Returns:
            tf.keras.callbacks.TensorBoard: Callback for visualizing training logs.
        """
        try:
            timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
            # Creating a sub-directory with a timestamp to avoid overwriting previous runs
            tb_log_dir = os.path.join(
                self.config.tensorboard_root_log_dir,
                f"tb_logs_at_{timestamp}"
            )
            return tf.keras.callbacks.TensorBoard(log_dir=tb_log_dir)
        except Exception as e:
            raise CustomException(e, sys)

    @property
    def create_ModelCheckpoint_callback(self) -> tf.keras.callbacks.ModelCheckpoint:
        """
        Creates a ModelCheckpoint callback to save the best model weights.

        Returns:
            tf.keras.callbacks.ModelCheckpoint: Callback to save the model file.
        """
        try:    
            return tf.keras.callbacks.ModelCheckpoint(
                filepath=self.config.checkpoint_model_filepath,
                save_best_only=True,
                monitor=monitor,
                verbose=verbose
            )
        except Exception as e:
            raise CustomException(e, sys)
        
    @property
    def create_ReduceLROnPlateau_callback(self) -> tf.keras.callbacks.ReduceLROnPlateau:
        """
        Creates a ReduceLROnPlateau callback to dynamically adjust learning rate.

        Behavior:
            Reduces the learning rate by a factor of 0.1 if validation loss 
            does not improve for 5 consecutive epochs.

        Returns:
            tf.keras.callbacks.ReduceLROnPlateau: Callback for LR scheduling.
        """
        try:    
            return tf.keras.callbacks.ReduceLROnPlateau(
                monitor=monitor,
                factor=factor,
                patience=patience,
                min_lr=min_lr,
                verbose=verbose
            )
        except Exception as e:
            raise CustomException(e, sys)
    
    @property
    def create_EarlyStopping_callback(self) -> tf.keras.callbacks.EarlyStopping:
        """
        Creates an EarlyStopping callback to prevent overfitting.

        Behavior:
            Stops training if validation loss does not improve for 10 
            consecutive epochs and restores the best found weights.

        Returns:
            tf.keras.callbacks.EarlyStopping: Callback to halt training early.
        """
        try:    
            return tf.keras.callbacks.EarlyStopping(
                monitor=monitor, # Fixed: was 'fmonitor'
                patience=patience,
                restore_best_weights=restore_best_weights,
                verbose=verbose
            )
        except Exception as e:
            raise CustomException(e, sys)

    def get_tb_ckpt_callbacks(self) -> list:
        """
        Compiles all initialized callbacks into a single list.

        Returns:
            list: A list containing TensorBoard, ModelCheckpoint, 
                  ReduceLROnPlateau, and EarlyStopping callbacks.
        """
        try:    
            return [
                self.create_TensorBoard_callback, 
                self.create_ModelCheckpoint_callback,
                self.create_ReduceLROnPlateau_callback,
                self.create_EarlyStopping_callback
            ]
        except Exception as e:
            raise CustomException(e, sys)