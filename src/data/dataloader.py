from typing import *
import pandas as pd
from src.data.datasets import RFMiDDataset
import pandas as pd
from .base import _BaseDataloader
from .build import DATALOADER_REGISTRY
import os 

@DATALOADER_REGISTRY.register("rfmid")
class RFMiDDataloader(_BaseDataloader):
    """ Dataloader for RFMiD datasets
    """
    def __init__(self, data_path:Union[str,list], cfg: dict):
        super().__init__(data_path, cfg)

    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Training and validation data preparation

        Returns:
            Union[Dataset, Dataset]: _description_
        """
        print("***Preparing data for training and evaluation...")
        train_metadata, val_metadata, test_metadata = self.prepare_metadata(self.data_path)
        
        train_data, val_data, test_data = RFMiDDataset(train_metadata, self.cfg["DATA"], is_training=True), RFMiDDataset(val_metadata, self.cfg["DATA"], is_training=False), RFMiDDataset(test_metadata, self.cfg["DATA"], is_training=False)
        print("...Data ready for training and evaluation.")
        return train_data, val_data, test_data
    
    def clean_metadata(self, metadata:pd.DataFrame) -> pd.DataFrame:
        """ Clean metadata for uniformised processing

        Args:
            metadata (pd.DataFrame): dataframe to be cleansed

        Returns:
            pd.DataFrame: Cleansed dataframe
        """
        # Rename columns
        metadata = metadata.rename(columns={"ID":"media_id"})
        metadata = metadata.rename(columns={"Disease_Risk":"target"})

        # Change type 
        metadata["media_id"] = metadata["media_id"].astype(str)

        # Add patient_id column
        metadata["patient_id"] = [-1]*len(metadata)
        return metadata
    
    def prepare_metadata(self, data_path:list) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """ Load, clean and split metadata.

        Args:
            metadata_path (str): Path to metadata csv file

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, val and test metadata
        """
        train_metadata = []
        val_metadata = []
        test_metadata = []
        for data in data_path:
            # Load data
            train = self.load_metadata(os.path.join(data, "RFMiD_Training_Labels.csv"))
            val = self.load_metadata(os.path.join(data, "RFMiD_Validation_Labels.csv"))
            test = self.load_metadata(os.path.join(data, "RFMiD_Testing_Labels.csv"))

            # Clean data
            train = self.clean_metadata(train)
            val = self.clean_metadata(val)
            test = self.clean_metadata(test)

            # Remove abn in training set
            train = train[train["target"]==0]

            # Add data_path
            train["data_path"] = [os.path.join(data, "Training")]*len(train)
            val["data_path"] = [os.path.join(data, "Validation")]*len(val)
            test["data_path"] = [os.path.join(data, "Test")]*len(test)

            # Store metadata
            train_metadata.append(train)
            val_metadata.append(val)
            test_metadata.append(test)

        # Concat datasets
        return pd.concat(train_metadata), pd.concat(val_metadata), pd.concat(test_metadata)

