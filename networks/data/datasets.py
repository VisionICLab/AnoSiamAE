from networks.data.transform.transform import Transform
from typing import *
import pandas as pd
from .base import _ImageDataset

class RFMiDDataset(_ImageDataset):
    def __init__(self, metadata:pd.DataFrame, cfg_data:dict, is_training:bool) -> None:
        """Custom pytorch dataset

        Args:
            data_folder (str): Path to data storage
            metadata (pd.DataFrame): Loaded and cleaned metadata.
            Should include the additionnal column:
                - target (int): 0 for normal and 1 for abnormal samples
            cfg_data (dict): Data configuration file
            is_training (bool): Training or evaluation augment scheme
        """
        self.metadata = metadata
        data_path, data_files = self.load_filepath(metadata)

        super().__init__(data_path,data_files, Transform(cfg_data, is_training))

    def load_filepath(self, metadata:pd.DataFrame) -> list:
        """Load filenames into a list from metadata

        Args:
            metadata (pd.DataFrame): metadata DataFrame

        Returns:
            list: filenames
        """
        try:
            assert hasattr(metadata, "media_id"), "No media_id attribute in metadata."
            assert hasattr(metadata, "data_path"), "No data_path attribute in metadata."            
            files = metadata["media_id"].values.tolist()
            data_path = metadata["data_path"].values.tolist()
            assert len(files)>0 and len(data_path)>0, "Metadata is empty."
            return data_path, files

        except AssertionError as msg:
            print(msg)
            return None

    def load_attributes(self, image_filename:str) -> dict:
        """Load corresponding attributes from the desired image

        Args:
            image_filename (str): Image filename

        Returns:
            dict: Patient attributes
        """
        img_metadata = self.metadata[self.metadata["media_id"]==image_filename]
        assert len(img_metadata)==1, IndexError(f"Non unique row for filename {image_filename} in metadata.")
        attributes = {attr: str(img_metadata[attr].values.item()) for attr in img_metadata.columns if not attr in ["media_id", "target"]}
        return attributes

    def load_target(self, image_filename:str) -> int:
        """Load corresponding AD target from the desired image

        Args:
            image_filename (str): Image filename

        Returns:
            int: target
        """
        img_metadata = self.metadata[self.metadata["media_id"]==image_filename]
        assert len(img_metadata)==1, IndexError(f"Non unique row for filename {image_filename} in metadata.")
        return img_metadata["target"].values.item()
