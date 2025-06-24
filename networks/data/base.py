from torch.utils.data import DataLoader, Dataset
from typing import *
import pandas as pd
import torch
from PIL import Image
import os
from .transform.transform import Transform

class _ImageDataset(Dataset):
    """A Torch base image dataset for vision models
    """
    def __init__(self, data_path:list, data_filenames:list, transform:Transform):
        self.data_filenames = data_filenames
        self.data_path = data_path
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data_filenames)

    def __getitem__(self, index:int) -> Union[torch.tensor, torch.tensor]:
        path = os.path.join(self.data_path[index], self.data_filenames[index])
        image = self.load_image(path)
        target = self.load_target(self.data_filenames[index])
        attributes = self.load_attributes(self.data_filenames[index])

        if self.transform is not None:
            transform_dict = self.transform(image)
        return {**transform_dict,
                "target": target,
                "filename":self.data_filenames[index],
                "path": path,
                **attributes
                }
    
    def load_image(self, path:str) -> torch.Tensor:
        try:
            extension = [".png", ".jpg", ".webp", ".jpeg", ".tiff",".tif", ""]
            for ext in extension:
                try:
                    image = Image.open(path + ext)
                    return image
                except:
                    pass
        except FileExistsError:
            print(f"Error. Failed to read image at {path}.")
            return None

    def load_filepath(self, metadata: pd.DataFrame) -> str:
        """ Load filepath function to be overridden
        """
        raise NotImplementedError("load_filepath method must be implemented by subclass.")

    def load_attributes(self, image_filename: str) -> dict:
        """ Load attributes function to be overridden
        """
        raise NotImplementedError("load_attributes method must be implemented by subclass.")

    def load_target(self, image_filename: str) -> int:
        """ Load target function to be overridden
        """
        raise NotImplementedError("load_target method must be implemented by subclass.")


class _BaseDataloader():
    """A torch base dataloader module 
    """
    def __init__(self, data_path:Union[str,list], cfg: dict):
        self.cfg = cfg
        self.data_path = self._list(data_path)
        self.train_dataset, self.val_dataset, self.test_dataset = self.prepare_data()

    def train_dataloader(self) -> DataLoader:
        """ Pytorch dataloader for training models

        Returns:
            DataLoader: Train dataloader
        """
        return DataLoader(self.train_dataset,
                          batch_size = self.cfg["DATA"]["BATCH_SIZE"],
                          shuffle=True,
                          num_workers = self.cfg["SYSTEM"]["NUM_WORKERS"], 
                          drop_last=True)
    
    def val_dataloader(self) -> DataLoader:
        """ Pytorch dataloader for validating models

        Returns:
            DataLoader: Validation dataloader
        """
        return DataLoader(self.val_dataset,
                          batch_size = self.cfg["DATA"]["BATCH_SIZE"],
                          shuffle=False,
                          num_workers = self.cfg["SYSTEM"]["NUM_WORKERS"], 
                          drop_last=False)
    
    def test_dataloader(self) -> DataLoader:
        """ Pytorch dataloader for testing models

        Returns:
            DataLoader: Test dataloader
        """
        return DataLoader(self.test_dataset,
                          batch_size = self.cfg["DATA"]["BATCH_SIZE"],
                          shuffle=False,
                          num_workers = self.cfg["SYSTEM"]["NUM_WORKERS"], 
                          drop_last=False)
    
    def _list(self, input:Union[str,list]) -> list:
        if isinstance(input, str):
            input = [input]
        assert isinstance(input, list), TypeError(f"Error. Input should be str or list but here is {type(input)}")
        return input

    def load_metadata(self, metadata_path:str) -> pd.DataFrame:
        """ Load metadata csv file from path 

        Args:
            metadata_path (str): Path to csv file

        Returns:
            pd.DataFrame: Metadata dataframe
        """
        try:
            df = pd.read_csv(metadata_path)
        except FileExistsError:
            print(f"Failed to read csv file at {metadata_path}")
            return None
        return df
    
    def load_text_metadata(self, txt_path:str) -> pd.DataFrame:
        """ Load metadata csv file from path

        Args:
            metadata_path (str): Path to csv file

        Returns:
            pd.DataFrame: Metadata dataframe
        """
        try:
            df = pd.read_csv(txt_path, sep=" ", header=None)
        except FileExistsError:
            print(f"Failed to read txt file at {txt_path}")
            return None
        return df
    
    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        raise NotImplementedError("Error. prepare_data method should be implemented in subclass.")
    
    def prepare_metadata(self, data_path:list) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        raise NotImplementedError("Error. prepare_data method should be implemented in subclass.")
           
    def clean_metadata(self, metadata:pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Error. clean_metadata method should be implemented in subclass.")

    def split_data(self, metadata:pd.DataFrame, cfg:dict) -> Union[pd.DataFrame, pd.DataFrame]:
        raise NotImplementedError("Error. split_data method should be implemented in subclass.")