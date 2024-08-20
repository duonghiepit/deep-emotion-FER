from __future__ import print_function
import pandas as pd
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
from typing import Optional

class GenerateData:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def split_test(self) -> None:
        """
        Splits the data into training and validation datasets and saves them as CSV files.
        """
        try:
            train_csv_path = os.path.join(self.data_path, 'train.csv')
            train = pd.read_csv(train_csv_path)
            
            # Split data into training and validation sets
            val_data = train.iloc[:3589]
            train_data = train.iloc[3589:]
            
            val_data.to_csv(os.path.join(self.data_path, "val.csv"), index=False)
            train_data.to_csv(os.path.join(self.data_path, "train.csv"), index=False)
            
            print("Done splitting the test file into validation & final test file")
        except Exception as e:
            print(f"An error occurred while splitting the data: {e}")

    def str_to_image(self, str_img: str) -> Image.Image:
        """
        Converts a space-separated string of pixel values into an image.
        
        Args:
            str_img (str): Space-separated string of pixel values.
        
        Returns:
            Image.Image: The generated image.
        """
        try:
            imgarray_str = str_img.split()
            imgarray = np.array(imgarray_str, dtype=np.uint8).reshape(48, 48)
            return Image.fromarray(imgarray)
        except Exception as e:
            print(f"An error occurred while converting string to image: {e}")
            raise

    def save_images(self, datatype: str) -> None:
        """
        Saves images from a CSV file into the specified directory.
        
        Args:
            datatype (str): The type of data ('train', 'test', or 'val').
        """
        folder_name = os.path.join(self.data_path, datatype)
        csvfile_path = os.path.join(self.data_path, f"{datatype}.csv")
        print(f"Saving images to: {folder_name}")
        print(f"CSV file path: {csvfile_path}")

        try:
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            
            # Load data from CSV
            data = pd.read_csv(csvfile_path)
            print(f"CSV columns: {data.columns.tolist()}")  # Print column names
            if 'pixels' not in data.columns:
                raise ValueError("CSV file does not contain 'pixels' column.")
            
            images = data['pixels']
            #n{images.head()}")  # Print first few rows of the 'pixels' column
            number_of_imgs = len(images)
            
            for index in tqdm(range(number_of_imgs), desc=f"Saving {datatype} images"):
                img = self.str_to_image(images.iloc[index])
                img.save(os.path.join(folder_name, f"{datatype}_{index}.jpg"), 'JPEG')
            
            print(f'[INFO] Done saving {folder_name} data')
        except Exception as e:
            print(f"An error occurred while saving images: {e}")


