import pandas as pd
import os
import numpy as np
import CONFIG
import time
from PIL import Image


class Utils:

    def __init__(self, make_dataset="both"):
        self.make_dataset = make_dataset
        self.objects = CONFIG.CLASSES
        self.main_folder = CONFIG.EXTRACTION_PATH
        self.train_path = ''
        self.train_txt_path = ''
        self.validate_path = ''
        self.validate_txt_path = ''
        self.train_path_images = ''
        self.train_path_labels = ''
        self.validate_path_images = ''
        self.validate_path_labels = ''
        self.list_train_images = []
        self.list_validate_images = []
        self.train_files = []
        self.validation_files = []

        self.create_main_folder()

    def create_main_folder(self):
        self.create_folder(self.main_folder)

        if self.make_dataset == "both":
            self.create_train_images()
            self.create_validation_images()

        if self.make_dataset == 'train':
            self.create_train_images()

        if self.make_dataset == 'validation':
            self.create_validation_images()

    def create_train_images(self):
        self.train_path = os.path.join(self.main_folder, 'train')
        self.train_txt_path = os.path.join(self.train_path, 'train.txt')
        self.train_path_images = os.path.join(self.train_path, 'images')
        self.train_path_labels = os.path.join(self.train_path, 'labels')

        # create the folders
        self.create_folder(self.train_path)
        self.create_folder(self.train_path)
        self.create_folder(self.train_path)

        # get the images we need
        self.train_files = self.extract_needed_images(CONFIG.TRAIN_PATHS[1])

    def create_validation_images(self):
        self.validate_path = os.path.join(self.main_folder, 'validation')
        self.validate_txt_path = os.path.join(self.train_path, 'validation.txt')
        self.validate_path_images = os.path.join(self.validate_path, 'images')
        self.validate_path_labels = os.path.join(self.validate_path, 'labels')

        # create the folders
        self.create_folder(self.validate_path)
        self.create_folder(self.validate_path_images)
        self.create_folder(self.validate_path_labels)

        # get the images we need
        self.validation_files = self.extract_needed_images(CONFIG.VALIDATION_PATHS[1])

    def create_folder(self, path_to_folder):
        if not os.path.exists(path_to_folder):
            os.makedirs(path_to_folder)

    def extract_needed_images(self, path):
        """
        This function takes as input the path for the labels (labels are in .txt format),
        it opens them with pandas and we check if it has the classes we need.

        Parameters
        ----------  
        path : str
            path to the labels of the set (train path or valid path).
        
        Returns
        -------
            a list with the names of all files we need.
        """

        # makes a list of all .txt files in labels folder
        image_labels = [f for f in os.listdir(path) if f.endswith('.txt')]

        # The list with all the file names we need (e.g. ["P0000", "P0001", "P0003", ...] )
        needed_filenames = []

        columnnames = ['x1', 'y1', 'x2', 'y2', 'x3',
                       'y3', 'x4', 'y4', 'object', 'difficulty']

        # itterate over the labels and keep those we need
        for image_label in image_labels:
            full_label_path = os.path.join(path, image_label)
            temp_labels_csv = pd.read_csv(full_label_path,
                                          names=columnnames,
                                          delim_whitespace=True,
                                          skiprows=2)
            mask = np.column_stack([temp_labels_csv[col].isin(self.objects) for col in temp_labels_csv])
            result_rows = temp_labels_csv.loc[mask.any(axis=1)]

            if not result_rows.empty:
                needed_filenames.append(image_label[:-4])

        return needed_filenames
