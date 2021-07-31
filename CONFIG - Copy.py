import os

# initialize the path to the *original* input directory of images
# main folder name
ORIG_INPUT_DATASET = "mini"

# Paths for training dataset
TRAIN_PATHS = ["mini/train/images",   # images path
               "mini/train/labels"]   # labels path

# Paths for validation dataset
VALIDATION_PATHS = ["mini/validation/images",  # images path
                    "mini/validation/labels"]  # labels path


# initialize the base path to the *new* directory that will contain
# our images after computing the training and testing split
EXTRACTION_PATH = "yolo"

# define the names of the training, testing, and validation
# directories
DIRECTORIES = ["train", "validation", "test"]

# initialize the list of class label names

#CLASSES = ["small-vehicle", "large-vehicle", "ship", "plane", "storage tank", "baseball diamond", "tennis court",
#           "basketball court", "ground track field", "harbor", "bridge", "helicopter", "roundabout", "soccer ball field"
#           "swimming pool", "container crane"]


CLASSES = ["small-vehicle", "large-vehicle", "ship", "plane"]

# set the batch size
BATCH_SIZE = 32

# define split images size
# must be multiple of 32 (e.g. 14x32 = 448)
INPUT_WIDTH = 640
INPUT_HEIGHT = 640

# txt containing the path and the name of the images
TXT_PATHS = ['test_yolo/train/train.txt',
             'test_yolo/validation/validate.txt',
             'test_yolo/test/test.txt']


OBJ_NAME = 'small_cars'
