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

# initialize the list of class label names

# CLASSES = ["small-vehicle", "large-vehicle", "ship", "plane", "storage tank", "baseball diamond", "tennis court",
# "basketball court", "ground track field", "harbor", "bridge", "helicopter", "roundabout", "soccer ball field"
# "swimming pool", "container crane"]

CLASSES = ["small-vehicle", "large-vehicle", "ship", "plane"]

# define split images size
# must be multiple of 32 (e.g. 14x32 = 448)
INPUT_WIDTH = 512
INPUT_HEIGHT = 512
