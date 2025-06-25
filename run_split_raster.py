import glob
import os
import config as config
from utils.utils_split_raster import split_raster


def ensure_directories():
    """
    Creates required output directories if they don't exist.
    """
    required_dirs = [
        config.TRAINVALTEST_DATASET_DIR,
        os.path.dirname(config.IMAGE_STATS_PATH),
        config.BASE_MODELPATH,
        config.FINE_MODELPATH,
    ]
    for dir_path in required_dirs:
        os.makedirs(dir_path, exist_ok=True)


def split_raster_function(data_path="PATH"):
    """
    This script splits the remote sensing data into small tiles and creates the labels.
    The labels are created for 3 classes: 0 and 1 for background and urban areas, and class 2 for slum polygons.
    All data should be of the same extent and resolution.
    Labels are added to the image tile file name and saved in the data/datasets directory, along with image statistics used for normalization.
    """
    ensure_directories()  # ✅ Ensure directories exist before anything else

    # get all tif files in the data directory
    datalist = glob.glob(os.path.join(data_path, "*.tif"))
    
    # split the raster files into tiles and create labels
    split_raster(datalist=datalist)


if __name__ == "__main__":
    split_raster_function(
        data_path=os.path.join(config.BASE_DIR, "data")
    )
