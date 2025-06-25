import os
import numpy as np
import tifffile as tif
import torch
from torch.nn import CrossEntropyLoss
import time
import sys
sys.path.append("..")
import config
from utils._preprocessing import preprocess_input
from utils.utils_dataset import Dataset
from utils.utils_augmentations import get_validation_augmentation
from utils.utils_run_data_sampling import data_sampling_function
from models import ResNet50Custom

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype("float32")

def get_preprocessing(preprocessing_fn):
    return albu.Compose([albu.Lambda(image=preprocessing_fn), albu.Lambda(image=to_tensor)])

def run_inference(city="mumbai", overlap=2, modelpath=None, image_path=None):
    df = data_sampling_function(LSP=config.LSP, mode="unbalanced")
    df_city = df[df["city"] == city]
    files_test = df_city["path"].tolist()
    class_values = df_city["class"].tolist()

    test_dataset = Dataset(files_test, class_values, augmentation=get_validation_augmentation(), preprocessing=get_preprocessing(preprocess_input))
    image = tif.imread(image_path)[:, :, :3]
    pred_image = np.zeros([image.shape[0], image.shape[1], overlap], dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet50Custom(input_channel=3, num_classes=config.CLASSES).to(device)
    model.load_state_dict(torch.load(modelpath)["model_state_dict"])
    model.eval()

    with torch.no_grad():
        for i in range(len(files_test)):
            image_tile, label, file_name = test_dataset[i]
            x_tensor = torch.from_numpy(image_tile).unsqueeze(0).to(device)
            pred = torch.softmax(model(x_tensor), dim=1).cpu().numpy()
            prob_slum = pred[0, 2]  # Class 2 is slum

            x0 = int(file_name.split("x0_")[-1].split("_")[0])
            x1 = int(file_name.split("x1_")[-1].split("_")[0].split(".")[0])
            pred_image[x0:x0+config.IMAGESIZE, x1:x1+config.IMAGESIZE, :] = prob_slum

    pred_majority = np.mean(pred_image, axis=-1)
    save_dir = os.path.join(config.BASE_DIR, "results", city)
    os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(os.path.join(save_dir, f"prediction_{city}.tif"), (pred_majority * 255).astype(np.uint8))

if __name__ == "__main__":
    for city in config.LOOCV_CITY:
        run_inference(city=city, modelpath=os.path.join(config.FINE_MODELPATH, "ResNet50Custom", "SS", f"ResNet50Custom_{city}_128px_200shots_42_SS_adamw_WCEL_.pth"), image_path=config.RGB_PATH.format(city=city))