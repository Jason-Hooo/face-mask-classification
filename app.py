
import gradio as gr
import torch
from PIL import Image
import numpy as np
from src.model import EfficientNet
from src.dataset import test_transform
from pillow_heif import register_heif_opener

register_heif_opener()

trained_model = EfficientNet("B1", 4)

trained_model.load_state_dict(torch.load("weights/trained_model_parameters.pth", map_location="cpu"))

input = gr.Image(label="Image")
output = gr.Label(label="Classification Result")

def face_mask_classification(img):
    if isinstance(img, np.ndarray): 
        img = Image.fromarray(img)
    img = test_transform(img)
    img = img.reshape(-1, 3, 240, 240)

    trained_model.eval()
    with torch.inference_mode():
        y_pred = trained_model(img)

    y_pred = torch.softmax(y_pred, dim=1) 

    y_pred = y_pred.reshape(-1) 
    y_pred_list = y_pred.tolist()
    mask_wearing = ["Mask on chin", "Mask not covering nose", "Mask properly worn", "No mask"]
    y_pred_formatted = {mask_wearing[i]: y_pred_list[i] for i in range(4)} 

    return y_pred_formatted

iface = gr.Interface(fn=face_mask_classification,
                     inputs=input,
                     outputs=output,
                     examples=["data/examples/with_mask_example.jpg",
                               "data/examples/without_mask_example.jpg",
                               "data/examples/mmc_example.jpg",
                               "data/examples/mc_example.jpg"],
                     title="Face Mask Classification",
                     description="Upload photos with/without masks",
                     article="Author : Jason Ho 2024 / 10 / 1 ",
                     theme="soft",
                     submit_btn="Start Recognition",
                     clear_btn="Clear Image",
                     example_labels=["Mask properly worn", "No mask", "Mask not covering nose", "Mask on chin"]
        )

if __name__ == "__main__":
    iface.launch()
