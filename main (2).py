from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
from skimage import io
from PIL import Image
import numpy as np
from utilities import preprocess_image, postprocess_image
from unet_model import UNetMODEL
from huggingface_hub import hf_hub_download
import os
import uvicorn
import random

app = FastAPI()

def save_file(upload_file: UploadFile, filename: str):
    with open(filename, "wb") as buffer:
        buffer.write(upload_file.file.read())

def load_model():
    model_path = 'model.pth'
    if not os.path.exists(model_path):
        model_path = hf_hub_download("briaai/RMBG-1.4", 'model.pth')
        
    net = UNetMODEL()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device)
    net.eval()
    return net, device

def process_image(image_path: str, model, device):
    model_input_size = [1024, 1024]
    orig_im = io.imread(image_path)
    orig_im_size = orig_im.shape[0:2]
    image = preprocess_image(orig_im, model_input_size).to(device)

    with torch.no_grad():
        result = model(image)

    result_image = postprocess_image(result[0][0], orig_im_size)
    return result_image, orig_im_size

@app.post("/remove_bg/")
async def remove_bg(file: UploadFile = File(...)):
    """
    Remove the background from an image.
    
    - **file**: The image file from which to remove the background.
    
    Returns the filename of the processed image with the background removed.
    """
    file_location = "temp_image.jpeg"
    save_file(file, file_location)
    
    net, device = load_model()
    result_image, orig_im_size = process_image(file_location, net, device)

    pil_im = Image.fromarray(result_image)
    no_bg_image = Image.new("RGBA", pil_im.size, (0, 0, 0, 0))
    orig_image = Image.open(file_location)
    no_bg_image.paste(orig_image, mask=pil_im)
    output_path = "example_image_no_bg.png"
    no_bg_image.save(output_path)

    return {"filename": output_path}

@app.post("/replace_bg/")
async def replace_bg(file: UploadFile = File(...)):
    """
    Remove the background from an image and replace it with a random background from the 'images/bg_imgs' folder.
    
    - **file**: The image file from which to remove the background.
    
    Returns the filename of the processed image with the new background.
    """
    # Save the uploaded image file
    file_location = "temp_image.jpeg"
    save_file(file, file_location)

    # Load the model
    net, device = load_model()

    # Process the image to get the result image and original image size
    result_image, orig_im_size = process_image(file_location, net, device)

    # Select a random background image from the specified folder
    bg_imgs_folder = "images/bg_imgs"
    bg_image_files = os.listdir(bg_imgs_folder)
    random_bg_image = random.choice(bg_image_files)
    background_location = os.path.join(bg_imgs_folder, random_bg_image)

    # Open and resize the background image
    background_im = Image.open(background_location).convert("RGBA")
    if background_im.size != orig_im_size:
        background_im = background_im.resize(orig_im_size[::-1], Image.LANCZOS)

    # Combine the original image with the background image
    pil_im = Image.fromarray(result_image)
    no_bg_image = Image.new("RGBA", pil_im.size, (0, 0, 0, 0))
    orig_image = Image.open(file_location).convert("RGBA")
    no_bg_image.paste(orig_image, mask=pil_im)
    no_bg_image = no_bg_image.resize(background_im.size)
    background_im.paste(no_bg_image, (0, 0), no_bg_image)

    # Save the final image with the new background
    output_path = "example_image_with_new_bg.png"
    background_im.save(output_path)

    return JSONResponse(content={"filename": output_path})

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
