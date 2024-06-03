import streamlit as st
import cv2
from matplotlib import pyplot as plt
import numpy as np
import time
import os
import io
import torch
import requests
from PIL import Image
from RealESRGAN import RealESRGAN
from skimage.metrics import mean_squared_error, structural_similarity
from io import BytesIO
from diffusers import StableDiffusionUpscalePipeline


def downscale_image(image, scale_factor=4):
    # Calculate the new dimensions
    new_width = image.width // scale_factor
    new_height = image.height // scale_factor
    # Resize the image
    return image.resize((new_width, new_height), Image.ANTIALIAS)


def main():
    st.title("Image Upscaling")
    uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        # Convert to an image:
        image = Image.open(io.BytesIO(bytes_data))
        
        # Show the original image
        st.image(image, caption='Original Image', use_column_width=True)
        
        # Downscale the image
        downscaled_image = downscale_image(image)
        
        # Show the downscaled image
        st.image(downscaled_image, caption='Downscaled Image', use_column_width=True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = RealESRGAN(device, scale=4)
        model.load_weights('weights/RealESRGAN_x4.pth', download=True)

        start_time = time.time()
        sr_image = model.predict(downscaled_image)
        end_time = time.time()

        st.image(sr_image, caption='Upscaled Image', use_column_width=True)

        # Calculate the elapsed time
        elapsed_time = end_time - start_time
        st.write(f"Time taken to run the RealESRGAN model: {elapsed_time:.2f} seconds")

        # Convert images to numpy arrays for metric calculations
        input_image_np = np.squeeze(image)
        sr_image_np = np.squeeze(sr_image)

        # Ensure both images have the same dimensions for comparison
        if input_image_np.shape != sr_image_np.shape:
            sr_image_np = np.array(sr_image.resize(image.size))

        # Calculate MSE
        mse_score = mean_squared_error(input_image_np, sr_image_np)
        st.write(f"MSE score: {mse_score}")


if __name__ == "__main__":
    main()