
## Diffusion and GAN-based Image Upscaling Techniques

This repository hosts a dynamic image upscaling application built with Streamlit, leveraging multiple advanced upscaling models. Users can upload an image, select their preferred model, and receive an upscaled version along with performance metrics like execution time and mean squared error (MSE). The application features a user-friendly interface, making it accessible for both experts and novices. 

**Read our comprehensive study on image upscaling methods and how to pick the best one** - www.mercity.ai/blog-post/comparing-diffusion-and-gan-imgae-upscaling-techniques

![Pipeline](https://drive.google.com/uc?export=view&id=1NPAoHDN_dG98uZnJ97mNBoX_TekB3kWW)

## Installation

1. Clone the Repository:

```bash
git clone https://github.com/Mercity-AI/Image-Upscaling-Benchmark.git
cd Image-Upscaling-Benchmark
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit App

```bash
streamlit run streamlit_app.py
```

Open your web browser and go to http://localhost:8501.


## Overview


The dynamic image upscaling application uses various methods like Classical, Deep Learning, GAN-based, and Diffusion models to improve image quality. It displays the original image, its downscaled version, and the enhanced version of the downscaled image. The application also measures and compares performance metrics between the original and upscaled images, providing users with insights into the quality of each upscaling method.


![Model Comparison](https://drive.google.com/uc?export=view&id=1JYIzoVO342ufk_rHDUjS-OtygfUxLc2v)



## Explanation of Code - GAN-based Image Upscaling

Installation Commands

```bash
!pip install torch Pillow numpy scikit-image
!pip install git+https://github.com/sberbank-ai/Real-ESRGAN.gi
```

These lines install the required Python packages using pip. torch is PyTorch, used for deep learning models; Pillow is for image handling; numpy is for numerical operations; scikit-image is for image processing tasks. The second line installs the RealESRGAN library directly from its GitHub repository.


```bash
import torch
from PIL import Image
import numpy as np
from RealESRGAN import RealESRGAN
import time
from skimage.metrics import mean_squared_error, structural_similarity
```

```bash
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

This line sets the device for computation. If CUDA is available (indicating the presence of a GPU), it uses the GPU for faster computation; otherwise, it falls back to the CPU.


```bash
model = RealESRGAN(device, scale=4)
model.load_weights('weights/RealESRGAN_x4.pth', download=True)
```

These lines initialize the RealESRGAN model with the specified device and a scaling factor (scale=4 which means the output image will have four times the resolution of the input). It then loads the pretrained weights from a specified path, with an option to download the weights if they're not present locally.


```bash
path_to_image = 'path to your image'
image = Image.open(path_to_image).convert('RGB')
sr_image = model.predict(image)
sr_image.save('output path to your image')
```
## Explanation of Code - Diffusion based Image Upscaling

Installation Commands

```bash
!pip install diffusers transformers accelerate scipy safetensors
```

- Diffusers: Library for diffusion models including Stable Diffusion.

- Transformers: Library providing models for natural language processing and more.

- Accelerate: Simplifies running machine learning models on multi-GPU/multi-TPU setups.

- Scipy: Library used for scientific computing and technical computing.

- Safetensors: Used for safely serializing and deserializing tensor data.

```bash
import requests
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionUpscalePipeline
import torch
```
```bash
model_id = "stabilityai/stable-diffusion-x4-upscaler"
pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipeline = pipeline.to("cuda")
```

The model_id is a string that uniquely identifies the model on Hugging Face's model hub, in this case, "stabilityai/stable-diffusion-x4-upscaler", which specifies a version of the Stable Diffusion model specifically trained to upscale images by a factor of four. The method StableDiffusionUpscalePipeline.from_pretrained is used to load this model. Here, it is initialized with a specific configuration to use 16-bit floating-point precision (torch.float16), which is a strategy to reduce the memory usage, allowing the model to run faster and more efficiently on compatible hardware. The pipeline.to("cuda") command then shifts the model’s computations to a GPU, assuming one is available and CUDA-compatible. This significantly accelerates the processing speed, leveraging the GPU's ability to handle parallel computations, which is ideal for the intensive calculations required in upscaling images using deep learning models.

```bash
path = 'path to your image'
low_res_img = Image.open(path).convert("RGB")
low_res_img = low_res_img.resize((128, 128))
```

```bash
prompt = "prompt describing your image"
```

Define a prompt to guide the model in upscaling the image. This can be used to describe the content of the image or provide additional context to influence the upscaling process.

```bash
upscaled_image = pipeline(prompt=prompt, image=low_res_img).images[0]
upscaled_image.save("path to your output image")
```
## Explanation of Code - GAN-based Image Upscaling

Installation Commands

```bash
!pip install torch Pillow numpy scikit-image
!pip install git+https://github.com/sberbank-ai/Real-ESRGAN.gi
```

These lines install the required Python packages using pip. torch is PyTorch, used for deep learning models; Pillow is for image handling; numpy is for numerical operations; scikit-image is for image processing tasks. The second line installs the RealESRGAN library directly from its GitHub repository.


```bash
import torch
from PIL import Image
import numpy as np
from RealESRGAN import RealESRGAN
import time
from skimage.metrics import mean_squared_error, structural_similarity
```

```bash
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

This line sets the device for computation. If CUDA is available (indicating the presence of a GPU), it uses the GPU for faster computation; otherwise, it falls back to the CPU.


```bash
model = RealESRGAN(device, scale=4)
model.load_weights('weights/RealESRGAN_x4.pth', download=True)
```

These lines initialize the RealESRGAN model with the specified device and a scaling factor (scale=4 which means the output image will have four times the resolution of the input). It then loads the pretrained weights from a specified path, with an option to download the weights if they're not present locally.


```bash
path_to_image = 'path to your image'
image = Image.open(path_to_image).convert('RGB')
sr_image = model.predict(image)
sr_image.save('output path to your image')
```
## Explanation of Code - Deep Learning Algorithm based Image Upscaling

Installation Commands

```bash
import time
import cv2
from cv2  import dnn_superres
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
```

The time module is imported to enable timing operations, which can be useful for measuring the performance of the image processing tasks. The cv2 module from OpenCV is imported along with the dnn_superres submodule, which is specifically used for deep learning-based super-resolution to enhance the quality of images. The os module is included to handle file system operations such as reading and writing files.

```bash
image_path='/content/images'
output_path='/content/outputs'
```

```bash
MODEL_LIST = [
    {"name": "EDSR", "path":"/content/drive/MyDrive/Image_Upscaling/EDSR_x2.pb", "scale":2},
    {"name": "ESPCN", "path":"/content/drive/MyDrive/Image_Upscaling/ESPCN_x2.pb", "scale":2},
    {"name": "FSRCNN", "path":"/content/drive/MyDrive/Image_Upscaling/FSRCNN_x2.pb", "scale":2},
    {"name": "LAPSRN", "path":"/content/drive/MyDrive/Image_Upscaling/LapSRN_x2.pb", "scale":2},

    {"name": "EDSR", "path":"/content/drive/MyDrive/Image_Upscaling/EDSR_x3.pb", "scale":3},
    {"name": "ESPCN", "path":"/content/drive/MyDrive/Image_Upscaling/ESPCN_x3.pb", "scale":3},
    {"name": "FSRCNN", "path":"/content/drive/MyDrive/Image_Upscaling/FSRCNN_x3.pb", "scale":3},

    {"name": "EDSR", "path":"/content/drive/MyDrive/Image_Upscaling/EDSR_x4.pb", "scale":4},
    {"name": "ESPCN", "path":"/content/drive/MyDrive/Image_Upscaling/ESPCN_x4.pb", "scale":4},
    {"name": "FSRCNN", "path":"/content/drive/MyDrive/Image_Upscaling/FSRCNN_x4.pb", "scale":4},
    {"name": "LAPSRN", "path":"/content/drive/MyDrive/Image_Upscaling/LapSRN_x4.pb", "scale":4},

    {"name": "LAPSRN", "path":"/content/drive/MyDrive/Image_Upscaling/LapSRN_x8.pb", "scale":8},
]
```

Intialized all the Image upscaling models for different scaling factors and defined the input and output images directory path

```bash
def calculate_psnr(original,new):
    return cv2.PSNR(original,new)

def calculate_mse(original,new):
    return (np.square(original - new)).mean(axis=None)

def calculate_ssim(original,new):
    return ssim(original, new, channel_axis=2)

def save_image(path, result):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    cv2.imwrite(path, result)

```
The calculate_psnr function computes the Peak Signal-to-Noise Ratio (PSNR) between the original and the new (processed or upscaled) images using OpenCV's built-in cv2.PSNR function.
calculate_mse function calculates the Mean Squared Error (MSE) between the original and new images.
The calculate_ssim function computes the Structural Similarity Index (SSIM) between the original and new images, leveraging the ssim function from the skimage.metrics module.
Finally, the save_image function saves the processed image to a specified path using OpenCV's cv2.imwrite function


```bash
def model_run(enable_write):
    for model in MODEL_LIST:
        name = model['name']
        path = model['path']
        scale = model['scale']

        sr=dnn_superres.DnnSuperResImpl_create()
        sr.readModel(path)
        sr.setModel(name.lower(),scale)

        for i in os.listdir(image_path):
            input_image=cv2.imread(os.path.join(image_path,i))
            input_image = cv2.resize(input_image, (int(input_image.shape[1]/scale),int(input_image.shape[0]/scale)))
            processed_image = cv2.resize(input_image, (int(input_image.shape[1]*scale),int(input_image.shape[0]*scale)))

            output_dir = os.path.join(output_path, f'x{scale}')
            output_file_path = os.path.join(output_dir, f'{name}_x{scale}_{i}')

            begin=time.time()
            result_image=sr.upsample(input_image)
            interval=time.time()-begin
            print(scale)
            print(name)
            print(f'Time is : {interval}')
            print(f'PSNR is : {calculate_psnr(processed_image, result_image)}')
            print(f'MSE is : {calculate_mse(processed_image, result_image)}')
            print(f'SSIM is : {calculate_ssim(processed_image, result_image)}')

            print('')
            if enable_write==True:
                save_image(output_file_path, result_image)

model_run(True)

```


The model_run function processes images using a list of predefined super-resolution models, enhancing the resolution and evaluating the results based on various metrics. It begins by iterating through each model in the MODEL_LIST, retrieving the model's name, path, and scale factor. For each model, it initializes the dnn_superres.DnnSuperResImpl_create object from OpenCV's deep neural network super-resolution module. The model is then read from the specified path and configured using the readModel and setModel methods, respectively.

Within the function, it iterates over all images located in the image_path directory. Each image is read into memory using OpenCV's cv2.imread function and resized to simulate a lower resolution by dividing its dimensions by the scale factor. The low-resolution image is subsequently resized back to its original dimensions to prepare it for the super-resolution process.

The output directory and file paths are constructed dynamically based on the model's scale factor and name, ensuring organized storage of the results. The function measures the time taken to upscale each image using the sr.upsample method, providing insights into the model's performance. It prints the scale, model name, time taken, and calculated image quality metrics—PSNR, MSE, and SSIM—by comparing the upscaled image with the processed image.

If the enable_write parameter is set to True, the function saves the upscaled image to the specified output directory using the save_image function. 


## Screenshots

![Downscale Image Pipeline](https://drive.google.com/uc?export=view&id=1_dseucOIOvcR9HFCvQ-yx7p-T6mWwK6o)
![App Screenshot](https://drive.google.com/uc?export=view&id=1wSw04fnGOu0AfVEfZWiaBqkQDBouP7Ux)


## Contributions

Contributions are welcome! Please open an issue or submit a pull 
request for any improvements or suggestions.

## References

https://arxiv.org/html/2405.17261v1

https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler

https://huggingface.co/ai-forever/Real-ESRGAN/blame/a86fc6182b4650b4459cb1ddcb0a0d1ec86bf3b0/RealESRGAN_x4.pth


