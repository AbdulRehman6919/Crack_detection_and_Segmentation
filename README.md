# Crack-Detection-And-Segmentation

The provided repository includes PyTorch code for detecting cracks on concrete surfaces. The implementation is based on the DeepCrack model, which utilizes Convolutional Neural Networks (CNNs) for crack damage detection. DeepCrack employs a deep hierarchical feature learning architecture specifically designed for crack segmentation.


# Overview

- Resources 
- Libs Requirements
- Dataset
- Results


## Resources: 
[(Paper: https://github.com/yhlleo/DeepCrack/blob/master/paper/DeepCrack-Neurocomputing-2019.pdf)]
Architecture: based on Holistically-Nested Edge Detection, ICCV 2015.


## Libs Requirements:

PyTorch,
OpenCV,
Dataset -The data set can be downloaded from this link: https://data.mendeley.com/datasets/5y9wdsg2zt/2

## Dataset:

The dataset is comprised of concrete images containing both cracked and non-cracked instances, sourced from diverse structures within the METU Campus Buildings. It is bifurcated into two distinct categories: negative class, representing images without cracks, and positive class, representing images with cracks. Each class encompasses a set of 20,000 images, resulting in a collective dataset of 40,000 images. The images are standardized to dimensions of 227 x 227 pixels with RGB channels. Originating from 458 high-resolution images (4032x3024 pixels), the dataset is generated utilizing the methodology proposed by Zhang et al. (2016). The high-resolution images showcase variations in surface finish and lighting conditions. Notably, no data augmentation techniques, such as random rotation or flipping, have been applied.

The dataset file incorporates a training dataset class tailored for integration into Convolutional Neural Networks (CNNs). This class dynamically determines the number of classes by inspecting the folders within the 'in_dir,' where the count of folders corresponds to the number of distinct classes.

![Capture](https://github.com/yhlleo/DeepCrack/blob/master/figures/architecture.jpg?raw=true)



The secondary prediction process involves the utilization of two files: cv2_utils.py and inference_utils.py. Inference_utils.py employs the DeepCrack model to generate predictions for the mask, while cv2_utils.py utilizes OpenCV to extract pertinent parameters. Subsequently, a web application is developed, enabling users to input a set of images featuring cracks. The output includes information on the length, width, and category of the cracks, along with a corresponding mask highlighting the crack areas.

## Results

<img src="Crack Segmentation Image.png">
