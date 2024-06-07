#write a python code to find the quality of your degraded image with respect to the original image(without built in function)
# to get the degraded image use built in function to degrade the image (to blur the image)
#add guassian noise to the degraded image( do compression jpeg 2000 and decompress it)

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import glymur
from skimage.metrics import structural_similarity as ssim

#for degradation of image
def degrade_image(original_image,blur_size=(25,25),noise_std=50):
    #blurring the image
    degraded_image = cv2.GaussianBlur(original_image,blur_size,1.5)
    #adding noise to the image
    noise = np.random.normal(0,noise_std,degraded_image.shape)
    noisy_image = np.clip(degraded_image+noise,0,255).astype(np.uint8)

    return noisy_image

original_image = cv2.imread('me.jpg')
degraded_image = degrade_image(original_image)
cv2.imshow('original_image',original_image)
cv2.imshow('degraded_image',degraded_image)
#save the degraded image
cv2.imwrite('degraded_image.jpg',degraded_image)
cv2.waitKey(0)

# #compression and decompression
# def compress_jpeg2000(image,compressname):
#     image = glymur.Jp2k(compressname,data=image)

# def decompress_jpeg2000(compressname,decompressname):
#     image = glymur.Jp2k(compressname).data
#     return image

# compress_jpeg2000(degraded_image,'degraded_image.jp2')
# decompressed_image = decompress_jpeg2000('degraded_image.jp2','decompressed_image.jp2')
# cv2.imwrite('decompressed_image.jpg',decompressed_image)

#quality of the degraded image
#using mse
def mse(original_image,degraded_image):
    if original_image.shape != degraded_image.shape:
        raise ValueError('The images have different dimensions')
    
    height,width,channel = original_image.shape
    mse = 0.0

    for i in range(height):
        for j in range(width):
            for k in range(channel):
                mse += (original_image[i,j,k]-degraded_image[i,j,k])**2
    mse = mse/(height*width*channel)
    return mse


mse_value = mse(original_image,degraded_image)
print('MSE:',mse_value)

#using psnr
def psnr(original_image,degraded_image):
    mse_value = mse(original_image,degraded_image)
    peak = 255
    psnr = 10*math.log10((peak**2)/mse_value)
    return psnr

psnr_value = psnr(original_image,degraded_image)
print('PSNR:',psnr_value)
    