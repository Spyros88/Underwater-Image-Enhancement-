import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.ndimage.filters import median_filter
import time
import timeit
path = 'C:\\opencv\\opencvImgEnh\\Underwater-Image-Enhancement-\\input.png'

def grayWorldAlgorithm(img): 

    b, g, r = cv2.split(img)
    m, n, t = img.shape
    sum_ = np.zeros(b.shape).astype(float)
    sum_ = (b.astype(float) + g.astype(float) + r.astype(float))  # Add each pixel rgb
    hists, bins = np.histogram(sum_.flatten(), 766, [0, 766])  # Array indexed by brightness
    Y = 765
    num, key = 0, 0
    ratio = 10
    while Y >= 0:  # Select threshold according to ratio
        num += hists[Y]
        if num > m * n * ratio / 100:
            key = Y
            break
        Y = Y - 1
    
    sum_b, sum_g, sum_r = 0, 0, 0
    time = 0
    maxvalue = np.max(img)

    # All pixels larger than the threshold are averaged
    a = np.where(sum_ >= key, b, 0 )
    h = np.where(sum_ >= key, g, 0 )
    f = np.where(sum_ >= key, r, 0 )
    sum_b = a.sum()
    sum_g = h.sum()
    sum_r = f.sum()

    time = cv2.countNonZero(a)

    avg_b = sum_b / time
    avg_g = sum_g / time
    avg_r = sum_r / time

    #Quantify each pixel 
    b = img[:,:,0].astype(float) * maxvalue / int(avg_b)
    g = img[:,:,1].astype(float) * maxvalue / int(avg_g)
    r = img[:,:,2].astype(float) * maxvalue / int(avg_r)
    intb = b.astype(int)
    intg = g.astype(int)
    intr = r.astype(int)
    b[b>255] = 255
    g[g>255] = 255
    r[r>255] = 255

    img[:,:,0] = b
    img[:,:,1] = g
    img[:,:,2] = r

    return img

def unsharp(image, sigma, strength):
    # Median filtering
    image_mf = median_filter(image, sigma)
    # Calculate the Laplacian
    lap = cv2.Laplacian(image_mf,cv2.CV_64F)
    # Calculate the sharpened image
    sharp = image-strength*lap
    # Saturate the pixels in either direction
    sharp[sharp>255] = 255
    sharp[sharp<0] = 0

    return sharp

def adjust_gamma(image, gamma=0.7):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

def adjust_gamma_sharp(img):
    adjusted = adjust_gamma(img)
    sharp = np.zeros_like(adjusted)
    for i in range(3):
        sharp[:,:,i] = unsharp(adjusted[:,:,i], 5, 0.8)
    return sharp

if __name__ == '__main__':
    start_time = time.time()
    #Get input and save original image
    img = cv2.imread(path)
    orig = np.array(img.shape)
    orig = np.copy(img)
    start_time = time.time()
    #Apply Grayworld Algorithm
    gray = grayWorldAlgorithm(img)
    #Apply gamma correction and sharpenning
    sharp = adjust_gamma_sharp(gray)
    #Apply CLAHE on L axis
    LAB = cv2.cvtColor(sharp, cv2.COLOR_BGR2LAB)
    (L, A, B) = cv2.split(LAB) 
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(L)
    merged = cv2.merge((equalized, A, B))
    final = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    outputImg = np.concatenate((orig, final), axis=1)
    # cv2.imwrite('PATH TO SAVE IMAGE', outputImg)
    cv2.namedWindow("OUTPUT", 0)
    cv2.resizeWindow("OUTPUT", 1000, 500)
    cv2.imshow("OUTPUT", outputImg) 
    cv2.waitKey(0)
    # Uncomment to calculate execution time
    # end_time = time.time()
    # print(f'Execution time:{end_time-start_time}')
