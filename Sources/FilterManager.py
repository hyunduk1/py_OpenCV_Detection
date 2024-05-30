import cv2
import numpy as np


#쿠다 gpu
def Cuda(img):
    print("CUDA_GPU_버젼 : " ,cv2.cuda.getCudaEnabledDeviceCount())
    cv2.cuda.setDevice(0)
    cuda_image = cv2.cuda_GpuMat()
    cuda_image.upload(img)


#명암
def beta(img, alpha, beta):
    adjusted_image = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return adjusted_image

#감마
def gamma(image, gamma=1.0): 
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


#밝기
def brightness(image, brightness=0):
    # 입력 이미지를 복사합니다.
    adjusted_image = image.copy()
    
    # 밝기 조절
    adjusted_image += brightness
    
    # 픽셀 값이 0 미만이 되지 않도록 클리핑합니다.
    adjusted_image = np.clip(adjusted_image, 0, 255)
    
    # uint8 형식으로 변환합니다.
    adjusted_image = adjusted_image.astype(np.uint8)
    return adjusted_image

#가우시안
def Gaussian(img, gaussian):
    blurred_image = cv2.GaussianBlur(img, (1, 1), gaussian)
    return blurred_image

#샤이닝 필터
def Sharpening(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian

#hsv And Seturation
def hsvFilter(img, hue_adjustment = 0, saturation_scale = 1.5):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_adjustment) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_scale, 0, 255)
    result_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return result_image

def MedianFilter(img, Blur):
    dst = cv2.medianBlur(img, Blur)
    
    return dst

def bilateralFilter(img, d, sigmaColor, sigmaSpace):
    #d : 공간의 지름
    #sigmaColor : 색상 공간에서 필터의 표준편차
    #sigmaSpace : 좌표 공간에서 필터의 표준편차
    dst = cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)
    return dst
    
    