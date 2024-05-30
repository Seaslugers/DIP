import cv2
import numpy as np

def get_dark_channel(img, size=15):
    """计算暗通道"""
    min_img = np.amin(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark_channel = cv2.erode(min_img, kernel)
    return dark_channel

def get_atmosphere(img, dark_channel, top_percent=0.001):
    """估计大气光"""
    flat_img = img.reshape(-1, 3)
    flat_dark = dark_channel.ravel()
    num_pixels = len(flat_dark)
    num_search_pixels = int(max(num_pixels * top_percent, 1))
    indices = np.argpartition(flat_dark, -num_search_pixels)[-num_search_pixels:]
    brightest = flat_img[indices]
    atmosphere = np.max(brightest, axis=0)
    return atmosphere

def get_transmission(img, atmosphere, omega=0.95, size=15):
    """计算传输率"""
    normalized_img = img / atmosphere
    dark_channel = get_dark_channel(normalized_img, size)
    transmission = 1 - omega * dark_channel
    return transmission

def adaptive_wiener_filter(veiling, noise_var):
    """自适应维纳滤波器"""
    mean_veiling = cv2.blur(veiling, (5, 5))
    var_veiling = cv2.blur(veiling ** 2, (5, 5)) - mean_veiling ** 2
    filtered_veiling = mean_veiling + (var_veiling - noise_var) / (var_veiling + 1e-6) * (veiling - mean_veiling)
    return filtered_veiling

def dehaze(img, atmosphere, transmission, t_min=0.1):
    """去雾处理"""
    transmission = np.clip(transmission, t_min, 1)
    J = (img - atmosphere) / transmission[:, :, None] + atmosphere
    J = np.clip(J, 0, 1)
    return J

def estimate_noise(transmission):
    """估计噪声"""
    mean_transmission = cv2.blur(transmission, (5, 5))
    noise_var = np.var(transmission - mean_transmission)
    return noise_var

def restore_image(img):
    dark_channel = get_dark_channel(img)
    atmosphere = get_atmosphere(img, dark_channel)
    transmission = get_transmission(img, atmosphere)
    noise_var = estimate_noise(transmission)
    refined_transmission = adaptive_wiener_filter(transmission, noise_var)
    dehazed_img = dehaze(img, atmosphere, refined_transmission)
    return (dehazed_img * 255).astype(np.uint8)