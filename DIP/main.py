import os
import cv2
import numpy as np
import time
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from skimage.color import deltaE_ciede2000
from skimage import img_as_float
from skimage.color import rgb2lab
import argparse
from baseline import Dehazer
from retinex import retinex_AMSR
from Restoration import restore_image

def calculate_psnr(img1, img2):
    mse_value = mse(img1, img2)
    if mse_value == 0:
        return float('inf')
    max_pixel = 1.0
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse_value))
    return psnr_value

def calculate_ciede2000(img1, img2):
    lab1 = rgb2lab(img1)
    lab2 = rgb2lab(img2)
    return np.mean(deltaE_ciede2000(lab1, lab2))

def compute_metrics(dehazed_img, groundtruth_img):
    dehazed_img = img_as_float(dehazed_img).astype(np.float32)
    groundtruth_img = img_as_float(groundtruth_img).astype(np.float32)

    m = mse(dehazed_img, groundtruth_img)

    data_range = 1
    min_dim = min(dehazed_img.shape[:2])
    win_size = min(7, min_dim - (min_dim % 2 - 1))

    if min_dim < 7:
        s = np.nan
    else:
        s = ssim(dehazed_img, groundtruth_img, win_size=win_size, channel_axis=2, data_range=data_range)

    p = calculate_psnr(dehazed_img, groundtruth_img)
    c = calculate_ciede2000(dehazed_img, groundtruth_img)

    return m, s, p, c

def resize_and_convert(img):
    height, width = img.shape[:2]
    max_dimension = 600

    if max(width, height) > max_dimension:
        if width > height:
            new_width = max_dimension
            new_height = int(height * (new_width / width))
        else:
            new_height = max_dimension
            new_width = int(width * (new_height / height))
    else:
        new_width, new_height = width, height

    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    normalized_img = resized_img.astype(np.float64) / 255  # Normalize the image to [0, 1]

    return normalized_img

def process_image(img, algorithm):
    start_time = time.time()
    if algorithm == 'dehazer':
        processor = Dehazer(img,use_guided_filter=True)
        dehazed = processor.dehaze()
        dehazed = (dehazed * 255).astype(np.uint8)
    elif algorithm == 'amsrcr':
        dehazed = retinex_AMSR(img)
    elif algorithm == 'restoration':
        dehazed = restore_image(img)
    end_time = time.time()
    return dehazed, end_time - start_time

def process_single_image(image_path, groundtruth_path, algorithms, output_base_path, save_images):
    img = cv2.imread(image_path)
    groundtruth_img = cv2.imread(groundtruth_path) if groundtruth_path else None
    img = resize_and_convert(img)
    if groundtruth_img is not None and groundtruth_img.size > 0:
        groundtruth_img = resize_and_convert(groundtruth_img)
    image_name = os.path.basename(image_path)
    
    for algorithm in algorithms:
        try:
            dehazed, runtime = process_image(img, algorithm)
            if save_images:
                output_path = os.path.join(output_base_path, algorithm, 'Single_Image_Output')
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                cv2.imwrite(os.path.join(output_path, image_name), dehazed)

            if groundtruth_img is not None and groundtruth_img.size > 0:
                m, s, p, c = compute_metrics(dehazed, groundtruth_img)
                print(f"Algorithm: {algorithm} - MSE: {m}, SSIM: {s}, PSNR: {p}, CIEDE2000: {c}, Time: {runtime:.2f}s")
            else:
                print(f"Algorithm: {algorithm} - Time: {runtime:.2f}s")
            print(f'Written the image to the {output_path} directory' if save_images else '')
        except Exception as e:
            print(f"Error processing {image_path} with algorithm {algorithm}: {str(e)}")

def process_folder(folder_path, groundtruth_folder, algorithms, output_base_path, save_images):
    mse_values, ssim_values, psnr_values, ciede2000_values, times = {}, {}, {}, {}, {}
    
    for algorithm in algorithms:
        mse_values[algorithm] = []
        ssim_values[algorithm] = []
        psnr_values[algorithm] = []
        ciede2000_values[algorithm] = []
        times[algorithm] = []
    
    for j, groundtruth_img in zip(sorted(os.listdir(folder_path)), sorted(os.listdir(groundtruth_folder))):
        img_path = os.path.join(folder_path, j)
        groundtruth_img_path = os.path.join(groundtruth_folder, groundtruth_img)
        img = cv2.imread(img_path)
        groundtruth_img = cv2.imread(groundtruth_img_path) if groundtruth_folder else None
        img = resize_and_convert(img)
        if groundtruth_img is not None and groundtruth_img.size > 0:
            groundtruth_img = resize_and_convert(groundtruth_img)

        for algorithm in algorithms:
            try:
                dehazed, runtime = process_image(img, algorithm)
                times[algorithm].append(runtime)
                if save_images:
                    output_path = os.path.join(output_base_path, algorithm, 'Outputs')
                    if not os.path.exists(output_path):
                        os.makedirs(output_path)
                    cv2.imwrite(os.path.join(output_path, j), dehazed)
                if groundtruth_img is not None and groundtruth_img.size > 0:
                    m, s, p, c = compute_metrics(dehazed, groundtruth_img)
                    mse_values[algorithm].append(m)
                    ssim_values[algorithm].append(s)
                    psnr_values[algorithm].append(p)
                    ciede2000_values[algorithm].append(c)
            except Exception as e:
                print(f"Error processing {img_path} with algorithm {algorithm}: {str(e)}")
                continue
    
    for algorithm in algorithms:
        if groundtruth_folder:
            print(f"Algorithm: {algorithm} - Avg MSE: {np.mean(mse_values[algorithm])}, Avg SSIM: {np.mean(ssim_values[algorithm])}, Avg PSNR: {np.mean(psnr_values[algorithm])}, Avg CIEDE2000: {np.mean(ciede2000_values[algorithm])}, Avg Time: {np.mean(times[algorithm]):.2f}s")
        else:
            print(f"Algorithm: {algorithm} - Avg Time: {np.mean(times[algorithm]):.2f}s")
        print(f'Processed all images with algorithm {algorithm} and written to the {output_base_path}/{algorithm}/Outputs directory' if save_images else '')

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='Path for the image or folder of images')
    parser.add_argument('groundtruth_path', type=str, nargs='?', default=None, help='Path for the groundtruth image or folder of images')
    parser.add_argument('--output', type=str, help='Base output path for saving results')
    parser.add_argument('--algorithms', type=str, nargs='+', choices=['dehazer', 'amsrcr', 'restoration'], default=['dehazer'],
                        help='Select the algorithms to use for dehazing')
    parser.add_argument('--save', type=bool, default=True, help='Whether to save the dehazed images')
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    path = args.path
    groundtruth_path = args.groundtruth_path
    output_base_path = args.output
    algorithms = args.algorithms
    save_images = args.save

    is_single_image = any(ext in path for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.gif'])
    if is_single_image:
        process_single_image(path, groundtruth_path, algorithms, output_base_path, save_images)
    else:
        process_folder(path, groundtruth_path, algorithms, output_base_path, save_images)

if __name__ == "__main__":
    main()
