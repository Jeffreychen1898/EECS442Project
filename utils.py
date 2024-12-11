import numpy as np
import cv2
import scipy

# using opencv to load an image and convert it to gray scale. Return both images
def load_image(path):
    original_image = cv2.imread(path)
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY).astype("float32")
    original_image = original_image.astype("float32")

    return original_image, gray_image

# calculate the median filter of a grayscale image given a kernel size
def median_filter(img, kernel):
    if kernel % 2 == 0:
        print(f"Warning: median filter kernel size: {kernel} is not odd. Be aware of unexpected behavior")
        
    Ih, Iw = img.shape

    border = kernel // 2
    
    padded_img = cv2.copyMakeBorder(
        img,
        border, border, border, border,
        cv2.BORDER_REPLICATE
    )

    output = np.zeros_like(img, dtype=img.dtype)

    for i in range(Ih):
        for j in range(Iw):
            area = padded_img[i:(i+kernel), j:(j+kernel)].flatten()
            output[i, j] = np.median(area)
    
    return output

# upsample a grayscale image by doubling the resolution
def upsample(img, kernel_size=(5, 5)):
    Ih, Iw = img.shape
    upsampled = np.zeros((Ih * 2, Iw * 2), dtype=img.dtype)

    for i in range(Ih):
        for j in range(Iw):
            upsampled[i * 2, j * 2] = img[i, j]
    
    gaussian_radius = (kernel_size[0] // 2, kernel_size[1] // 2)
    upsampled = scipy.ndimage.gaussian_filter(
        upsampled,
        sigma=1,
        radius=gaussian_radius
    )
    upsampled *= 4

    return upsampled

#downsample a grayscale image by halving the resolution
def downsample(img, kernel_size=(5, 5)):
    Ih, Iw = img.shape
    downsampled = np.zeros((Ih // 2, Iw // 2), dtype=img.dtype)

    gaussian_radius = (kernel_size[0] // 2, kernel_size[1] // 2)
    blurred_image = scipy.ndimage.gaussian_filter(
        img,
        sigma=1,
        radius=gaussian_radius,
    )

    for i in range(Ih // 2):
        for j in range(Iw // 2):
            downsampled[i, j] = blurred_image[2 * i, 2 * j]
    
    return downsampled

# create a gaussian pyramid of a grayscale image from smallest to largest
def gaussian_pyramid(img, levels=3, kernel_size=(5, 5)):
    pyramid = [img]

    for i in range(levels - 1):
        downsampled_image = downsample(pyramid[-1], kernel_size)
        pyramid.append(downsampled_image)
    
    pyramid.reverse()
    return pyramid

# calculate the horizontal, vertical, and temporal gradients
# between the 2 images.
def calculate_image_gradients(img1, img2):
    Ih, Iw = img1.shape

    kernel_x = np.array([[-1, 1], [-1, 1]])
    kernel_y = np.array([[-1, -1], [1, 1]])
    kernel_t = np.array([[1, 1], [1, 1]])

    grad_x = cv2.filter2D(img1, -1, kernel_x)
    grad_y = cv2.filter2D(img1, -1, kernel_y)
    grad_t = cv2.filter2D(img2, -1, kernel_t) - cv2.filter2D(img1, -1, kernel_t)

    return grad_x, grad_y, grad_t

def out_of_bounds(value, low, high):
    return value < low or value > high