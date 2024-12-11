import numpy as np
import cv2

from utils import out_of_bounds

def calculate_least_squares(Ix, Iy, It, window=9):
    Ih, Iw = Ix.shape

    dx = np.zeros((Ih, Iw), dtype=np.float32)
    dy = np.zeros((Ih, Iw), dtype=np.float32)

    # add padding for edge pixels to not go out of bounds
    border = window // 2

    Ix = cv2.copyMakeBorder(
        Ix,
        border, border, border, border,
        cv2.BORDER_CONSTANT,
        value=0
    ).astype("float32")
    Iy = cv2.copyMakeBorder(
        Iy,
        border, border, border, border,
        cv2.BORDER_CONSTANT,
        value=0
    ).astype("float32")
    It = cv2.copyMakeBorder(
        It,
        border, border, border, border,
        cv2.BORDER_CONSTANT,
        value=0
    ).astype("float32")

    for i in range(Ih):
        for j in range(Iw):
            # select the window of values
            ix_area = Ix[i:(i+window), j:(j+window)].flatten()
            iy_area = Iy[i:(i+window), j:(j+window)].flatten()
            it_area = It[i:(i+window), j:(j+window)].flatten()

            # concatenate into matrices
            A = np.concatenate((ix_area[:, np.newaxis], iy_area[:, np.newaxis]), axis=1)
            b = -it_area.T
            Ata = A.T @ A

            Ata_inverse = None
            if np.linalg.det(Ata) == 0:
                Ata_inverse = np.linalg.pinv(Ata)
            else:
                Ata_inverse = np.linalg.inv(Ata)
            
            result = (Ata_inverse @ (A.T @ b))

            dx[i, j] = result[0]
            dy[i, j] = result[1]
    
    return dy, dx

def warp_img(img, u, v, filterColor=-1, colored=False):
    Ih, Iw = u.shape

    output = np.zeros_like(img, dtype=np.float32)

    for i in range(Ih):
        for j in range(Iw):
            # find the u and v coordinates
            sample_u = round(j + u[i, j])
            sample_v = round(i + v[i, j])

            if out_of_bounds(sample_u, 0, Iw - 1) or out_of_bounds(sample_v, 0, Ih - 1):
                continue

            # ignore filter color because they are just background
            gray_color = 0
            if colored:
                gray_color = 0.114 * img[i, j, 0] + 0.587 * img[i, j, 1] + 0.299 * img[i, j, 2]
            else:
                gray_color = img[i, j]

            if gray_color == filterColor:
                continue

            if u[i, j] < 1:
                sample_u = j
            if v[i, j] < 1:
                sample_v = i
            
            # write the color to the output
            if colored:
                output[sample_v, sample_u, :] = img[i, j, :]
            else:
                output[sample_v, sample_u] = img[i, j]
    
    return output