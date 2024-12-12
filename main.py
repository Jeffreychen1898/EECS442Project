import argparse
import numpy as np
import cv2

import utils
import lucas_kanade

LEVELS = 3
LK_WINDOW = 16
FILTER_WINDOW = 5
FILTER_COLOR = 0
MOTION_SCALE = 1

def calculate_flow(img1, img2, outimg, flowpath):
    frame1, gray1 = utils.load_image(img1)
    frame2, gray2 = utils.load_image(img2)

    # ASSERT IMAGE SIZE HERE

    pyramid1 = utils.gaussian_pyramid(gray1, LEVELS)
    pyramid2 = utils.gaussian_pyramid(gray2, LEVELS)

    low_res_h, low_res_w = pyramid1[0].shape
    prev_flow_u = None
    prev_flow_v = None
    output_flow_u = np.zeros((low_res_h, low_res_w), dtype=np.float32)
    output_flow_v = np.zeros((low_res_h, low_res_w), dtype=np.float32)

    for layer_src, layer_dst in zip(pyramid1, pyramid2):
        warped = lucas_kanade.warp_img(layer_src, output_flow_u, output_flow_v, FILTER_COLOR)
        # calculate optical flow
        grad_x, grad_y, grad_t = utils.calculate_image_gradients(layer_src, layer_dst)

        v, u = lucas_kanade.calculate_least_squares(grad_x, grad_y, grad_t, LK_WINDOW)

        # update flow and upscale for next level
        output_flow_u += u
        output_flow_v += v
        prev_flow_u = output_flow_u
        prev_flow_v = output_flow_v
        output_flow_u = utils.upsample(output_flow_u) * 2
        output_flow_v = utils.upsample(output_flow_v) * 2
    
    prev_flow_u = utils.median_filter(prev_flow_u, FILTER_WINDOW)
    prev_flow_v = utils.median_filter(prev_flow_v, FILTER_WINDOW)
    
    output_image = lucas_kanade.warp_img(
        frame1,
        prev_flow_u * MOTION_SCALE,
        prev_flow_v * MOTION_SCALE,
        FILTER_COLOR,
        True
    )
    cv2.imwrite(outimg, output_image)

    # create flow map image
    if flowpath == "":
        return

    brightness = (np.zeros_like(u) + 1) * 255
    hue = np.atan2(v, u)
    hue[hue < 0] += 2 * np.pi
    hue = np.degrees(hue + np.pi / 2)
    saturation = np.sqrt(u ** 2 + v ** 2)
    saturation[gray1 == FILTER_COLOR] = 0

    flow_img = np.stack((hue, saturation, brightness), axis=2)
    flow_img = cv2.cvtColor(flow_img, cv2.COLOR_HSV2BGR)
    cv2.imwrite(flowpath, flow_img)

if __name__ == "__main__":

    help_messages = [
        "The path to the starting frame!",
        "The path to the ending frame!",
        "The path of the output image!",
        "The scale of the object's motion.",
        "The path of the optical flow image!",
        "The number of gaussian pyramid layers to generate! Increase layer count for larger motions!",
        "The window size used in the Lucas Kanade algorithm! Increase size for larger objects!",
        "The kernel size of the median filter used for smoothing optical flow! Increase size for noisy optical flow!",
        "The background color of the image! This is used to mask optical flow for visualization purposes!",
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("start_frame", type=str, help=help_messages[0])
    parser.add_argument("end_frame", type=str, help=help_messages[1])
    parser.add_argument("output", type=str, help=help_messages[2])
    parser.add_argument("-m", "--motionScale", type=float, default=1, help=help_messages[3])
    parser.add_argument("-f", "--flowPath", type=str, default="", help=help_messages[4])
    parser.add_argument("-l", "--levels", type=int, default=3, help=help_messages[5])
    parser.add_argument("-w", "--window", type=int, default=9, help=help_messages[6])
    parser.add_argument("-k", "--kernel", type=int, default=5, help=help_messages[7])
    parser.add_argument("-b", "--background", type=int, default=-1, help=help_messages[8])

    args = parser.parse_args()

    LEVELS = args.levels
    LK_WINDOW = args.window
    FILTER_WINDOW = args.kernel
    FILTER_COLOR = args.background
    MOTION_SCALE = args.motionScale

    calculate_flow(args.start_frame, args.end_frame, args.output, args.flowPath)
