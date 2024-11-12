import cv2
import numpy as np
import os
from copy import deepcopy


def make_cropped_image(image, bbox, keypoint_coord):
    """
    Makes a 128x128 image that centers the taillight in the new cropped image
    :param image: The full image that contains the taillight of interest
    :param bbox: the bounding box in the following format: [upper_left_x, upper_left_y, bottom_right_x, bottom_right_y]
    :param keypoint_coord: the (x,y) coordinate of the taillight in the full image
    :return:
    """
    center_x, center_y = int(keypoint_coord[0]), int(keypoint_coord[1])
    bbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
    cropped_image = deepcopy(image[bbox[1]:bbox[3], bbox[0]:bbox[2]])
    translated_keypoint_x, translated_keypoint_y = int(center_x) - bbox[0], int(center_y) - bbox[1]

    # Crop the image around the internal
    top_left_x = max(translated_keypoint_x - 64, 0)
    padding_left = min(max(64 - translated_keypoint_x, 0), 128)
    top_left_y = max(translated_keypoint_y - 64, 0)
    padding_top = min(max(64 - translated_keypoint_y, 0), 128)
    bot_right_x = min(translated_keypoint_x + 64, cropped_image.shape[1])
    padding_right = min(max(translated_keypoint_x + 64 - cropped_image.shape[1], 0), 128)
    bot_right_y = min(translated_keypoint_y + 64, cropped_image.shape[0])
    padding_bottom = min(max(translated_keypoint_y + 64 - cropped_image.shape[0], 0), 128)

    cropped_internal = deepcopy(cropped_image[int(top_left_y):int(bot_right_y), int(top_left_x): int(bot_right_x)])

    # Padded image with internal centered in image
    padded_image = cv2.copyMakeBorder(cropped_internal, padding_top, padding_bottom, padding_left, padding_right,
                                      cv2.BORDER_CONSTANT, None, [0, 0, 0])
    if padded_image.shape[1] == 129:
        print(f"cropped_internal shape: {cropped_internal.shape}")
        print(f"padding right: {padding_right}")
        print(f"padding left: {padding_left}")
        print(f"padding top: {padding_top}")
        print(f"padding bottom: {padding_bottom}")

    return padded_image
