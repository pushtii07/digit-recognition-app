import numpy as np
import cv2

from scipy.ndimage import center_of_mass, shift


def preprocess_to_grayscale(image):
    """
    Convert an RGBA image to grayscale and binarize it.
    """

    image = image.astype(np.uint8)

    grayscale = cv2.cvtColor(
        image,
        cv2.COLOR_RGBA2GRAY
    )

    _, binary = cv2.threshold(
        grayscale,
        50,
        255,
        cv2.THRESH_BINARY
    )

    return grayscale, binary


def get_bounding_crop(binary_image):
    """
    Find the bounding box around the digit
    and crop the image tightly around it.
    """

    coordinates = cv2.findNonZero(binary_image)

    if coordinates is None:
        return None

    x, y, width, height = cv2.boundingRect(
        coordinates
    )

    cropped_image = binary_image[
        y:y + height,
        x:x + width
    ]

    return cropped_image


def pad_to_square(image):
    """
    Add black padding around the image
    so that height and width become equal.
    """

    height, width = image.shape

    square_size = max(height, width)

    vertical_padding = square_size - height
    horizontal_padding = square_size - width

    top = vertical_padding // 2
    bottom = vertical_padding - top

    left = horizontal_padding // 2
    right = horizontal_padding - left

    padded_image = cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_CONSTANT,
        value=0
    )

    return padded_image


def resize_image(image, target_size):
    """
    Resize the image to the selected resolution
    using area-based interpolation.
    """

    return cv2.resize(
        image,
        (target_size, target_size),
        interpolation=cv2.INTER_AREA
    )


def recenter_image(image):
    """
    Calculate the center of mass of the digit
    and shift it toward the image center.
    """

    center_y, center_x = center_of_mass(image)

    image_height, image_width = image.shape

    target_y = image_height / 2
    target_x = image_width / 2

    shift_y = target_y - center_y
    shift_x = target_x - center_x

    recentered_image = shift(
        image,
        shift=(shift_y, shift_x),
        mode="constant",
        cval=0
    )

    return recentered_image