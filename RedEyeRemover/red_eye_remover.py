import cv2
import numpy as np

def remove_red_eye(red_eye_image):
    """
    Receives an image as input, copies it, and removes the red eye from the image
    Returns a new image without red eye
    """
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    fixed_image = red_eye_image.copy()

    eyes = eye_cascade.detectMultiScale(red_eye_image, 1.3, 5)

    for x, y, w, h in eyes:
        eye_section = red_eye_image[y:y + h, x:x + w]

        cv2.rectangle(fixed_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        blue, green, red = cv2.split(eye_section)
        blue_green_sum = cv2.add(blue, green)

        # generate a mask with rules for where red dominates
        mask = (red > blue_green_sum) & (red > 90)
        mask = mask.astype(np.uint8) * 255

        # find the largest contour that adheres to the mask, this is most likely the red eye region
        _, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        eye_contour = get_largest_contour(contours)

        mask = resetMask(mask)

        cv2.drawContours(mask, [eye_contour], 0, 255, -1)

        # expand the eye correction area because the edges are always picked up
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)

        blue_green_mean = configure_mean(blue_green_sum, mask)

        # reconvert the mask to the BGR color space and invert it to fill the original red eye hole
        mask_inverse = ~cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        eye = cv2.bitwise_and(mask_inverse, eye_section) + blue_green_mean

        fixed_image[y:y + h, x:x + w] = eye

    return fixed_image


def configure_mean(blue_green_sum, mask):
    """
    Helper function for adjusting the blue-green color mean for blending the old red eye
    """
    blue_green_mean = blue_green_sum / 2
    blue_green_mean = cv2.bitwise_and(blue_green_mean, mask)
    blue_green_mean = cv2.cvtColor(blue_green_mean, cv2.COLOR_GRAY2BGR)
    return blue_green_mean


def resetMask(current_mask):
    """
    Resets the mask to its default value
    """
    return current_mask * 0


def get_largest_contour(contours):
    """
    In the given images, there may be various sections picked up by the detection algorithm,
    this chooses the largest section to correct, as it is most likely the actual red eye
    """
    largest_contour_area = 0
    largest_contour = None

    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area > largest_contour_area:
            largest_contour_area = contour_area
            largest_contour = contour

    return largest_contour




