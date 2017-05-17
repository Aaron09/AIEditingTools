import red_eye_remover
import cv2
import numpy as np

if __name__=="__main__":

	red_eye_image = cv2.imread("red_eye_photo.jpg")

	fixed_image = red_eye_remover.remove_red_eye(red_eye_image)

	result = np.hstack((red_eye_image, fixed_image))

	cv2.imshow("Red Eye Correction", result)
	cv2.waitKey()
	cv2.destroyAllWindows()
