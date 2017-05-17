import teeth_whitener
import cv2
import numpy as np

if __name__=="__main__":

	yellow_teeth_photo = cv2.imread("full_person_yellow_photo_2.jpg")

	fixed_image = teeth_whitener.whiten_teeth(yellow_teeth_photo)

	result = np.hstack((yellow_teeth_photo, fixed_image))

	cv2.imshow("Teeth Whitening", result)
	cv2.waitKey()
	cv2.destroyAllWindows()
