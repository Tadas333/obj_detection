# USAGE
# python detect_color.py --image pokemon_games.png

# import the necessary packages
import numpy as np
import argparse
import cv2

# construct the argument parse and parse the arguments


# load the image
image = cv2.imread("pic1.jpeg")

# define the list of boundaries
blue = [
	
	([250, 250, 250], [255, 255, 255])
	
]

# loop over the boundaries
for (lower, upper) in blue:
	# create NumPy arrays from the boundaries
	lower = np.array(lower, dtype = "uint8")
	upper = np.array(upper, dtype = "uint8")

	# find the colors within the specified boundaries and apply
	# the mask
	mask = cv2.inRange(image, lower, upper)
	output = cv2.bitwise_and(image, image, mask = mask)
	ret,thresh=cv2.threshold(output,0,155,cv2.THRESH_BINARY_INV)
 	
	 
	cv2.imshow("output",thresh)

	bluearea = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

	print(cv2.countNonZero(bluearea))
	# show the images
	cv2.imshow("images", np.hstack([image, output]))
	cv2.waitKey(0)