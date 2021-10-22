


import numpy as np
import cv2
from PIL import Image
import scipy.misc





# load the image
st = cv2.imread("test1.png")
image = np.float32(st)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


# define the color bounds, try out several different color bounds depending on image brightneess
blue = [
	#([86, 31, 4], [220, 88, 50])
	([112, 119, 88], [117, 206, 253])
	 
    
]
red = [
    #([0, 20, 100], [68, 70, 220])
	#([0, 135, 65], [179, 233, 245])
	([0, 39, 24], [23, 255, 187])

]
green = [
   #([0, 45, 0], [65, 255, 65])
	([53, 124, 124], [179, 213, 255])
]

white = [
    ([0, 0, 96], [255, 255, 255])
]

black = [
    #([0, 0, 0], [50, 50, 50])
	([0, 0, 0], [0, 90, 128])
]


def dec(color):
	for (lower, upper) in color:
		# create NumPy arrays from the boundaries 
		lower = np.array(lower, dtype = "uint8")
		upper = np.array(upper, dtype = "uint8")

		
		mask = cv2.inRange(hsv, lower, upper)
	
		cv2.imshow("out2", image)
		cv2.imshow("out3", mask)

		return(mask)




bluearea=dec(blue)
greenarea=dec(green)
redarea=dec(red)
whitearea=dec(white)
blackarea=dec(black)
	
ba = np.sum(bluearea) *3
ga = np.sum(greenarea)*3
ra = np.sum(redarea)*3
wa = np.sum(whitearea) *.80
bla = np.sum(blackarea)
print(ba, ra, ga, wa, bla)



print(image.dtype)

if (ba >= ra) and (ba >= ga) and (ba >= wa) and (ba >= bla): 
	maincolor = ("blue")
	cv2.imshow("out3", bluearea)
	
if (ra >= ba) and (ra >= ga) and (ra >= wa) and (ra >= bla): 
	maincolor = ("red")
	cv2.imshow("out3", redarea)
	
if (ga >= ba) and (ga >= ra) and (ga >= wa) and (ga >= bla): 
	maincolor = ("green")
	cv2.imshow("out3", greenarea)

if (wa >= ba) and (wa >= ga) and (wa >= ra) and (wa >= bla): 
	maincolor = ("white")
	cv2.imshow("out3", whitearea)
	
if (bla >= ba) and (bla >= ra) and (bla >= wa) and (bla >= ga): 
	maincolor = ("black")
	cv2.imshow("out3", blackarea)
	
 

print(maincolor)

cv2.waitKey(0)