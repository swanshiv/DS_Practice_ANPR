import numpy as np
import pytesseract
import imutils
import cv2
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files (x86)/Tesseract-OCR/tesseract.exe"

# Read the image file
image = cv2.imread('Resource/Car_Images/5.jpg')

image = imutils.resize(image, width=500) # Resize the image - change width to 500
cv2.imshow("Original Image", image) # Display the original image
cv2.waitKey(0)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # RGB to Gray scale conversion
cv2.imshow("1 - Grayscale Conversion", gray) # Display the image converted to Grayscale
cv2.waitKey(0)

# Noise removal with iterative bilateral filter(removes noise while preserving edges)
gray = cv2.bilateralFilter(gray, 11, 17, 17)
cv2.imshow("2 - Bilateral Filter", gray)
cv2.waitKey(0)

# Find Edges of the grayscale image
edged = cv2.Canny(gray, 170, 200)
cv2.imshow("3 - Canny Edges", edged)
cv2.waitKey(0)

# Find contours based on Edges
cnts, new  = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
img1 = image.copy() # Create copy of original image to draw all contours
cv2.drawContours(img1, cnts, -1, (0,255,0), 3)
cv2.imshow("4- All Contours", img1)
cv2.waitKey(0)

#sort contours based on their area keeping minimum required area as '30' (anything smaller than this will not be considered)
cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:30]
NumberPlateCnt = None #we currently have no Number plate contour

# Top 30 Contours
img2 = image.copy()
cv2.drawContours(img2, cnts, -1, (0,255,0), 3)
cv2.imshow("5- Top 30 Contours", img2)
cv2.waitKey(0)

# Code for PC Session 76


# loop over our contours to find the best possible approximate contour of number plate
count = 0
idx =7
for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # print ("approx = ",approx)
        if len(approx) == 4:  # Select the contour with 4 corners
            NumberPlateCnt = approx #This is our approx Number Plate Contour

            # Crop those contours and store it in Cropped Images folder
            x, y, w, h = cv2.boundingRect(c) #This will find out co-ord for plate
            new_img = gray[y:y + h, x:x + w] #Create new image
            cv2.imwrite('Resource/Cropped_Images/' + str(idx) + '.png', new_img) #Store new image
            idx+=1

            break


# Drawing the selected contour on the original image
#print(NumberPlateCnt)
cv2.drawContours(image, [NumberPlateCnt], -1, (0,255,0), 3)
cv2.imshow("Final Image With Number Plate Detected", image)
cv2.waitKey(0)

Cropped_img_loc = 'Resource/Cropped_Images/7.png'
cv2.imshow("Cropped Image ", cv2.imread(Cropped_img_loc))

# Use tesseract to covert image into string
text = pytesseract.image_to_string(Cropped_img_loc, lang='eng')
print("Number is :", text)

cv2.waitKey(0) #Wait for user input before closing the images displayed
