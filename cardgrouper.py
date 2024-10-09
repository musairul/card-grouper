import cv2 as cv
import numpy as np

BKG_THRESH = 100  # Make it higher for bright lighting

#playing card is normally 6.4cm x 8.9cm
#depending on distance of camera I assume this needs to be changed

CARD_MAX_AREA = 120000
CARD_MIN_AREA = 25000

capture = cv.VideoCapture(1)

while True:
    isTrue, frame = capture.read()
    
    if not isTrue or cv.waitKey(1) & 0xFF == ord('q'):
        break
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), cv.BORDER_DEFAULT)
    canny = cv.Canny(blurred, 125, 175)

    frame_w, frame_h = np.shape(canny)[:2]
    bkg_level = gray[int(frame_h / 100)][int(frame_w / 2)]
    thresh_level = bkg_level + BKG_THRESH

    retval, thresh = cv.threshold(blurred, thresh_level, 255, cv.THRESH_BINARY)

    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Draw contours that are within the specified area range
    for contour in contours:
        area = cv.contourArea(contour)
        if CARD_MIN_AREA < area < CARD_MAX_AREA:
            cv.drawContours(frame, [contour], -1, (0, 255, 0), 2)  # Draw the contour if it's in the area range

    # Show the original video frame with filtered contours
    cv.imshow('Video with Contours', frame)

capture.release()
cv.destroyAllWindows()
