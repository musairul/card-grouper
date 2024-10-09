import cv2 as cv
import numpy as np

BKG_THRESH = 60  # Make it higher for bright lighting
CARD_MAX_AREA = 120000
CARD_MIN_AREA = 25000
GROUP_DISTANCE_THRESH = 100  # Define a threshold for group distance

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

    valid_contours = []
    
    # Filter contours that are within the area range
    for contour in contours:
        area = cv.contourArea(contour)
        if CARD_MIN_AREA < area < CARD_MAX_AREA:
            valid_contours.append(contour)
    
    # Sort the contours from left to right by x-coordinate
    bounding_boxes = [cv.boundingRect(c) for c in valid_contours]
    sorted_contours_with_boxes = sorted(zip(valid_contours, bounding_boxes), key=lambda b: b[1][0])

    groups = []
    current_group = []
    prev_x = None
    
    # Group contours based on the distance between their bounding boxes
    for contour, bbox in sorted_contours_with_boxes:
        x, y, w, h = bbox
        if prev_x is None:
            # Start the first group
            current_group.append((contour, bbox))
        else:
            # Calculate the distance between this contour's left edge and the previous contour's right edge
            distance = x - prev_x
            if distance < GROUP_DISTANCE_THRESH:
                current_group.append((contour, bbox))
            else:
                # If the distance is too large, save the current group and start a new group
                groups.append(current_group)
                current_group = [(contour, bbox)]
        
        # Update previous x-coordinate (right edge of current bounding box)
        prev_x = x + w
    
    # Add the last group
    if current_group:
        groups.append(current_group)
    
    # Draw individual contours
    for contour, bbox in sorted_contours_with_boxes:
        cv.drawContours(frame, [contour], -1, (0, 255, 0), 2)  # Draw green contours

    # Draw bounding boxes for each group
    for group in groups:
        x_min = min([bbox[0] for _, bbox in group])
        y_min = min([bbox[1] for _, bbox in group])
        x_max = max([bbox[0] + bbox[2] for _, bbox in group])
        y_max = max([bbox[1] + bbox[3] for _, bbox in group])

        # Draw a bounding box around the group
        cv.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)  # Blue bounding box for groups

    # Show the original video frame with contours and group bounding boxes
    cv.imshow('Video with Contours and Groups', frame)

capture.release()
cv.destroyAllWindows()
