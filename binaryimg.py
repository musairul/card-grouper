import cv2
import numpy as np

# Start video capture from the default camera (0)
cap = cv2.VideoCapture(1)

while True:
    # Read a frame from the video feed
    ret, frame = cap.read()
    if not ret:
        break

    # Define the range for white color in BGR
    lower_white = np.array([200, 200, 200])  # Lower bound for white
    upper_white = np.array([255, 255, 255])  # Upper bound for white

    # Create a mask that identifies white pixels
    mask = cv2.inRange(frame, lower_white, upper_white)

    # Create a black image to hold the filtered result
    filtered_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # Show the original and filtered frames
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Filtered Frame', filtered_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
