import cv2
import numpy as np

# Function to calculate distance between two points
def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

# Function to group cards by proximity
def group_cards(contours, distance_threshold=50):
    groups = []
    for contour in contours:
        if len(groups) == 0:
            groups.append([contour])
        else:
            added_to_group = False
            for group in groups:
                for member in group:
                    if calculate_distance(np.mean(contour, axis=0)[0], np.mean(member, axis=0)[0]) < distance_threshold:
                        group.append(contour)
                        added_to_group = True
                        break
                if added_to_group:
                    break
            if not added_to_group:
                groups.append([contour])
    return groups

# Function to draw bounding boxes around groups of contours
def draw_bounding_boxes(groups, frame):
    for group in groups:
        x_min = min([cv2.boundingRect(c)[0] for c in group])
        y_min = min([cv2.boundingRect(c)[1] for c in group])
        x_max = max([cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in group])
        y_max = max([cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in group])
        
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

# Capture video feed from the webcam
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours in the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours that are not likely to be cards
    card_contours = [c for c in contours if cv2.contourArea(c) > 1000]

    # Group the cards based on proximity
    card_groups = group_cards(card_contours)

    # Draw bounding boxes around each group
    draw_bounding_boxes(card_groups, frame)

    # Display the result
    cv2.imshow("Card Detection", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
