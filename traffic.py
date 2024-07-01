import cv2
import numpy as np

def detect_traffic_light_color(image):
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color ranges for red, yellow, and green in HSV
    red_lower1 = np.array([0, 100, 100])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([160, 100, 100])
    red_upper2 = np.array([180, 255, 255])
    yellow_lower = np.array([15, 100, 100])
    yellow_upper = np.array([35, 255, 255])
    green_lower = np.array([40, 100, 100])
    green_upper = np.array([90, 255, 255])

    # Create masks for each color
    mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
    mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
    mask_green = cv2.inRange(hsv, green_lower, green_upper)

    # Find contours for each mask
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Function to check for significant contours
    def has_significant_contours(contours, area_threshold=500):
        for contour in contours:
            if cv2.contourArea(contour) > area_threshold:
                return True
        return False

    # Check for significant contours in each color mask
    if has_significant_contours(contours_red):
        return 'Red'
    elif has_significant_contours(contours_yellow):
        return 'Yellow'
    elif has_significant_contours(contours_green):
        return 'Green'
    else:
        return 'None'

def process_video(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect traffic light color in the frame
        color = detect_traffic_light_color(frame)

        # Display the detected color on the frame
        cv2.putText(frame, f'Traffic Light: {color}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show the frame
        cv2.imshow('Traffic Light Detection', frame)

        # Press 'q' to exit the video loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video_path = r"C:\Users\ssmp7\Desktop\Enterprise_Project\Traffic Light Detection using Opencv and YOLOv3 (Tutorial in description).mkv"  # Replace with your video path
    process_video(video_path)
