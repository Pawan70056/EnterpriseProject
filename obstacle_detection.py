import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Load the pre-trained SSD MobileNet V2 model from TensorFlow Hub
model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")

# Load the webcam feed
cap = cv2.VideoCapture(0)  # Change 0 to the index of your camera if necessary

# Function to perform object detection
def detect_objects(image):
    # Convert image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor(rgb_image)
    input_tensor = input_tensor[tf.newaxis, ...]
    
    # Perform object detection
    detections = model(input_tensor)
    return detections

# Function to visualize detections
def visualize_detections(image, detections, threshold=0.5):
    height, width, _ = image.shape
    detection_boxes = detections['detection_boxes'][0].numpy()
    detection_classes = detections['detection_classes'][0].numpy().astype(np.int32)
    detection_scores = detections['detection_scores'][0].numpy()

    for i in range(len(detection_boxes)):
        if detection_scores[i] >= threshold:
            box = detection_boxes[i]
            ymin, xmin, ymax, xmax = box
            xmin, xmax, ymin, ymax = int(xmin * width), int(xmax * width), int(ymin * height), int(ymax * height)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            label = f'Object {detection_classes[i]}: {detection_scores[i]:.2f}'
            cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

# Function to check for obstacles
def check_for_obstacles(detections, threshold=0.5):
    detection_scores = detections['detection_scores'][0].numpy()
    for score in detection_scores:
        if score >= threshold:
            return True
    return False

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects in the frame
    detections = detect_objects(frame)

    # Check for obstacles
    obstacle_detected = check_for_obstacles(detections)

    # Control the car (print statements for demonstration)
    if obstacle_detected:
        print("Obstacle detected! Stop the car.")
        # Add code to stop the car
    else:
        print("Path is clear. Move the car.")
        # Add code to move the car

    # Visualize detections
    result_frame = visualize_detections(frame, detections)

    # Display the result
    cv2.imshow('Object Detection', result_frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
