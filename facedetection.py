import cv2
import numpy as np

# Load YOLOv5 model for face detection
net = cv2.dnn.readNet("yolov5s.weights", "yolov5s.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Load face recognition model for facial reactions
face_recognition_model = cv2.dnn.readNetFromCaffe("face_recognition.prototxt", "face_recognition.caffemodel")

# Load age and sex estimation model
age_sex_model = cv2.dnn.readNetFromCaffe("age_sex.prototxt", "age_sex.caffemodel")

# Load video capture device
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces using YOLOv5
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    faces = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # 0 is the class ID for faces
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Extract face ROI
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                face_roi = frame[y:y + h, x:x + w]

                # Recognize facial reaction
                face_recognition_blob = cv2.dnn.blobFromImage(face_roi, 1, (224, 224), (0, 0, 0), True, crop=False)
                face_recognition_model.setInput(face_recognition_blob)
                face_recognition_out = face_recognition_model.forward()
                facial_reaction = np.argmax(face_recognition_out)

                # Estimate age and sex
                age_sex_blob = cv2.dnn.blobFromImage(face_roi, 1, (224, 224), (0, 0, 0), True, crop=False)
                age_sex_model.setInput(age_sex_blob)
                age_sex_out = age_sex_model.forward()
                age = int(age_sex_out[0][0] * 100)
                sex = np.argmax(age_sex_out[1])

                # Draw bounding box and display facial reaction, age, and sex
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Facial Reaction: {facial_reaction}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, f"Age: {age}", (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, f"Sex: {sex}", (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
