import cv2
import matplotlib.pyplot as plt

config_file = r"C:\Users\ssmp7\Desktop\Enterprise_Project\models\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
frozen_model = r"C:\Users\ssmp7\Desktop\Enterprise_Project\models\frozen_inference_graph.pb"

model = cv2.dnn_DetectionModel( config_file, frozen_model)

model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean(127.5, 127.5, 127.5)
model.setInputSwapRB(True)

#Load the class Labels
# calssLabels = []
# with open('coco.names', 'r') as f:
#     classLables = f.read().splitlines

cap = cv2.VideoCapture(0)
font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN

while True:
    ret, frame = cap.read()
    ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.55)
    print(ClassIndex)
    if len(ClassIndex) != 0:
        for ClassInd, conf, boxes in zip (ClassIndex.flatten(), confidence.flatten(), bbox):
            if ClassInd <= 80:
                cv2.rectangle(frame,boxes, (225, 0, 0), 2) # Draw bounding box only
    
    #DIsplay the frame with object detection
    cv2.imshow('Object Detection', frame)
    if cv2.waitkey(2) & 0xff == ord('q'):
        break

    cap.release()
    cv2.destroyAllWindows()