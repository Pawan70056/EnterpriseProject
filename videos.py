import cv2
import os
import datetime

frameWidth = 640 
frameHeight = 480 
cap = cv2.VideoCapture (0) 
cap.set(4, frameHeight)
cap.set(3, frameWidth)
cap.set(2, 10)
# Calculate the time when the window should be closed (1 minute = 60 seconds) 
end_time = cv2.getTickCount() + 60* cv2.getTickFrequency()
while True:
    success, img = cap.read()
    cv2.imshow("Result", img)
    #Check if the current time has exceede the end time
    if cv2.waitKey(1) & 0xFF==ord("q") or cv2.getTickCount() > end_time:
        break
cap.release()
cv2.destroyAllWindows()