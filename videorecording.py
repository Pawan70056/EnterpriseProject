import cv2
import os
import datetime

frameWidth = 640 
frameHeight = 480 
fps = 20.0  # Define the frames per second

cap = cv2.VideoCapture(0) 
cap.set(4, frameHeight)
cap.set(3, frameWidth)
cap.set(10, 150)

# Set the save directory path 
save_directory = "C:/Users/ssmp7/Desktop/Enterprise_Project/video"
# Create the directory if it doesn't exist 
if not os.path.exists(save_directory): 
    os.makedirs(save_directory)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
video_path = os.path.join(save_directory, f"captured_video_{timestamp}.avi")
out = cv2.VideoWriter(video_path, fourcc, fps, (frameWidth, frameHeight))

# Calculate the end time for 10 seconds from the start
end_time = cv2.getTickCount() + 10 * cv2.getTickFrequency()

while True:
    success, img = cap.read()
    if not success:
        break
    cv2.imshow("Result", img)

    # Write the frame to the video file
    out.write(img)

    # Check if the current time has exceeded the end time
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getTickCount() > end_time:
        break

# Release the VideoCapture and VideoWriter objects
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video Saved at {video_path}")
