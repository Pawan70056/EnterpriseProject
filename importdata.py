import cv2
from IPython.display import display, Image
import PIL. Image
frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture("C:\\Users\\TULASI KATTEL\\Desktop\\Image_processing\\testing.mp4")
while True:
    success, img = cap.read()
    if not success:
        break
    img = cv2.resize(img, (frameWidth, frameHeight))
# Convert the image from OpenCV BGR format to RGB (which is the format expected by PIL) 
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# Convert the image to a PIL Image 
    pil_img = PIL.Image.fromarray(img_rgb)
# Display the image in the Jupyter Notebook 
    display(pil_img)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break
    cap.release() 
    cv2.destroyAllWindows()