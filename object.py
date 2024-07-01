import cv2
import face_recognition # type: ignore
import speech_recognition as sr # type: ignore
import time

# Hypothetical SDK import for robotic dog control
# from xgo_lite_sdk import XGOLite


# Initialize camera
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it
known_image = face_recognition.load_image_file("known_person.jpg")
known_face_encoding = face_recognition.face_encodings(known_image)[0]

known_face_encodings = [known_face_encoding]
known_face_names = ["Known Person"]

# Initialize robot (uncomment and replace with actual initialization if available)
# robot = XGOLite()


def recognize_face_and_voice():
    recognizer = sr.Recognizer()
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Find all the faces and face encodings in the frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = face_distances.argmin()
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            print(f"Detected: {name}")

            # Integrate with robot's behavior
            if name != "Unknown":
                print(f"Greeting {name}")
                # Replace with actual commands to control the robot
                # robot.move_forward()
                # robot.perform_greeting()

        # Display the resulting frame
        cv2.imshow('Video', frame)

        # Voice recognition
        with sr.Microphone() as source:
            print("Listening for voice commands...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
            try:
                command = recognizer.recognize_google(audio)
                print(f"Command: {command}")

                # Integrate with robot's behavior based on command
                if "forward" in command.lower():
                    print("Command: Move forward")
                    # Replace with actual command to move robot forward
                    # robot.move_forward()
                elif "stop" in command.lower():
                    print("Command: Stop")
                    # Replace with actual command to stop robot
                    # robot.stop()
                # Add more commands as needed

            except sr.UnknownValueError:
                print("Google Speech Recognition could not understand the audio")
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")

        # Hit 'q' on the keyboard to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "_main_":
    try:
        recognize_face_and_voice()
    except KeyboardInterrupt:
        print("Program stopped by User")
        video_capture.release()
        cv2.destroyAllWindows()