import cv2
import sys
import numpy as np
import pyautogui
import os
import datetime
import time
import subprocess

def get_trainer_path():
    #Return the path to the trainer.yml file.
    trainer_path = r"C:\Users\HOME\Desktop\Projects\IOMP\new\trainer\trainer.yml"
    return trainer_path

def verify_user_face(gray_frame, confidence_threshold=70):
    #Verify if the user's face is recognized in the given frame.
    """    
    Args:
        gray_frame (numpy.ndarray): Grayscale frame containing a face.
        confidence_threshold (int): Confidence threshold for face recognition.
    Returns:
        bool: True if the user is recognized, False otherwise.
    """
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    trainer_path = get_trainer_path()
    recognizer.read(trainer_path)

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        label, confidence = recognizer.predict(gray_frame[y:y + h, x:x + w])
        if confidence < confidence_threshold:
            return True  # User recognized
        
    return False  # User not recognized

def perform_recognition_actions(gray_frame):
    recognized = verify_user_face(gray_frame, confidence_threshold=70)  # Check face recognition
    if recognized:
        print("User recognized")
        # Execute mouse-cursor-control.py using subprocess
        subprocess.run(["python", "mouse-cursor-control.py"], check=True)

if __name__ == "__main__":
    cam = cv2.VideoCapture(0)

    while True:
        ret, frame = cam.read()
        
        if not ret:
            print("Error: Unable to capture frame")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        perform_recognition_actions(gray)
        
        cv2.imshow('Face Recognition', frame)
        key = cv2.waitKey(1)
        
        if key == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

