import cv2
import os
import numpy as np
from PIL import Image
def assure_path_exists(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
def train_face_recognition_model(dataset_path, trainer_path):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(r"C:\Users\HOME\Desktop\Projects\IOMP\new\haarcascade_frontalface_default.xml")
    def get_images_and_labels(path):
        image_paths = [os.path.join(path, f) for f in os.listdir(path)]
        face_samples = []
        ids = []
        for image_path in image_paths:
            PIL_img = Image.open(image_path).convert('L')
            img_numpy = np.array(PIL_img, 'uint8')
            user_id = int(os.path.split(image_path)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)        
            for (x, y, w, h) in faces:
                face_samples.append(img_numpy[y:y + h, x:x + w])
                ids.append(user_id)
        return face_samples, ids
    assure_path_exists(trainer_path)
    faces, ids = get_images_and_labels(dataset_path)
    recognizer.train(faces, np.array(ids))
    model_path = os.path.join(trainer_path, "trainer.yml")
    recognizer.save(model_path)

if __name__ == "__main__":
    dataset_path = r"C:\Users\HOME\Desktop\Projects\IOMP\new\datasets"
    trainer_path = r"C:\Users\HOME\Desktop\Projects\IOMP\new\trainer"

    train_face_recognition_model(dataset_path, trainer_path)
