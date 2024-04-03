import os
import cv2

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def capture_face_images(output_directory, cascade_path, user_id):
    vid_cam = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cascade_path)
    count = 0
    check_path(output_directory)
    while True:
        _, image_frame = vid_cam.read()
        gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.4, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(image_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1
            image_path = os.path.join(output_directory, f'User.{str(user_id)}.{str(count)}.jpg')
            cv2.imwrite(image_path, gray[y:y + h, x:x + w])
            cv2.imshow('Creating Dataset', image_frame)
        if cv2.waitKey(100) & 0xFF == 27:
            break
        elif count >= 100:
            vid_cam.release()
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    user_id = 1  # Update this ID for each user
    dataset_path = r"C:\Users\HOME\Desktop\Projects\IOMP\new\datasets"
    cascade_classifier_path = r"C:\Users\HOME\Desktop\Projects\IOMP\new\haarcascade_frontalface_default.xml"
    capture_face_images(dataset_path, cascade_classifier_path, user_id)