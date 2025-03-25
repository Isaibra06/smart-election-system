import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import tkinter as tk
import threading

# Load the face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Prepare the data structures for face data and labels
faces_data = []
labels = []
label_id = 0  # To generate unique IDs for each face label

# Create KNeighborsClassifier instance
knn = KNeighborsClassifier(n_neighbors=5)
is_trained = False  # Flag to track if the classifier is trained

# Function to collect face data and train the KNN classifier
def collect_data_and_train():
    global faces_data, labels, label_id, is_trained

    # Open the webcam for face data collection
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 2, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face_img, (200, 200))
            flattened_face = face_resized.flatten()
            faces_data.append(flattened_face)
            labels.append(label_id)
            label_id += 1
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow("Face Data Collection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

    # Train the classifier after collecting data
    knn.fit(faces_data, labels)
    is_trained = True  # Set the flag to True after training
    print("Training complete")

# Function to recognize faces during login
def login_with_face():
    if not is_trained:
        print("Model is not trained yet!")
        return

    # Open the webcam again for login face scan
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 2, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face_img, (200, 200))
            flattened_face = face_resized.flatten()

            predicted_label = knn.predict([flattened_face])
            print(f"Predicted label: {predicted_label}")

            cv2.putText(frame, f"Label: {predicted_label}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow("Login Face Scan", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# Tkinter GUI
root = tk.Tk()
root.title("Login")
root.geometry("400x300")

# Ensure the Tkinter window is on top
root.lift()

# Username and Password Entry
username_label = tk.Label(root, text="Username:")
username_label.pack()
username_entry = tk.Entry(root)
username_entry.pack()

password_label = tk.Label(root, text="Password:")
password_label.pack()
password_entry = tk.Entry(root, show="*")
password_entry.pack()

# Login function (You can add logic here to verify credentials)
def login():
    username = username_entry.get()
    password = password_entry.get()
    # Add your username/password verification logic here
    print(f"Login attempt with username: {username} and password: {password}")

# Login Button
login_button = tk.Button(root, text="Login", command=login)
login_button.pack()

# Face Scan Button
def start_face_scan():
    # Run the face scan in a separate thread so the GUI remains responsive
    scan_thread = threading.Thread(target=login_with_face)
    scan_thread.daemon = True
    scan_thread.start()

scan_button = tk.Button(root, text="Scan Face to Login", command=start_face_scan)
scan_button.pack()

# Start the face data collection and training in a separate thread (this is for training phase)
def start_face_data_collection():
    data_thread = threading.Thread(target=collect_data_and_train)
    data_thread.daemon = True
    data_thread.start()

# Collect face data and train when the program starts
start_face_data_collection()

root.mainloop()
