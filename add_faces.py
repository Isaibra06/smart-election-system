import os
import pickle
import numpy as np
import cv2

# Check whether 'data/' directory exists, if not, create it
if not os.path.exists('data/'):
    os.makedirs('data/')

# Setting up the primary camera
video = cv2.VideoCapture(0)

# Load the face detection cascade classifier
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize faces_data to store images
faces_data = []

# Initialize frame counter
i = 0
name = input("Enter your number:")  # Get the user's name/ID for labeling the images

# Set the total number of frames to capture
framesTotal = 51

# Set how many frames to skip before capturing the next one
captureAfterFrames = 2

# Start reading the video stream
while True:
    ret, frame = video.read()
    
    if not ret:
        break  # Exit if the frame capture fails

    # Convert the frame to grayscale (this is for face detection)
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Use the face detection module to detect faces
    faces = facedetect.detectMultiScale(grey, 1.3, 5)

    # Draw a box around the face and capture the face region
    for (x, y, w, h) in faces:
        crop_img = frame[y:y + h, x:x + w]
        resized_img = cv2.resize(crop_img, (50, 50))  # Resize face image to 50x50
        
        # Append the face image to the faces_data list after every 'captureAfterFrames' frames
        if len(faces_data) < framesTotal and i % captureAfterFrames == 0:
            faces_data.append(resized_img)  # Append face to the list
        
        i += 1  # Increment frame counter
        cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 0)  # Show the number of faces captured
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 0)  # Draw rectangle around face

    # Display the frame
    cv2.imshow('frame', frame)

    # Check if 'q' key is pressed or the required number of frames have been captured
    k = cv2.waitKey(1)
    if k == ord('q') or len(faces_data) >= framesTotal:
        break

# Release the video capture and close all OpenCV windows
video.release()
cv2.destroyAllWindows()

# Print the number of faces captured
print(len(faces_data))

# Convert the faces_data list to a numpy array and reshape it for saving
faces_data = np.array(faces_data)

# Reshape to (framesTotal, 50*50*3) for easier saving
faces_data = faces_data.reshape(framesTotal, -1)

print(faces_data)

# Handle 'names.pkl' file (Check for empty or non-existent file)
if 'names.pkl' not in os.listdir('data/') or os.stat('data/names.pkl').st_size == 0:
    names = [name] * framesTotal
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)
else:
    with open('data/names.pkl', 'rb') as f:  # Open in 'rb' mode to read
        try:
            names = pickle.load(f)  # Attempt to load existing names
        except EOFError:  # Handle case where file is empty
            names = []  # Initialize as an empty list if EOFError is raised
    names = names + [name] * framesTotal  # Append the new names
    with open('data/names.pkl', 'wb') as f:  # Open in 'wb' mode to write
        pickle.dump(names, f)

# Handle 'faces_data.pkl' file (Check for empty or non-existent file)
if 'faces_data.pkl' not in os.listdir('data/') or os.stat('data/faces_data.pkl').st_size == 0:
    with open('data/faces_data.pkl', 'wb') as f:  # Open in 'wb' mode to write new data
        pickle.dump(faces_data, f)
else:
    with open('data/faces_data.pkl', 'rb') as f:  # Open in 'rb' mode to read existing data
        try:
            faces = pickle.load(f)  # Attempt to load existing faces data
        except EOFError:  # Handle case where file is empty
            faces = np.array([])  # Initialize as an empty numpy array if EOFError is raised
    faces = np.append(faces, faces_data, axis=0)  # Append new faces data
    with open('data/faces_data.pkl', 'wb') as f:  # Open in 'wb' mode to overwrite the file
        pickle.dump(faces, f)
