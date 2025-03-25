# Import necessary libraries
import cv2
import os
import csv
import time
import numpy as np
import pickle
from datetime import datetime
from win32com.client import Dispatch  # For text-to-speech functionality
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk  # Correct import for Image and ImageTk

# Function to speak a string using text-to-speech (SAPI)
def speak(str):
    speak = Dispatch(("SAPI.Spvoice"))  # Initialize speech engine
    speak.Speak(str)  # Convert text to speech and speak it out loud

# Initialize the webcam video capture
video = cv2.VideoCapture(0)

# Load face detection classifier (Haar Cascade)
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the labels (names) and face data (images) that were previously saved in pickle files
with open('data/names.pkl', 'rb') as f:
    LABELS = pickle.load(f)  # Loaded labels (e.g., names of people)

with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)  # Loaded corresponding face images

# Initialize the KNeighborsClassifier (KNN)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)  # KNN with 5 neighbors
knn.fit(FACES, LABELS)  # Train the KNN classifier with face data and their corresponding labels

# Define the column names for the attendance CSV
COL_NAMES = ['NAME', 'VOTE', 'DATE', 'TIME']

# Function to check if the voter has already voted
def check_if_exists(value):
    try:
        with open('Votes.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            # Loop through the rows and check if the name already exists in the CSV file
            for row in reader:
                if row[0] == value:  # If the name exists in the CSV file
                    return True
    except FileNotFoundError:
        print("File not found or unable to open the CSV file.")
    return False

# Tkinter GUI setup
root = tk.Tk()
root.title("Election Voting System")
root.geometry("800x600")  # Adjust size of window

# Create a frame for displaying video and buttons
frame_video = tk.Frame(root)
frame_video.pack()

# Create labels for candidates
candidates = ['Guild President', 'Guild Vice President', 'Guild Secretary', 'Cultural Secretary']

# Create a label to show detected voter
voter_label = tk.Label(root, text="Voter: ", font=("Arial", 14))
voter_label.pack(pady=10)

# Function to record vote
def record_vote(candidate, voter_name):
    # Get current date and time
    ts = time.time()
    date = datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
    time_str = time.strftime("%H:%M:%S")

    # Check if the CSV file already exists
    exist = os.path.isfile('Votes.csv')

    # Record the vote into CSV
    if exist:
        with open('Votes.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            attendance = [voter_name, candidate, date, time_str]
            writer.writerow(attendance)
    else:
        with open('Votes.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(COL_NAMES)  # Write column names (first time creating the file)
            attendance = [voter_name, candidate, date, time_str]
            writer.writerow(attendance)

    speak(f"YOUR VOTE FOR {candidate} HAS BEEN RECORDED")
    messagebox.showinfo("Vote Recorded", f"Thank you for voting for {candidate}!")

# Function to handle voting
def vote_for(candidate):
    global output_name  # Global variable for the detected name
    if output_name:  # Ensure that a valid name is detected
        voter_exist = check_if_exists(output_name)
        if voter_exist:
            speak("YOU HAVE ALREADY VOTED")
            messagebox.showwarning("Already Voted", "You have already voted!")
        else:
            record_vote(candidate, output_name)  # Record the vote for the selected candidate

# Create buttons for each candidate
for candidate in candidates:
    button = tk.Button(root, text=candidate, font=("Arial", 14), width=20, height=2,
                       command=lambda c=candidate: vote_for(c))
    button.pack(pady=5)

# Function to update the webcam feed in the Tkinter window
def update_frame():
    ret, frame = video.read()  # Capture a new frame from the webcam

    # Convert the frame to grayscale (required for face detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    global output_name
    output_name = None  # Reset the name each time

    # Loop through all detected faces
    for (x, y, w, h) in faces:
        # Crop the image to focus only on the face region
        crop_img = frame[y:y + h, x:x + w]

        # Resize the cropped face image to 50x50 pixels to match the model input size
        resized_img = cv2.resize(crop_img, (50, 50))

        # Flatten the resized image (50x50x3) into a 1D array (7500 elements)
        resized_img_flattened = resized_img.flatten()

        # Use the KNN model to predict the label (person) for the face
        output = knn.predict([resized_img_flattened])  # Make the prediction

        output_name = output[0]  # Get the name of the person detected

        # Draw rectangles around the detected face on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
        cv2.putText(frame, output_name, (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 225, 255), 1)

        # Update the voter label with the detected name
        voter_label.config(text=f"Voter: {output_name}")

    # Convert the frame from OpenCV to Tkinter-compatible format (PIL)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))  # Use ImageTk.PhotoImage


    # Display the webcam feed in the Tkinter window
    label_video = tk.Label(frame_video, image=img)
    label_video.img = img  # Keep a reference to avoid garbage collection
    label_video.grid(row=0, column=0)

    # Update the frame every 10 milliseconds
    root.after(10, update_frame)

# Start updating the frame
update_frame()

# Start the Tkinter event loop
root.mainloop()

# Release the video capture after closing the GUI
video.release()

cv2.destroyAllWindows()
