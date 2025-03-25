import tkinter as tk
import threading
from face_recognition import collect_data_and_train, login_with_face, is_trained  # Correct import

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

# Label for status
status_label = tk.Label(root, text="Model is training...")
status_label.pack()

# Face Scan Button
def start_face_scan():
    
    if not is_trained:
        status_label.config(text="Model is not trained yet! Please train it first.")
        return
    else:
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
