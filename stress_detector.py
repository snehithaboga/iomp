import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import pyttsx3
from plyer import notification
import matplotlib.pyplot as plt
from collections import Counter
import threading
import time
import os
import tkinter as tk
model = load_model("fer2013_model.h5")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
engine = pyttsx3.init()
exit_button_width = 150
exit_button_height = 50

pickup_lines = {
    'Angry': "Your anger makes thunder look soft... but I bet I can handle your storm!",
    'Disgust': "Even with that look, you're still charming! You're like a fine wine... with a twist of lemon.",
    'Fear': "No fear, I’m here for you. Just don’t be afraid to smile... it won’t bite, I promise!",
    'Happy': "Your smile could light up the city! Seriously, I think I need sunglasses just to look at you.",
    'Sad': "If tears were diamonds, you'd be rich by now. But hey, don't worry, you’ve got a treasure chest of happiness ahead!",
    'Surprise': "That surprised look suits you! You should wear it more often – it’s like your new superpower.",
    'Neutral': "Mysterious and composed – love that vibe. You're like a cool breeze on a hot day... seriously, I could use that right now."
}

chatbot_lines = {
    'Angry': "Hey, I can sense some anger. Let's take a deep breath together!",
    'Disgust': "Hmm, looks like something's bothering you. Let's talk about it.",
    'Fear': "You're safe now, don't worry.",
    'Happy': "I'm happy to see you happy! Keep it going!",
    'Sad': "I understand you're feeling down. Let's cheer you up.",
    'Surprise': "That caught you off guard, huh?",
    'Neutral': "You seem calm. That's great!"
}
def speak(text):
    engine.say(text)
    engine.runAndWait()
def detect_stress(emotion):
    suggestion = chatbot_lines.get(emotion, "You're doing great, keep going!")
    speak(suggestion)
def show_notification(emotion):
    line = pickup_lines.get(emotion, "Keep smiling!")
    try:
        notification.notify(
            title=f"Emotion: {emotion}",
            message=line,
            timeout=3
        )
    except:
        print(f"Notification failed for {emotion}")
emotion_history = []
running = True
def create_exit_gui():
    def exit_app():
        global running
        running = False
        speak("Exiting the application now!")
        root.destroy()

    root = tk.Tk()
    root.title("Exit Panel")
    root.geometry("200x100")
    tk.Button(root, text="Exit App", command=exit_app, bg="red", fg="white", font=("Arial", 14)).pack(expand=True)
    root.mainloop()
def plot_emotion_graph():
    plt.ion()
    fig, ax = plt.subplots()
    while running:
        if emotion_history:
            ax.clear()
            counts = Counter(emotion_history[-30:])
            ax.bar(counts.keys(), counts.values(), color='skyblue')
            ax.set_title('Emotion Trend')
            ax.set_ylabel('Count')
            plt.pause(2)
    plt.close('all')
threading.Thread(target=create_exit_gui, daemon=True).start()
threading.Thread(target=plot_emotion_graph, daemon=True).start()
cap = cv2.VideoCapture(0)
cv2.namedWindow("Emotion Detection")
prev_emotion = None
last_spoken_time = time.time()
last_emotion_time = time.time()
while running:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        if time.time() - last_spoken_time > 3:
            speak("Umm! No face detected!")
            last_spoken_time = time.time()

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (48, 48))
        face = face.astype('float32') / 255.0
        face = np.expand_dims(face, axis=-1)
        face = np.expand_dims(face, axis=0)
        emotion_probs = model.predict(face, verbose=0)
        max_index = np.argmax(emotion_probs[0])
        predicted_emotion = emotion_labels[max_index]

        if predicted_emotion != prev_emotion or (time.time() - last_emotion_time > 5):
            emotion_history.append(predicted_emotion)
            detect_stress(predicted_emotion)
            show_notification(predicted_emotion)
            prev_emotion = predicted_emotion
            last_emotion_time = time.time()

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 100, 100), 2)
        cv2.putText(frame, f"{predicted_emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"{pickup_lines.get(predicted_emotion)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Emotion Detection & Chatbot", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False
cap.release()
cv2.destroyAllWindows()
time.sleep(1)
def send_final_mood_notification():
    final_counts = Counter(emotion_history)
    positive_emotions = final_counts['Happy'] + final_counts['Neutral']
    negative_emotions = sum(final_counts[emo] for emo in emotion_labels if emo not in ['Happy', 'Neutral'])

    if positive_emotions >= negative_emotions:
        mood_message = "The person seems to be in a good and calm mood! 😊"
    else:
        mood_message = "The person appears stressed or tensed. Consider taking a break or relaxing. 😟"

    try:
        notification.notify(
            title="🧠 Final Mood Summary",
            message=mood_message,
            timeout=6
        )
    except:
        print("Final mood notification failed.")

    print("📊 Final Mood Notification Sent.")
    speak(mood_message)
send_final_mood_notification()