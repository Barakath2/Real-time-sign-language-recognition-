import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from tamil_translator import translate_to_tamil
from voice_output import speak_tamil
import time

# Load model and labels
model = load_model(r"/Path-sign_cnn_model.h5")
with open(r"/Path-gesture_labels.txt", "r") as f:
    labels = [line.strip() for line in f]

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Webcam and buffers
cap = cv2.VideoCapture(0)
sentence = ""
current_char = ""
char_buffer = []
buffer_size = 20
confidence_threshold = 0.85
last_added_time = time.time()

def preprocess_image(image):
    image = cv2.resize(image, (28, 28))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.reshape(1, 28, 28, 1).astype("float32") / 255.0
    return image

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_list = [lm.x for lm in hand_landmarks.landmark]
            y_list = [lm.y for lm in hand_landmarks.landmark]
            h, w, _ = frame.shape
            x_min, x_max = int(min(x_list) * w), int(max(x_list) * w)
            y_min, y_max = int(min(y_list) * h), int(max(y_list) * h)

            x_min, y_min = max(0, x_min - 20), max(0, y_min - 20)
            x_max, y_max = min(w, x_max + 20), min(h, y_max + 20)

            hand_img = frame[y_min:y_max, x_min:x_max]

            if hand_img.size > 0:
                try:
                    input_img = preprocess_image(hand_img)
                    prediction = model.predict(input_img)[0]
                    max_prob = np.max(prediction)
                    predicted_label = labels[np.argmax(prediction)]

                    if max_prob >= confidence_threshold:
                        char_buffer.append(predicted_label)

                        if len(char_buffer) >= buffer_size:
                            most_common = max(set(char_buffer), key=char_buffer.count)
                            current_time = time.time()

                            if most_common == current_char and current_time - last_added_time > 1.0:
                                sentence += most_common
                                last_added_time = current_time
                            elif most_common != current_char:
                                sentence += most_common
                                current_char = most_common
                                last_added_time = current_time

                            char_buffer.clear()

                    # Display prediction with confidence
                    cv2.putText(frame, f"{predicted_label} ({max_prob:.2f})", (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

                except Exception as e:
                    print("Prediction error:", e)

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display sentence
    cv2.putText(frame, f"Sentence: {sentence}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Tamil Sign Language Translator", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("c"):
        sentence = ""
        current_char = ""
    elif key == ord("s"):
        tamil = translate_to_tamil(sentence)
        speak_tamil(tamil)
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()
