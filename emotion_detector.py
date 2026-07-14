import cv2
import numpy as np
import mediapipe as mp

# FIX: Force Python to use the Keras 2 legacy layer to prevent h5 dimension errors
try:
    from tf_keras.models import load_model
except ImportError:
    import os
    os.environ['TF_USE_LEGACY_KERAS'] = '1'
    from tensorflow.keras.models import load_model

# Load the pre-trained emotion recognition model safely
try:
    model = load_model('emotion_model.h5')
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize MediaPipe components
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("----EMOTION SENSOR IS LIVE-------; PRESS 'q' to exit")

with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detector:
    
    while True:
        ret, frame = camera.read()
        if not ret:
            print("Failed to grab camera frame.")
            break

        img_h, img_w, _ = frame.shape  
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detector.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(frame, detection)

                box = detection.location_data.relative_bounding_box

                # Convert percentages to pixel indices
                x = int(box.xmin * img_w)
                y = int(box.ymin * img_h)
                w = int(box.width * img_w)
                h = int(box.height * img_h)

                # Clamp index scopes to stay inside the screen dimensions
                x_start = max(0, min(x, img_w - 1))
                y_start = max(0, min(y, img_h - 1))
                x_end = max(0, min(x + w, img_w))
                y_end = max(0, min(y + h, img_h))

                cropped_face = frame[y_start:y_end, x_start:x_end]
                
                if cropped_face.size > 0:
                    gray_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY)
                    resized_face = cv2.resize(gray_face, (48, 48))

                    # Rescale values between 0.0 and 1.0
                    normal_face = resized_face / 255.0

                    # Standard batch array reshaping (batch, height, width, channels)
                    input_data = np.reshape(normal_face, (1, 48, 48, 1))
                    
                    predictions = model.predict(input_data, verbose=0)
                    max_index = np.argmax(predictions)
                    predicted_emotion = emotion_labels[max_index]

                    # Bound protection for the label rendering layout
                    text_y = max(20, y_start - 10)
                    cv2.putText(frame, predicted_emotion, (x_start, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    cv2.imshow("What AI sees", resized_face)

        cv2.imshow('LIVE FACE EMOTION TRACKER', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

camera.release()
cv2.destroyAllWindows()









#ANOTHER BLOCK TO TRY TOMORROW, UNDERSTAND.
'''
import sys
import types

# Fake the missing pkg_resources module so the 'fer' library stops crashing
if "pkg_resources" not in sys.modules:
    dummy_pkg = types.ModuleType("pkg_resources")
    def dummy_resource(package, resource):
        import os, fer
        return os.path.join(os.path.dirname(fer.__file__), resource)
    dummy_pkg.resource_filename = dummy_resource
    sys.modules["pkg_resources"] = dummy_pkg

# Your existing imports continue below...
from fer.fer import FER
import cv2


# 1. Initialize the Camera
camera = cv2.VideoCapture(0)
print("----EMOTION SENSOR IS LIVE----; PRESS 'q' to exit")

# 2. Initialize the AI Brain (This automatically handles the model download for you!)
# mtcnn=True makes it incredibly accurate at finding faces
detector = FER(mtcnn=True)

while True:
    # Snap a live picture frame from the camera
    ret, frame = camera.read()
    if not ret:
        print("Failed to grab frame from camera.")
        break

    # 3. THE BRAIN: Let the detector analyze the whole frame
    # It automatically finds the face, crops it, and guesses the emotion
    results = detector.detect_emotions(frame)

    for face in results:
        # Get the box coordinates around the face
        (x, y, w, h) = face["box"]
        
        # Draw a green rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Get the top winning emotion and its score
        emotions = face["emotions"]
        top_emotion = max(emotions, key=emotions.get)
        confidence = emotions[top_emotion]

        # Label the emotion on the video screen
        label = f"{top_emotion.capitalize()} ({int(confidence * 100)}%)"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 4. Show the MAIN camera video window
    cv2.imshow('LIVE FACE EMOTION TRACKER', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up windows
camera.release()
cv2.destroyAllWindows()'''
