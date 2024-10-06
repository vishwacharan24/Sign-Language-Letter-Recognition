import cv2
import numpy as np
import mediapipe as mp
import keras
from keras.preprocessing.image import img_to_array


CNN = keras.models.load_model("model.h5")  # load model
out_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
              'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']  # labels


def detection(frame):
    h, w, c = frame.shape  # get shape of frame
    frame3 = frame.copy()  # duplicate frame
    # Detect hands in the image
    results = hands.process(frame)
    hand_detect = False  # flag
    if results.multi_hand_landmarks:  # if hand is detected
        hand_detect = True
        for hand_landmarks in results.multi_hand_landmarks:  # for each detected hand image
            # Calculate bounding box coordinates
            x_min, y_min = frame.shape[1], frame.shape[0]
            x_max, y_max = 0, 0
            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])

                # bounds
                if x < x_min:
                    x_min = x
                if x > x_max:
                    x_max = x
                if y < y_min:
                    y_min = y
                if y > y_max:
                    y_max = y

            # Add some padding to the bounding box to avoid loss of required pixels
            x_min -= 20
            y_min -= 20
            x_max += 20
            y_max += 20

            width = x_max - x_min  # width of the image
            height = y_max - y_min  # height of the image

            # choose a square to crop the image for prediction
            if width > height:
                avg_dif = int((width - height) / 2)
                y_max += avg_dif
                y_min -= avg_dif
            elif height > width:
                avg_dif = int((height - width) / 2)
                x_max += avg_dif
                x_min -= avg_dif

            # handle negative bounds
            if x_min < 0:
                x_min = 0
            if y_min < 0:
                y_min = 0

            frame3 = frame[y_min:y_max, x_min:x_max].copy()  # croo hand image
            # Draw rectangle around the bounding box
            cv2.rectangle(frame, (x_min, y_min),
                          (x_max, y_max), (0, 255, 0), 2)
            # mark landmarks in live feed
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            break

    if hand_detect:
        return [True, frame3]
    else:
        return [False, frame3]


def preprocess(frame):
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # grayscale image
    # gaussian blur to smoothen
    gray_image = cv2.GaussianBlur(gray_image, (7, 7), 0)
    # flip image to original for prediction
    gray_image = cv2.flip(gray_image, 1)
    # convert to a 28*28 image. Dataset images had same size
    gray_image = cv2.resize(gray_image, (28, 28))
    gray_image = gray_image.astype("float") / 255.0
    gray_image = img_to_array(gray_image)
    return gray_image


def predict(frame):
    frame = np.expand_dims(frame, axis=0)  # prepare image for prediction
    predicted_label = np.argmax(CNN.predict(frame))  # predict using model
    return out_labels[predicted_label]


# Load the MediaPipe Hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands()

# set up camera
vc = cv2.VideoCapture(0)

# While loop
while True:

    # Capture frame-by-frame
    ret, frame = vc.read()
    frame = cv2.flip(frame, 1)  # flip frame for display
    if not ret:
        print("Error")
        break

    # detect hand, draw a bounding box, mark landmarks and crop the hand image
    [y, frame2] = detection(frame)

    if y and frame2.any() != 0:
        gray_image = preprocess(frame2)  # preprocessing before prediction
        image = predict(gray_image)  # prediction
        font_style = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(frame, image, (10, 90), font_style, 5,
                    (0, 0, 255), 3)  # Annotate on live feed
    # Show the captured image
    cv2.imshow('SIGN-SAVVY', frame)

    # break loop on keyboard input
    if cv2.waitKey(1) == ord('q'):
        break

vc.release()
cv2.destroyAllWindows()
