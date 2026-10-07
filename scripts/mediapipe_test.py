import cv2
import mediapipe as mp

hands = mp.solutions.hands.Hands(max_num_hands=1, model_complexity=0)
drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if result.multi_hand_landmarks:
        drawing.draw_landmarks(frame, result.multi_hand_landmarks[0],
                               mp.solutions.hands.HAND_CONNECTIONS)

    cv2.imshow("MediaPipe", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
