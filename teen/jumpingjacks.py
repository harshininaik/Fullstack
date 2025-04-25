# jumping_jacks_counter.py

import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

count = 0
position = None

def detect_jumping_jack(landmarks):
    global position, count

    left_hand = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_hand = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    left_foot = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    right_foot = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

    hands_up = left_hand.y < 0.3 and right_hand.y < 0.3
    legs_apart = abs(left_foot.x - right_foot.x) > 0.5

    if hands_up and legs_apart and position != "open":
        position = "open"
    elif not hands_up and not legs_apart and position == "open":
        position = "close"
        count += 1
        print(f"Reps: {count}")

while True:
    success, frame = cap.read()
    if not success:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(image_rgb)

    if result.pose_landmarks:
        mp_draw.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        detect_jumping_jack(result.pose_landmarks.landmark)

    cv2.putText(frame, f'Reps: {count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Jumping Jacks Counter", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
