# push_ups_counter.py

import cv2
import mediapipe as mp
import sys

try:
    print("Initializing camera...")
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Could not open camera. Please check if camera is connected.")

    count = 0
    position = None

    def detect_push_up(landmarks):
        global position, count
        shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        distance = abs(shoulder.y - hip.y)

        if distance < 0.10 and position != "down":
            position = "down"
        elif distance > 0.20 and position == "down":
            position = "up"
            count += 1
            print(f"Push-up count: {count}")

    print("Camera initialized. Starting push-up detection...")
    print("Press ESC to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Flip frame horizontally for a selfie-view display
        frame = cv2.flip(frame, 1)
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(image_rgb)

        if result.pose_landmarks:
            mp_draw.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            detect_push_up(result.pose_landmarks.landmark)

        cv2.putText(frame, f'Push-ups: {count}', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Push-Up Counter", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            print("Exiting...")
            break

except Exception as e:
    print(f"Error: {str(e)}")
    sys.exit(1)

finally:
    if 'cap' in locals():
        cap.release()
    cv2.destroyAllWindows()
    print("Cleanup complete")
