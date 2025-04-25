import cv2
import mediapipe as mp
import numpy as np
import time

try:
    # Initialize MediaPipe Pose
    print("Initializing MediaPipe Pose...")
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    # Start video capture
    print("Starting video capture...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Could not open camera. Please check if camera is connected.")

    # Motion tracking variables
    prev_nose_y = None
    vertical_movement = 0
    frame_count = 0
    label = "Detecting..."
    start_time = time.time()

    def detect_cardio(activity_score, knee_angle):
        if activity_score > 20:
            return "Running"
        elif 60 < knee_angle < 110:
            return "Cycling"
        else:
            return "Standing"

    def get_angle(a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = abs(radians * 180 / np.pi)
        return angle if angle < 180 else 360 - angle

    print("Cardio detection started. Press 'q' to quit.")

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            landmarks = results.pose_landmarks.landmark

            # Track nose vertical movement
            nose_y = landmarks[mp_pose.PoseLandmark.NOSE].y
            if prev_nose_y is not None:
                dy = abs(nose_y - prev_nose_y)
                if dy > 0.01:
                    vertical_movement += dy * 100  # amplify sensitivity
            prev_nose_y = nose_y
            frame_count += 1

            # Get knee angle for cycling
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]
            knee_angle = get_angle(hip, knee, ankle)

            if frame_count == 30:  # Analyze every ~1 second
                label = detect_cardio(vertical_movement, knee_angle)
                vertical_movement = 0
                frame_count = 0

        # Display result
        cv2.putText(frame, f"Activity: {label}", (40, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        # Add elapsed time
        elapsed_time = int(time.time() - start_time)
        cv2.putText(frame, f"Time: {elapsed_time}s", (40, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Cardio Activity Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting...")
            break

except Exception as e:
    print(f"An error occurred: {str(e)}")

finally:
    if 'cap' in locals():
        cap.release()
    cv2.destroyAllWindows()
    print("Cleanup complete")

