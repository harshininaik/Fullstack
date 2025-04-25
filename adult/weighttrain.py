import cv2
import mediapipe as mp
import numpy as np
import time

try:
    print("Initializing weight training detection...")
    
    # MediaPipe initialization
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # Angle calculation function
    def calculate_angle(a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = abs(radians*180.0/np.pi)
        return angle if angle <= 180 else 360 - angle

    # Video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Could not open camera. Please check if camera is connected.")

    rep_count = 0
    direction = 0  # 0: down, 1: up
    start_time = time.time()

    print("Setup complete. Stand back from camera and show your right arm.")
    print("Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(img_rgb)

        if result.pose_landmarks:
            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Get landmarks for right arm
            landmarks = result.pose_landmarks.landmark
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
            elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
            wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]

            angle = calculate_angle(shoulder, elbow, wrist)

            # Display angle
            cv2.putText(frame, f'Angle: {int(angle)}', (30, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Count logic
            if angle > 160:
                if direction == 1:
                    rep_count += 1
                    print(f"Rep completed! Count: {rep_count}")
                    direction = 0
            if angle < 50:
                direction = 1

            # Display info
            cv2.putText(frame, f'Reps: {rep_count}', (30, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
            
            # Display exercise status
            status = "UP" if direction == 1 else "DOWN"
            cv2.putText(frame, f'Status: {status}', (30, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Display time
            elapsed_time = int(time.time() - start_time)
            cv2.putText(frame, f'Time: {elapsed_time}s', (30, 160),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Weight Training Counter (Right Arm)', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            print(f"\nWorkout Summary:")
            print(f"Total Reps: {rep_count}")
            print(f"Total Time: {elapsed_time} seconds")
            break

except Exception as e:
    print(f"An error occurred: {str(e)}")

finally:
    if 'cap' in locals():
        cap.release()
    cv2.destroyAllWindows()
    print("\nCleanup complete")
