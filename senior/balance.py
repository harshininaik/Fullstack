import cv2
import mediapipe as mp
import time
import numpy as np

try:
    print("Initializing balance training system...")

    # Initialize MediaPipe pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Could not open camera. Please check if camera is connected.")

    # Variables for tracking
    balance_start_time = None
    balancing = False
    best_time = 0
    total_balance_time = 0
    balance_attempts = 0
    start_time = time.time()

    print("Setup complete. Stand back from camera.")
    print("Lift one leg to start balance detection.")
    print("Press 'q' to quit.")

    while cap.isOpened():
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
            left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
            right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
            left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
            right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]

            # Detect leg lift
            left_leg_up = left_ankle.y < right_knee.y - 0.1
            right_leg_up = right_ankle.y < left_knee.y - 0.1

            # Track balance state
            if left_leg_up or right_leg_up:
                if not balancing:
                    balance_start_time = time.time()
                    balancing = True
                    balance_attempts += 1
            else:
                if balancing:
                    duration = time.time() - balance_start_time
                    total_balance_time += duration
                    best_time = max(best_time, duration)
                balancing = False
                balance_start_time = None

        # Display information
        if balancing and balance_start_time:
            elapsed = round(time.time() - balance_start_time, 1)
            cv2.putText(frame, f"Balancing: {elapsed}s", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(frame, "Keep it steady!", (50, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Not Balancing", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.putText(frame, "Lift one leg", (50, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display stats
        cv2.putText(frame, f"Best Time: {round(best_time, 1)}s", (50, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
        cv2.putText(frame, f"Attempts: {balance_attempts}", (50, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)

        # Display session time
        session_time = int(time.time() - start_time)
        cv2.putText(frame, f"Session: {session_time}s", (50, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Balance Training Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nBalance Training Summary:")
            print(f"Total Attempts: {balance_attempts}")
            print(f"Best Balance Time: {round(best_time, 1)} seconds")
            print(f"Average Balance Time: {round(total_balance_time/max(1, balance_attempts), 1)} seconds")
            print(f"Total Session Time: {session_time} seconds")
            break

except Exception as e:
    print(f"An error occurred: {str(e)}")

finally:
    if 'cap' in locals():
        cap.release()
    cv2.destroyAllWindows()
    print("\nSession complete")
