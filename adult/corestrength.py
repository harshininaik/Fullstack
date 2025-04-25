import cv2
import mediapipe as mp
import numpy as np
import time

try:
    print("Initializing core strength detection...")
    
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose()

    # Start webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Could not open camera. Please check if camera is connected.")

    rep_count = 0
    position = 0  # 0 = down, 1 = up
    start_time = time.time()

    print("Setup complete. Lie down and position yourself for sit-ups.")
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

            landmarks = result.pose_landmarks.landmark
            shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]

            # Calculate distance
            distance = hip.y - shoulder.y

            # Display distance for debugging
            cv2.putText(frame, f'Position: {distance:.2f}', (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Sit-up detection logic
            if distance > 0.25:  # Down position
                status = "DOWN"
                if position == 1:
                    rep_count += 1
                    print(f"Rep completed! Count: {rep_count}")
                    position = 0
            elif distance < 0.18:  # Up position
                status = "UP"
                position = 1
            else:
                status = "HOLD"

            # Display information
            cv2.putText(frame, f'Reps: {rep_count}', (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
            cv2.putText(frame, f'Status: {status}', (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Display elapsed time
            elapsed_time = int(time.time() - start_time)
            cv2.putText(frame, f'Time: {elapsed_time}s', (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Core Strengthening - Sit-Up Counter", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            print(f"\nWorkout Summary:")
            print(f"Total Sit-ups: {rep_count}")
            print(f"Total Time: {elapsed_time} seconds")
            break

except Exception as e:
    print(f"An error occurred: {str(e)}")

finally:
    if 'cap' in locals():
        cap.release()
    cv2.destroyAllWindows()
    print("\nCleanup complete")
