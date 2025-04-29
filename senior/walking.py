import cv2
import mediapipe as mp
import time

try:
    print("Initializing walking detection...")
    
    # Initialize MediaPipe Pose model
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Could not open camera. Please check if camera is connected.")

    # Variables to track
    prev_time = 0
    walking = False
    step_count = 0
    start_time = time.time()

    print("Setup complete. Stand back from camera.")
    print("Walk naturally in view of the camera.")
    print("Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Get ankle landmarks
            left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
            right_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]

            # Detect walking
            ankle_diff = abs(left_ankle.y - right_ankle.y)
            if ankle_diff > 0.05:
                if not walking:  # Started walking
                    step_count += 1
                walking = True
            else:
                walking = False

            # Display information
            status = "Walking" if walking else "Standing"
            color = (0, 255, 0) if walking else (0, 0, 255)
            
            cv2.putText(frame, f"Status: {status}", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, f"Steps: {step_count}", (50, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)
            
            # Display elapsed time
            elapsed_time = int(time.time() - start_time)
            cv2.putText(frame, f"Time: {elapsed_time}s", (50, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Walking Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nExercise Summary:")
            print(f"Total Steps: {step_count}")
            print(f"Total Time: {elapsed_time} seconds")
            break

except Exception as e:
    print(f"An error occurred: {str(e)}")

finally:
    if 'cap' in locals():
        cap.release()
    cv2.destroyAllWindows()
    print("\nCleanup complete")
