import cv2
import mediapipe as mp
import time

try:
    print("Initializing stretch detection system...")
    
    # Setup MediaPipe
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Could not open camera. Please check if camera is connected.")

    # Initialize variables
    stretch_count = 0
    start_time = time.time()
    last_stretch = ""

    print("Setup complete. Stand back from camera.")
    print("Perform stretches slowly and carefully.")
    print("Press 'q' to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame")
            break

        # Flip and convert
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Pose detection
        results = pose.process(rgb)
        stretch_detected = ""

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Get Y coordinates
            left_hand_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y
            right_hand_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y
            head_y = landmarks[mp_pose.PoseLandmark.NOSE].y
            hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP].y

            # Detect stretches
            if left_hand_y < head_y and right_hand_y < head_y:
                stretch_detected = "Overhead Arm Stretch"
            elif left_hand_y > hip_y and right_hand_y > hip_y:
                stretch_detected = "Toe Touch Stretch"

            # Count unique stretches
            if stretch_detected and stretch_detected != last_stretch:
                stretch_count += 1
                print(f"Stretch detected: {stretch_detected}")
                last_stretch = stretch_detected

            # Draw landmarks
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display information
        if stretch_detected:
            cv2.putText(frame, f"{stretch_detected}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No Stretch Detected", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display stretch count and time
        elapsed_time = int(time.time() - start_time)
        cv2.putText(frame, f"Stretches: {stretch_count}", (50, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)
        cv2.putText(frame, f"Time: {elapsed_time}s", (50, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Senior Stretch Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nExercise Summary:")
            print(f"Total Stretches Performed: {stretch_count}")
            print(f"Total Time: {elapsed_time} seconds")
            break

except Exception as e:
    print(f"An error occurred: {str(e)}")

finally:
    if 'cap' in locals():
        cap.release()
    cv2.destroyAllWindows()
    print("\nSession complete")
