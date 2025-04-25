import cv2
import mediapipe as mp
import math
import time

try:
    print("Initializing stretching detection...")
    
    # Initialize MediaPipe
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Could not open camera. Please check if camera is connected.")

    stretch_count = 0
    stretched = False
    start_time = time.time()

    def get_distance(p1, p2):
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    print("Setup complete. Stand back from camera.")
    print("Touch your toes to count stretches.")
    print("Press ESC to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Process frame
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb_frame)

        if result.pose_landmarks:
            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = result.pose_landmarks.landmark

            # Measure distance between wrist and ankle
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
            distance = get_distance(left_wrist, left_ankle)

            # Detect stretch
            if distance < 0.15 and not stretched:
                stretch_count += 1
                stretched = True
                print(f"Stretch detected! Count: {stretch_count}")
            elif distance > 0.25:
                stretched = False

            # Display information
            cv2.putText(frame, f"Stretches: {stretch_count}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 3)
            
            status = "TOUCHING TOES" if stretched else "STANDING"
            cv2.putText(frame, f"Status: {status}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display elapsed time
            elapsed_time = int(time.time() - start_time)
            cv2.putText(frame, f"Time: {elapsed_time}s", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Stretching Detector", frame)
        if cv2.waitKey(10) & 0xFF == 27:  # ESC to quit
            print("\nWorkout Summary:")
            print(f"Total Stretches: {stretch_count}")
            print(f"Total Time: {elapsed_time} seconds")
            break

except Exception as e:
    print(f"An error occurred: {str(e)}")

finally:
    if 'cap' in locals():
        cap.release()
    cv2.destroyAllWindows()
    print("\nCleanup complete")
