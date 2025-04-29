import cv2
import mediapipe as mp
import math
import time

try:
    print("Initializing resistance band exercise detection...")

    # Setup MediaPipe Pose
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def calculate_angle(a, b, c):
        """Calculate angle between three points: shoulder (a), elbow (b), wrist (c)"""
        a = [a.x, a.y]
        b = [b.x, b.y]
        c = [c.x, c.y]

        radians = math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
        angle = abs(radians * 180.0 / math.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

    # Initialize variables
    counter = 0
    stage = None
    start_time = time.time()

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Could not open camera. Please check if camera is connected.")

    print("Setup complete. Stand back from camera.")
    print("Hold resistance band and perform bicep curls.")
    print("Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Flip and process
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Get required joints for right arm
            shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
            wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

            # Calculate elbow angle
            angle = calculate_angle(shoulder, elbow, wrist)

            # Count logic for bicep curl
            if angle > 160:
                stage = "down"
            if angle < 40 and stage == "down":
                stage = "up"
                counter += 1
                print(f"Rep completed! Count: {counter}")

            # Draw angle
            cv2.putText(frame, f'Angle: {int(angle)}', (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            # Draw stage
            stage_color = (0, 255, 0) if stage == "up" else (0, 0, 255)
            cv2.putText(frame, f'Stage: {stage}', (30, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, stage_color, 2)

        # Display reps
        cv2.putText(frame, f'Reps: {counter}', (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        # Display elapsed time
        elapsed_time = int(time.time() - start_time)
        cv2.putText(frame, f'Time: {elapsed_time}s', (30, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Draw landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow("Resistance Band Curl Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nExercise Summary:")
            print(f"Total Reps: {counter}")
            print(f"Total Time: {elapsed_time} seconds")
            break

except Exception as e:
    print(f"An error occurred: {str(e)}")

finally:
    if 'cap' in locals():
        cap.release()
    cv2.destroyAllWindows()
    print("\nSession complete")
