import cv2
import mediapipe as mp
import time

cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

_, sample_frame = cam.read()
frame_h, frame_w, _ = sample_frame.shape

try:
    while True:
        # Capture and process frame
        _, frame = cam.read()
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = face_mesh.process(rgb_frame)
        landmark_points = output.multi_face_landmarks

        if landmark_points:
            landmarks = landmark_points[0].landmark

            # Track the iris using landmarks 474-478
            iris_x, iris_y = 0, 0
            for id, landmark in enumerate(landmarks[474:478]):
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)  # Green dot for iris tracking
                
                if id == 1:
                    iris_x, iris_y = landmark.x, landmark.y  # Save normalized iris position

            # Track upper and lower eyelid points for blink detection
            left = [landmarks[145], landmarks[159]]
            for landmark in left:
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)
                cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)  # Yellow dots for eyelid tracking

            # Blink detection (difference in y-coordinates)
            if (left[0].y - left[1].y) < 0.004:
                print("Blink detected!")  # You can trigger an action here

        # Display frame
        cv2.imshow('Iris Tracking', frame)

        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Live video stopped.")

finally:
    cam.release()
    cv2.destroyAllWindows()
