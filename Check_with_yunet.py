import cv2
import math

def check_face_roll(landmarks, max_tilt_deg=30.0):
    """
    A rough check for whether the face is too tilted (roll).
    We compute the line between the left eye (landmarks[0]) 
    and right eye (landmarks[1]) from YuNet and measure the angle.
    """
    left_eye = landmarks[0]
    right_eye = landmarks[1]

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]

    angle_radians = math.atan2(dy, dx)
    angle_degs = abs(math.degrees(angle_radians))

    if angle_degs > max_tilt_deg and (180 - angle_degs) > max_tilt_deg:
        return False
    return True

def main():
    # 1. Load YuNet face detector
    face_detector = cv2.FaceDetectorYN.create(
        model='./face_detection_yunet_2023mar.onnx',
        config="", 
        input_size=(320, 320),   # Inference size
        score_threshold=0.9,     # Confidence threshold
        nms_threshold=0.3,
        top_k=5000,
        backend_id=cv2.dnn.DNN_BACKEND_DEFAULT,
        target_id=cv2.dnn.DNN_TARGET_CPU
    )

    # 2. Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    # 3. Real-time loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        face_detector.setInputSize((w, h))

        # 4. Face detection
        results = face_detector.detect(frame)

        if results[1] is not None:
            faces = results[1]
        else:
            faces = []

        # 5. Go through each detected face
        for i, face in enumerate(faces):
            x, y, w_box, h_box = face[:4].astype(int)

            # Handle landmarks (excluding the confidence score)
            landmarks_data = face[4:-1]
            landmarks = landmarks_data.reshape(-1, 2)

            # Draw bounding box and landmarks
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
            for (lx, ly) in landmarks:
                cv2.circle(frame, (int(lx), int(ly)), 2, (0, 255, 0), -1)

            # Extract specific landmarks
            roll_ok = check_face_roll(landmarks, max_tilt_deg=30.0)
            print(f"Roll check: {'Valid' if roll_ok else 'Invalid'}")

        # 6. Show output
        cv2.imshow("YuNet Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
