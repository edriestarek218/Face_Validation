import cv2
import math
import insightface
from insightface.app import FaceAnalysis
import numpy as np

# Check the tilt or rotation using keypoints in a very rough manner.
def check_face_pose(face, max_tilt_deg=30.0):
    """
    Returns True if face is within 'max_tilt_deg' of horizontal in yaw/roll,
    otherwise returns False.
    """

    # face.kps is typically [ [x_left_eye, y_left_eye],
    #                        [x_right_eye, y_right_eye],
    #                        [x_nose, y_nose],
    #                        [x_mouth_left, y_mouth_left],
    #                        [x_mouth_right, y_mouth_right] ]
    kps = face.kps

    # For a quick 'roll' estimate, measure the line between the two eyes.
    left_eye = kps[0]
    right_eye = kps[1]

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]

    # Estimate roll in degrees by slope of eye line
    roll_radians = math.atan2(dy, dx)
    roll_deg = abs(math.degrees(roll_radians))

    # If roll angle is near 0 or near 180, face is "upright". If near 90, face is sideways.
    # We'll do a simple check if roll > 30 deg => possibly invalid for front-facing usage.
    if roll_deg > max_tilt_deg and (180 - roll_deg) > max_tilt_deg:
        return False  # face is too tilted

    # You could do similar logic for 'pitch' or 'yaw' if you have 3D pose or do advanced checks.
    # For brevity, we won't do it all here. The roll check alone can filter out heavily rotated faces.

    return True  # Within acceptable tilt range

def main():
    # Initialize
    app = FaceAnalysis(name='buffalo_s')  
    app.prepare(ctx_id=0, det_size=(640, 640))

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error opening webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = app.get(frame)
        
        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            conf = face.det_score

            # Basic confidence check
            if conf < 0.60:
                color = (0, 0, 255)
                label = f"LowConf ({conf:.2f})"
            else:
                # Check pose
                if check_face_pose(face, max_tilt_deg=30):
                    color = (0, 255, 0)
                    label = f"Valid ({conf:.2f})"
                else:
                    color = (0, 0, 255)
                    label = f"Tilted ({conf:.2f})"

            # Draw
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # OPTIONAL: Draw the 5 keypoints to visualize
            for kp in face.kps:
                cv2.circle(frame, (int(kp[0]), int(kp[1])), 3, color, -1)
        
        cv2.imshow("Pose Validation", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
