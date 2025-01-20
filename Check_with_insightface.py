import cv2
import time
import insightface
from insightface.app import FaceAnalysis

def main():
    # 1. Initialize the face analysis (detector + recognition models)
    #    'buffalo_s' is a common prepackaged model set that includes SCRFD for detection.
    app = FaceAnalysis(name='buffalo_s')
    
    # ctx_id=0 means "try GPU 0"; use ctx_id=-1 to force CPU if you have no GPU
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    # 2. Open the default camera (change the index if you have multiple cameras)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Adjust these thresholds as appropriate:
    FACE_CONFIDENCE_THRESHOLD = 0.5
    MIN_FACE_SIZE = 60  # minimum face width or height in pixels to consider valid

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam.")
            break
        
        # OPTIONAL: resize frame to speed up detection or process at a lower resolution
        frame = cv2.resize(frame, (640, 480))

        # 3. Use InsightFace to detect faces in the frame
        faces = app.get(frame)
        
        # 4. Loop through detected faces and draw bounding boxes
        for face in faces:
            # face.bbox: [x1, y1, x2, y2]
            x1, y1, x2, y2 = face.bbox.astype(int)
            conf = face.det_score  # detection confidence
            
            # Calculate face width and height
            w = x2 - x1
            h = y2 - y1
            
            # Check if face meets "valid" criteria
            #  - detection confidence above threshold
            #  - bounding box size above minimum
            if conf >= FACE_CONFIDENCE_THRESHOLD and w >= MIN_FACE_SIZE and h >= MIN_FACE_SIZE:
                color = (0, 255, 0)  # green for valid face
                label = f"Valid ({conf:.2f})"
            else:
                color = (0, 0, 255)  # red for invalid face
                label = f"Invalid ({conf:.2f})"
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Label above bounding box
            cv2.putText(
                frame, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )
        
        # 5. Show the frame
        cv2.imshow("InsightFace Real-Time Detection", frame)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
