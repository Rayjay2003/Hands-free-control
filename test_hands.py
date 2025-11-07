import cv2
from cvzone.HandTrackingModule import HandDetector
import time

# Initialize detector
detector = HandDetector(detectionCon=0.7, maxHands=1)

cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

prev_time = 0

print("Show your hand to the camera")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Find hands
    hands, frame = detector.findHands(frame)
    
    if hands:
        hand = hands[0]  # First hand
        lmList = hand['lmList']  # List of 21 landmarks
        fingers = detector.fingersUp(hand)  # Which fingers are up
        
        # Get index finger tip (landmark 8)
        index_tip = lmList[8]
        x, y = index_tip[0], index_tip[1]
        
        # Draw circle at fingertip
        cv2.circle(frame, (x, y), 15, (0, 255, 0), cv2.FILLED)
        
        # Show which fingers are up
        cv2.putText(frame, f'Fingers: {fingers}', (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Hand Tracking Test', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()