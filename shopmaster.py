from ultralytics import YOLO
import cv2

# Load the YOLOv8 model
model = YOLO('C://Users//klair//Downloads//weights.pt')  # Adjust the path to your model

# Initialize the webcam
cap = cv2.VideoCapture(0)  # '0' for the default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run YOLOv8 model on the frame with a confidence threshold of 0.6
    results = model.predict(frame, conf=0.6)
    
    # Render the results on the frame
    annotated_frame = results[0].plot()

    # Display the frame with bounding boxes
    cv2.imshow('Webcam', annotated_frame)
    
    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
