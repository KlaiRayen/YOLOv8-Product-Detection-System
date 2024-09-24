import cv2
import time
import numpy as np
from ultralytics import YOLO
import tkinter as tk
from tkinter import messagebox

# Load the YOLOv8 model
model = YOLO('C://Users//klair//Downloads//weights.pt')  # Adjust the path to your model

# Initialize the webcam
cap = cv2.VideoCapture(0)  # '0' for the default webcam

# Initialize detections buffer
detections_buffer = []

# Initialize product count dictionary
product_count = {}

# Start time
start_time = time.time()

# Detection active flag
detecting = False

# Function to display the report
def display_report(product_count):
    report_text = "Detection Report:\n"
    for product, count in product_count.items():
        report_text += f"{product}: {count} detections\n"
    
    messagebox.showinfo("Detection Report", report_text)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Press '1' to start detection
    if cv2.waitKey(1) & 0xFF == ord('1'):
        detecting = True
        start_time = time.time()  # Reset the timer

    if detecting:
        # Run YOLOv8 model on the frame with a confidence threshold of 0.6
        results = model.predict(frame, conf=0.6)
        
        # Render the results on the frame
        annotated_frame = results[0].plot()

        # Collect detections in the buffer
        detections_buffer.extend(results[0].boxes)

        # Check for high confidence detections
        high_confidence_detected = any(det.conf[0] > 0.9 for det in results[0].boxes)

        # Display the frame with bounding boxes
        cv2.imshow('Webcam', annotated_frame)

        # Check every 2 seconds or if high confidence is detected
        if time.time() - start_time >= 2 or high_confidence_detected:
            # Reset the timer
            start_time = time.time()
            
            # Get unique products detected
            unique_products = list({model.names[int(det.cls[0])]: det for det in detections_buffer}.keys())
            
            for product in unique_products:
                # Count the detected products
                if product in product_count:
                    product_count[product] += 1
                else:
                    product_count[product] = 1

            # Clear the detections buffer
            detections_buffer = []
            detecting = False  # Stop detecting until '1' is pressed again

    # Create a white background on the right side for the product count
    count_bg = 255 * np.ones(shape=[frame.shape[0], 300, 3], dtype=np.uint8)

    # Display the product count
    y_offset = 30
    for product, count in product_count.items():
        count_text = f"{product}: {count}"
        cv2.putText(count_bg, count_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        y_offset += 30

    # Concatenate the annotated frame with the product count background
    combined_frame = np.hstack((annotated_frame if detecting else frame, count_bg))

    # Display the combined frame
    cv2.imshow('Webcam', combined_frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # Show detection report on exit
        display_report(product_count)
        break

cap.release()
cv2.destroyAllWindows()