from typing import Counter
import cv2
import os
import pytesseract
import re
# re is for cleaning the plate text

def clean_plate_text(text):
    text = text.upper()
    text = re.sub(r'[^A-Z0-9]', '', text)  # remove junk characters
    return text

# Get the path to the XML cascade file in the same folder as the script
cascade_path = os.path.join(os.path.dirname(__file__), 'indian_license_plate_cascade.xml')

# Load the cascade
plate_cascade = cv2.CascadeClassifier(cascade_path)

# Debug: check if it loaded correctly
if plate_cascade.empty():
    print("❌ Failed to load cascade")
    exit()
else:
    print("✅ Cascade loaded successfully")
# Start video capture from the default camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

from collections import Counter
plate_buffer = []

# Indian number plate regex pattern
INDIAN_PLATE_REGEX = r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$'

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect number plates
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100,30))

    # Draw rectangles around plates
    for (x, y, w, h) in plates:
        # Draw rectangle
        if w < 120 or h < 40:
            continue

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Crop number plate
        plate_roi = frame[y:y+h, x:x+w]

        # Preprocessing for OCR
        gray_plate = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
        gray_plate = cv2.resize(gray_plate, (400, 100))
        gray_plate = cv2.bilateralFilter(gray_plate, 11, 17, 17)
        _, thresh_plate = cv2.threshold(gray_plate, 150, 255, cv2.THRESH_BINARY)
        # OCR
        raw_text = pytesseract.image_to_string(
            thresh_plate,
            config='--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        )

        plate_text = clean_plate_text(raw_text)

        # Validate plate format
        if re.match(INDIAN_PLATE_REGEX, plate_text):
            plate_buffer.append(plate_text)

            if len(plate_buffer) > 15:
                plate_buffer.pop(0)

            most_common = Counter(plate_buffer).most_common(1)

            if most_common and most_common[0][1] >= 5:
                final_plate = most_common[0][0]
                print("🚗 CONFIRMED PLATE:", final_plate)

                cv2.putText(
                    frame,
                    final_plate,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )

        if re.match(INDIAN_PLATE_REGEX, plate_text):
            print("✅ VALID PLATE:", plate_text)
    

    cv2.imshow("Number Plate Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
