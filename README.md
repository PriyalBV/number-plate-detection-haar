#  Real-Time Indian Number Plate Recognition (Learning Project)

This repository represents my learning journey while building a **real-time Automatic Number Plate Recognition (ANPR)** system using **Haar Cascades, OpenCV, and Tesseract OCR**.

Instead of focusing only on results, this project documents **what I learned, why certain decisions were made, and the limitations I encountered** while working with classical computer vision techniques.


##  Project Objective

To build an end-to-end ANPR pipeline using traditional computer vision:
Webcam Feed
↓
Haar Cascade Plate Detection
↓
Region Cropping
↓
Image Preprocessing
↓
OCR (Tesseract)
↓
Regex Validation
↓
Temporal Confirmation


##  What I Learned

- How **Haar Cascade classifiers** detect structured objects like number plates using contrast-based features rather than text.

- The importance of tuning Haar parameters such as `scaleFactor`, `minNeighbors`, and `minSize` to balance detection accuracy and false positives.

- That **not every detected region is suitable for OCR**, and filtering weak detections early improves overall system performance.

- How **image preprocessing directly affects OCR accuracy**, including:
  - Grayscale conversion
  - Resizing for character clarity
  - Noise reduction using bilateral filtering
  - Thresholding to improve text contrast

- That **OCR performance depends more on input quality than the OCR engine itself**.

- How to configure **PyTesseract** for license plate recognition using:
  - Appropriate page segmentation mode (`psm 8`)
  - Character whitelisting to reduce misclassification

- The necessity of **post-processing OCR output**, using regex-based cleaning to remove noise and invalid characters.

- How applying **domain-specific validation rules** (Indian number plate format) significantly reduces false positives.

- Why **single-frame OCR is unreliable** in real-time systems and how temporal consistency improves accuracy.

- How implementing a **frame buffer with frequency-based confirmation** increases confidence in detected plates.

- The importance of thinking in terms of an **end-to-end computer vision pipeline** rather than isolated steps.

- The practical limitations of classical computer vision approaches under:
  - Poor lighting
  - Motion blur
  - Non-frontal or angled plates

- Why modern ANPR systems prefer **deep learning–based detection and OCR**, after experiencing the constraints of traditional methods firsthand.



