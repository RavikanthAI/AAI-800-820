
import cv2
import pickle
import cvzone
import numpy as np
from sklearn.metrics import precision_score, f1_score

# Load ground truth data safely
try:
    with open('groundTruth.pkl', 'rb') as f:
        groundTruth = pickle.load(f)
except FileNotFoundError:
    print("Ground truth file not found, initializing empty data.")
    groundTruth = [{'occupied': False}]

# Load the positions of parking spots
with open('carParkPos', 'rb') as f:
    posList = pickle.load(f)

cap = cv2.VideoCapture('carPark.mp4')
width, height = 107, 48

def checkParkingSpace(imgPro, groundTruth):
    spaceCounter = 0
    correct_detections = 0
    predictions = []  # Store prediction results for metric calculation

    for pos, truth in zip(posList, groundTruth):
        x, y = pos
        imgCrop = imgPro[y:y + height, x:x + width]
        count = cv2.countNonZero(imgCrop)
        prediction = count < 900
        predictions.append(prediction)
        color = (0, 255, 0) if prediction else (0, 0, 255)
        thickness = 5 if prediction else 2
        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, thickness)
        cvzone.putTextRect(img, str(count), (x, y + height - 3), scale=1.1, thickness=2, offset=1, colorR=color)
        if not prediction:  # If the spot is occupied
            spaceCounter += 1  # Increment the non-free space counter
        if prediction == truth['occupied']:
            correct_detections += 1

    accuracy = correct_detections / len(posList)
    true_labels = [gt['occupied'] for gt in groundTruth]
    f1 = f1_score(true_labels, predictions, zero_division=0)
    precision = precision_score(true_labels, predictions, zero_division=0)

    cvzone.putTextRect(img, f'Free Spaces: {69-spaceCounter} / {len(posList)}', (100, 35), scale=1.5, thickness=2, offset=10, colorR=(0, 255, 0))
    cvzone.putTextRect(img, f'Accuracy: {accuracy:.2%}', (100, 70), scale=1.5, thickness=2, offset=10, colorR=(255, 255, 0))
    cvzone.putTextRect(img, f'F1 Score: {f1:.2f}', (850, 35), scale=1.5, thickness=2, offset=10, colorR=(225, 255, 0))
    cvzone.putTextRect(img, f'Precision: {precision:.2f}', (850, 70), scale=1.5, thickness=2, offset=10, colorR=(0, 255, 0))

    return img  # Return the modified image

while True:
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    success, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)
    imgMedian = cv2.medianBlur(imgThreshold, 5)
    kernel = np.ones((3, 3), np.uint8)
    imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)

    modified_img = checkParkingSpace(imgDilate, groundTruth)  # Get the modified image with metrics
    cv2.imshow('Image', modified_img)
    if cv2.waitKey(22) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()





