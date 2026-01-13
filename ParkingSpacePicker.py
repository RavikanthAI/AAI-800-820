

import cv2
import pickle

width, height = 107, 48

try:
    with open('carParkPos', 'rb') as f:
        posList = pickle.load(f)
except:
    posList = []

try:
    with open('groundTruth.pkl', 'rb') as f:
        groundTruth = pickle.load(f)
except:
    groundTruth = [{'occupied': False} for _ in posList]

def mouseClick(events, x, y, flags, params):
    if events == cv2.EVENT_LBUTTONDOWN:
        for i, pos in enumerate(posList):
            x1, y1 = pos
            if x1 < x < x1 + width and y1 < y < y1 + height:
                groundTruth[i]['occupied'] = not groundTruth[i]['occupied']  # Toggle occupancy on left-click
                break
        else:
            posList.append((x, y))
            groundTruth.append({'occupied': False})  # Append new spot as unoccupied
    elif events == cv2.EVENT_RBUTTONDOWN:
        for i, pos in enumerate(posList):
            x1, y1 = pos
            if x1 < x < x1 + width and y1 < y < y1 + height:
                posList.pop(i)
                groundTruth.pop(i)
                break  # Remove spot on right-click

    with open('carParkPos', 'wb') as f:
        pickle.dump(posList, f)
    with open('groundTruth.pkl', 'wb') as f:
        pickle.dump(groundTruth, f)

while True:
    img = cv2.imread('carParkImg.png')
    for pos, status in zip(posList, groundTruth):
        cv2.rectangle(img, pos, (pos[0]+width, pos[1]+height), (0,255,0) if status['occupied'] else (255,0,0), 2)
    cv2.imshow("Image", img)
    cv2.setMouseCallback("Image", mouseClick)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()






