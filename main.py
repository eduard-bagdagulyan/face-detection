import cv2
from d_class import Detector

cap = cv2.VideoCapture(0)
img2 = cv2.imread('photo.jpg')


def nothing(x):
    pass


cv2.namedWindow("Face Detection")
cv2.createTrackbar("Face Detection", "Face Detection", 0, 1, nothing)
cv2.createTrackbar("Eye Detection", "Face Detection", 0, 1, nothing)
cv2.createTrackbar("Mouth Detection", "Face Detection", 0, 1, nothing)
cv2.createTrackbar("Thresholding", "Face Detection", 0, 1, nothing)
cv2.createTrackbar("Canny Edge", "Face Detection", 0, 1, nothing)
cv2.createTrackbar("Mask", "Face Detection", 0, 1, nothing)
cv2.createTrackbar("Blending", "Face Detection", 0, 100, nothing)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    height = img.shape[0]
    width = img.shape[1]
    dim = (width, height)

    pos1 = cv2.getTrackbarPos("Face Detection", "Face Detection")
    if pos1 == 0:
        pass
    else:
        Detector.detect_face(img, gray)
    pos2 = cv2.getTrackbarPos("Eye Detection", "Face Detection")
    if pos2 == 0:
        pass
    else:
        Detector.detect_eyes(img, gray)
    pos3 = cv2.getTrackbarPos("Mouth Detection", "Face Detection")
    if pos3 == 0:
        pass
    else:
        Detector.detect_mouth(img, gray)
    pos4 = cv2.getTrackbarPos("Thresholding", "Face Detection")
    if pos4 == 0:
        pass
    else:
        img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                    cv2.THRESH_BINARY, 11, 2)
    pos5 = cv2.getTrackbarPos("Canny Edge", "Face Detection")
    if pos5 == 0:
        pass
    else:
        img = cv2.Canny(img, 100, 200)
    pos6 = cv2.getTrackbarPos("Mask", "Face Detection")
    if pos6 == 0:
        pass
    else:
        Detector.mask(img, gray)
    pos7 = cv2.getTrackbarPos("Blending", "Face Detection")
    if pos7 == 0:
        pass
    else:
        pos7 = pos7 / 100
        img2 = cv2.resize(img2, dim)
        img = cv2.addWeighted(img, 1 - pos7, img2, pos7, 0)

    cv2.imshow("Face Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
