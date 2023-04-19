import cv2
from lib.rmbg import SelfiSegmentation

cam = cv2.VideoCapture(0)
cam.set(3,640)
cam.set(4,480)
segm = SelfiSegmentation()
background = cv2.imread("C:/Users/Hero/Downloads/1680413121354.jpg")
background = cv2.resize(background, (640,480))

while True :
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)
    median = segm.removeBG(frame, background, threshold=0.8)

    median = cv2.medianBlur(median,3)
    cv2.imshow("Image_processed", median)
    cv2.waitKey(1)
# cv2.destroyAllWindows()