import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", default="assets/care.png", required=True)
args = vars(parser.parse_args())

img_color = cv2.imread(args["path"])
img_color = cv2.resize(img_color, None, None, fx=0.75, fy=0.75)
img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

img = cv2.GaussianBlur(img, (3, 3), 0)
blurred = cv2.bilateralFilter(img, 5, sigmaColor=50, sigmaSpace=50)
edged = cv2.Canny(blurred, 100, 150, 255)

cv2.imshow("Outline of device", edged)
cv2.waitKey(0)

cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
cv2.drawContours(img_color, cnts, -1, (75, 0, 130), 4)
cv2.imshow("Target Contour", img_color)
cv2.waitKey(0)

x, y, w, h = cv2.boundingRect(cnts[0])
roi = img[y : y + h, x : x + w]
cv2.imshow("ROI", roi)

output_path = "inter/output-roi"
cv2.imwrite(output_path+".png", roi)
cv2.waitKey(0)
