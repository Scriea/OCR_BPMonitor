import cv2
import pytesseract
import argparse
import numpy
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

root = "imgs/img"
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--index", required=True, type=int)
args = vars(parser.parse_args())

def get_roi(image):
    img_color = image
    cv2.imshow("Original Image", image)
    cv2.waitKey(0)
    img_color = cv2.resize(img_color, None, None, fx=0.75, fy=0.75)
    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    blurred = cv2.bilateralFilter(img, 5, sigmaColor=50, sigmaSpace=50)
    edged = cv2.Canny(blurred, 30, 150, 255)
    cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    cv2.drawContours(img_color, cnts, -1, (75, 0, 130), 4)
    x, y, w, h = cv2.boundingRect(cnts[0])
    roi = img[y : y + h, x : x + w]
    cv2.imshow("ROI", roi)
    # output_path = "inter/output-roi"
    # cv2.imwrite(output_path+".png", roi)
    cv2.waitKey(0)

    return roi

path = root + str(args['index'])+ ".png"
image = cv2.imread(path)
image = get_roi(image)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.resize(image, None, None, fx=0.5, fy=0.5)
thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 11, 5)

cv2.imshow("A",  cv2.resize(thresh, None, None, fx=2, fy=2))
cv2.waitKey(0)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
img = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
cv2.imshow("B", cv2.resize(img, None, None, fx=2, fy=2))
cv2.waitKey(0)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 3))
img = cv2.dilate(img, kernel, iterations=1)
cv2.imshow("Dilated", cv2.resize(img, None, None, fx=2, fy=2))
cv2.waitKey(0)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1,2))
img = cv2.erode(img, kernel, iterations=1)
cv2.imshow("Eroded",  cv2.resize(img, None, None, fx=2, fy=2))
cv2.waitKey(0)

inverted = cv2.bitwise_not(img)
cv2.imshow("C", cv2.resize(inverted, None, None, fx=2, fy=2))
cv2.waitKey(0)

custom_config = r'--oem 3 --psm 6 outputbase digits'
extracted_text = pytesseract.image_to_string(cv2.resize(inverted, None, None, fx=1.25, fy=1.25), config=custom_config)

print(extracted_text)
cv2.destroyAllWindows()


























# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # Filter contours
# filtered_contours = []
# for cnt in contours:
#     perimeter = cv2.arcLength(cnt, True)
#     x, y, w, h = cv2.boundingRect(cnt)
#     aspect_ratio = float(w) / h

#     # Define your own criteria for aspect ratio, perimeter etc.
#     if perimeter > 50 and 0.2 < aspect_ratio < 1.0:
#         filtered_contours.append(cnt)

# # Draw contours for visualization
# cv2.drawContours(image, filtered_contours, -1, (0, 255, 0), 2)

# # Iterate through the filtered contours to OCR
# for cnt in filtered_contours:
#     x, y, w, h = cv2.boundingRect(cnt)
#     roi = image[y:y+h, x:x+w]

#     # OCR the individual ROI
#     text = pytesseract.image_to_string(roi, config='outputbase digits')
#     print(text)

# # print(extracted_text)
# cv2.destroyAllWindows()