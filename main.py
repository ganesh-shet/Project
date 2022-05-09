import time
import cv2
from matplotlib import pyplot as plt
import imutils
import firebase_admin
from firebase_admin import credentials,firestore
from datetime import datetime

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)

db=firestore.client()
filled = db.collection('Filled')
unfilled = db.collection('Unfilled')

# read image and take first channel only
cover_3_channel = cv2.VideoCapture(0)
#cover_gray = cv2.split(cover_3_channel)[0]
def make_720p():
    cover_3_channel.set(3, 640)
    cover_3_channel.set(4, 480)
make_720p()
while(True):
    ret, img= cover_3_channel.read()
    cv2.imshow("Original Image", img)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
cover_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", cover_gray)

# blur image
cover_gray = cv2.GaussianBlur(cover_gray, (7, 7), 0)
cv2.imshow("Gray Smoothed 7 x 7", cover_gray)
cv2.waitKey(0)

# draw histogram
plt.hist(cover_gray.ravel(), 256, [0, 256]);
plt.show()

# manual threshold
(T, cover_threshold) = cv2.threshold(cover_gray, 80, 255, cv2.THRESH_BINARY_INV)
cv2.imshow(" Gray Threshold 27.5", cover_threshold)
cv2.waitKey(0)

# apply opening operation
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
cover_open = cv2.morphologyEx(cover_threshold, cv2.MORPH_OPEN, kernel)
cv2.imshow(" 5 x 5", cover_open)
cv2.waitKey(0)

# find all contours
contours = cv2.findContours(cover_open.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
cover_clone = img.copy()
cv2.drawContours(cover_clone, contours, -1, (255, 0, 0), 2)
cv2.imshow("All Contours", cover_clone)
cv2.waitKey(0)

# sort contours by area
areas = [cv2.contourArea(contour) for contour in contours]
(contours, areas) = zip(*sorted(zip(contours, areas), key=lambda a: a[1]))
# print contour with largest area
cover_clone = img.copy()
cv2.drawContours(cover_clone, [contours[-1]], -1, (255, 0, 0), 2)
cv2.imshow("Largest contour", cover_clone)
cv2.waitKey(0)

# draw bounding box, calculate aspect and display decision
cover_clone = img.copy()
(x, y, w, h) = cv2.boundingRect(contours[-1])
aspectRatio = w / float(h)
print(aspectRatio)

dt1 = time.localtime()
dt = time.asctime(dt1)

f_count = 0
u_count = 0
if aspectRatio < 1:
    cv2.rectangle(cover_clone, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(cover_clone, "FILLED", (x + 10, y + 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
    status = "Filled"
    filled.add({'Status': status, 'TimeStamp': dt})

else:
    cv2.rectangle(cover_clone, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(cover_clone, "Error", (x + 10, y + 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
    status = "UnFilled"
# dt = datetime.now()
    unfilled.add({'Status': status, 'TimeStamp': dt})
    unfilled.add({'Status': status, 'TimeStamp': dt})
cv2.imshow("FINAL OUTPUT", cover_clone)
cv2.waitKey(0)


