import cv2
import time

print("Enter ESC to exit\nplease wait... ")

time.sleep(1.5)


bg_subtractor = cv2.createBackgroundSubtractorKNN(detectShadows=True)
erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 5))
dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 7))


cam = cv2.VideoCapture(0)


if not cam.isOpened():
   print("error opening camera")
   exit()

success, frame = cam.read()



while True:

   if not success:
      print("error in retrieving frame")
      break

   fg_mask = bg_subtractor.apply(frame)
   _, thresh = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)

   cv2.erode(thresh, erode_kernel, thresh, iterations=2)

   cv2.dilate(thresh, dilate_kernel, thresh, iterations=2)
   contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
   cv2.CHAIN_APPROX_SIMPLE)

   for c in contours:
        if cv2.contourArea(c) > 1000:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
   cv2.imshow('knn', fg_mask)
   cv2.imshow('thresh', thresh)
   cv2.imshow('background',
            bg_subtractor.getBackgroundImage())
   cv2.imshow('detection', frame)

   k = cv2.waitKey(30)
   if k == 27: # Escape
      cam.release()
      cv2.destroyAllWindows()
      break
        
   success, frame = cam.read()
