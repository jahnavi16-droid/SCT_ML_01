import cv2
import numpy as np
import math

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 640)
cap.set(4, 480)

bg = None
bg_captured = False

print("Press 'b' to capture background")
print("Then put your hand inside box")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    x1, y1, x2, y2 = 200, 100, 450, 350
    roi = frame[y1:y2, x1:x2]

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    if bg_captured:
        diff = cv2.absdiff(bg, cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        thresh = cv2.GaussianBlur(thresh, (5,5), 0)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            contour = max(contours, key=cv2.contourArea)

            if cv2.contourArea(contour) > 2000:

                hull = cv2.convexHull(contour)
                hull_defects = cv2.convexHull(contour, returnPoints=False)

                finger_count = 0

                if hull_defects is not None:
                    defects = cv2.convexityDefects(contour, hull_defects)

                    if defects is not None:
                        for i in range(defects.shape[0]):
                            s,e,f,d = defects[i,0]

                            start = tuple(contour[s][0])
                            end = tuple(contour[e][0])
                            far = tuple(contour[f][0])

                            a = math.dist(start,end)
                            b = math.dist(start,far)
                            c = math.dist(end,far)

                            if b*c != 0:
                                angle = math.degrees(
                                    math.acos((b*b + c*c - a*a)/(2*b*c))
                                )

                                if angle <= 90:
                                    finger_count += 1

                finger_count += 1

                cv2.putText(frame, f"Fingers: {finger_count}",
                            (50,50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,(255,0,0),2)

        cv2.imshow("Threshold", thresh)

    cv2.putText(frame, "Press 'b' to set background",
                (50,450),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,(0,255,255),2)

    cv2.imshow("Finger Counter", frame)

    key = cv2.waitKey(1)

    if key == ord('b'):
        bg = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        bg_captured = True
        print("Background Captured")

    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()