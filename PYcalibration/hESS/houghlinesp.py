#!/usr/bin/env python2.7

import cv2
import math
import numpy as np
from datetime import datetime

FILES=['3447L.ppm.sobel7.png']

#for i in range(28, 48):
#    FILES.append("34"+str(i)+"L.ppm")

if __name__ == '__main__':
    for FILE in FILES:
        print str(datetime.now()) + ": " + FILE
        img = cv2.imread(FILE)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#        edges = cv2.Canny(gray, 80, 120)
#        lines = cv2.HoughLinesP(edges, 100, math.pi/180, 50, minLineLength=500);
        lines = cv2.HoughLinesP(gray, 100, math.pi/180, 50, minLineLength=500);
        for line in lines[0]:
            pt1 = (line[0],line[1])
            pt2 = (line[2],line[3])
            cv2.line(img, pt1, pt2, (0,0,255), 3)
        cv2.imwrite(FILE+".lines.png", img)
