#!/usr/bin/env python2.7

import cv2
import numpy as np
from datetime import datetime

FILES=['3447L.ppm']

#for i in range(28, 48):
#    FILES.append("34"+str(i)+"L.ppm")

if __name__ == '__main__':
    for FILE in FILES:
        print str(datetime.now()) + ": " + FILE
        for KSIZE in [3, 5, 7]:
            image = cv2.imread(FILE)
            rows = image.rows
            cols = image.cols
            src = np.asarray(image)
            
            dst = cv2.Sobel(src, 3, 1, 0, ksize=KSIZE) + cv2.Sobel(src, 3, 0, 1, ksize=KSIZE)
            dst = abs(dst)
            dst *= (255.0/dst.max())
            cv2.imwrite(FILE+".sobel"+str(KSIZE)+".png", dst)
