#!/usr/bin/env python2.7

import csample
from datetime import datetime
import cv2.cv as cv
import cv2
from copy import deepcopy
import numpy as np

PREFIX="atp"

def main():
    for IMAGENUM in [1]:
        print str(datetime.now()) + ": " + PREFIX + str(IMAGENUM)
        r = 5
        img1 = cv.LoadImageM(PREFIX+str(IMAGENUM)+".png")
        rows = img1.rows
        cols = img1.cols
        img1arr = np.asarray(img1)
        img2arr = np.empty((rows-2*r, cols-2*r))
        img2arr.fill(240)

        for ypos in range(r, rows-r):
            for xpos in range(r, cols-r):
                spl = csample.CSample(r, img1arr, center=(xpos, ypos))
                img2arr[ypos-r][xpos-r] = spl.response()
        
        img2arr *= (255.0/img2arr.max())
        
        cv2.imwrite(PREFIX+str(IMAGENUM)+".png.5px.vanilla.pgm", img2arr)

# Run main function if invoked as script
if __name__ == '__main__':
    main()
