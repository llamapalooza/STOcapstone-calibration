#!/usr/bin/env python2.7

import pyximport; pyximport.install()

from datetime import datetime

from chess import cython_ChESS
from corner_approximation import approximate_corners

FROM    = 38
TO      = 47

TOTAL       = str(2*(1+TO-FROM))

IN_PREFIX   = "images/Jan24/ChESS/34"
IN_SUFFIX   = ".pgm"
OUT_PREFIX  = "images/Jan24/corners/34"
OUT_SUFFIX  = ".png"

def ChESS(): # put in real images and get out ChESS images
    print ""
    
    for i in range(FROM, TO+1):
        for s, side in enumerate(["L", "R"]):
            IN_FILENAME = IN_PREFIX + str(i) + side + IN_SUFFIX
            OUT_FILENAME = OUT_PREFIX + str(i) + side + OUT_SUFFIX
            
            
            announce(IN_FILENAME + " (" + str(2*(i-FROM) + s + 1) + " of " + TOTAL + ")")
            
            cython_ChESS(IN_FILENAME, OUT_FILENAME)
    
    announce("Done.")
    
    print ""

def approx(): # put in ChESS images and get out corners
    print ""
    
    for i in range(FROM, TO+1):
        for s, side in enumerate(["L", "R"]):
            IN_FILENAME = IN_PREFIX + str(i) + side + IN_SUFFIX
            OUT_STEM = OUT_PREFIX + str(i) + side
            OUT_FILENAME = OUT_STEM + OUT_SUFFIX
            
            announce(IN_FILENAME + " (" + str(2*(i-FROM) + s + 1) + " of " + TOTAL + ")")
            
            approximate_corners(IN_FILENAME, IMAGE_OUT=OUT_FILENAME, PTS_OUT=OUT_STEM, HOM_OUT=OUT_STEM)
    
    announce("Done.")
    
    print ""

def announce(msg):
    print str(datetime.now()) + ":", msg

# Run main function if invoked as script
if __name__ == '__main__':
    approx()

