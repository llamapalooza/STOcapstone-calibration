#!/usr/bin/env python2.7

import pyximport; pyximport.install()

from datetime import datetime

from calibimage import CalibImage

TAB     = "    "

FROM    = 10
TO      = 20

IN_PREFIX   = "images/Jan25/ChESS/IMG_00"
IN_SUFFIX   = ".pgm"
OUT_PREFIX  = "images/Jan25/corners/IMG_00"
OUT_SUFFIX  = ".png"

def announce(MSG):
    print str(datetime.now()) + ": " + MSG

def process(IN_FILENAME, OUT_STEM):
    img = CalibImage(IN_FILENAME)
    img.cornerfind_from_ChESS()
    img.imwrite(OUT_STEM+OUT_SUFFIX)
    img.ptwrite(OUT_STEM)

def one():
    announce("Begin")
    process("in", "out")
    announce("Done")

def many():
    print ""
    
    for i in range(FROM, TO+1):
        for s, side in enumerate(["L", "R"]):
            IN_FILENAME = IN_PREFIX + str(i) + side + IN_SUFFIX
            OUT_STEM = OUT_PREFIX + str(i) + side
            
            announce(IN_FILENAME + " (" + str(2*(i-FROM) + s + 1) + " of " + str(2*(TO+1-FROM)) + ")")
            
            process(IN_FILENAME, OUT_STEM)
    
    announce("Done")
    
    print ""

# Run main function if invoked as script
if __name__ == '__main__':
    many()
