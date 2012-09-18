#!/usr/bin/env python2.7

import pyximport; pyximport.install()

from datetime import datetime

from chess import cython_ChESS
from corner_approximation import approximate_corners

def announce(msg):
    print str(datetime.now()) + ":", msg

# Run main function if invoked as script
if __name__ == "__main__":
    print ""
    announce("Starting...")
    cython_ChESS("3444L.ppm", "3444L.pgm")
    announce("ChESS detection done.")
    approximate_corners("3444L.pgm",
                        IMAGE_OUT="3444L.png",
                        PTS_OUT="3444L",
                        HOM_OUT="3444L")
    announce("Done.")

