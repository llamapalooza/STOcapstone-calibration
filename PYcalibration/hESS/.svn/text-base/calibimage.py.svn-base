import cv2
import numpy as np

from geom2d import Point, get_x, get_y # rm get_x,y later...
from bennettlasenby import ChESS
from targetcluster import kmeans_tc, identify_boards

class CalibImage(object):
    def __init__(self, fname):
        """
        todo
        """
        img = cv2.imread(fname)
        self.set_img(img)
    
    def set_img(self, img):
        self.img = np.asarray(img)
        self.rows = len(self.img)
        self.cols = len(self.img[0])
    
    def cornerfind_from_image(self):
        """
        todo
        """
        self.set_img(ChESS(self.img))
        self.calibrate_from_ChESS()
    
    def cornerfind_from_ChESS(self):
        """
        todo
        """
        points = [ Point(x, y, sum(self.img[y][x]))
                   for y in range(self.rows)
                   for x in range(self.cols) ]
        
        clusters = kmeans_tc(points, [Point(0, 0), Point(self.cols - 1, 0)])
        
        identify_boards(clusters)
        
        for cluster in clusters:
            print str(cluster.size())
            for corner in cluster.corners:
                print "\t" + str(get_x(corner)) + "," + str(get_y(corner))
    
    def imwrite(self, fname):
        cv2.imwrite(fname, self.img)
