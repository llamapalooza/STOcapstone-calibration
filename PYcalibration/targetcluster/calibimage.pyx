import cv2
import numpy as np
cimport numpy as np
cimport cython
DTYPE = np.int
ctypedef np.int_t DTYPE_t

from geom2d import Point, get_x, get_y
from targetcluster import kmeans_tc, identify_boards

TAB = "    "

class CalibImage(object):
    def __init__(self, fname):
        """
        todo
        """
        img = cv2.imread(fname)
        self.set_img(img)
    
    def set_img(self, img):
        """
        todo
        """
        self.img = np.asarray(img, dtype=DTYPE)
        self.rows = len(self.img)
        self.cols = len(self.img[0])
    
    @cython.boundscheck(False)
    def cornerfind_from_ChESS(self):
        """
        todo
        """
        cdef np.ndarray[DTYPE_t, ndim=3] img = self.img
        cdef int y
        cdef int x
        cdef int z
        cdef int rows = self.rows
        cdef int cols = self.cols
        
        points = []
        
        for y in range(rows):
            for x in range(cols):
                z = img[y][x][0]
                if z > 15:
                    points.append(Point(x, y, z))
        
        x_min = self.cols
        x_max = 0
        
        for point in points:
            x = get_x(point)
            if x < x_min:
                x_min = x
            elif x > x_max:
                x_max = x
        
        clusters = kmeans_tc(points, [Point(x_min, 0), Point(x_max, 0)])
        
        identify_boards(clusters)
        self.clusters = clusters
        
        self.img_out = np.zeros((self.rows, self.cols, 3))
        
        for c, cluster in enumerate(clusters):
            for p, point in enumerate(cluster.points):
                x = get_x(point)
                y = get_y(point)
                for delta_y in range(-5,6):
                    for delta_x in range(-5,6):
                        self.img_out[y+delta_y][x+delta_x][c] = 255 - 4*p
                        self.img_out[y+delta_y][x+delta_x][c-1] = 4*p
            
            x = get_x(cluster.corners[0])
            y = get_y(cluster.corners[0])
            for delta_y in range(-5,6):
                for delta_x in range(-5,6):
                    for channel in range(3):
                        self.img_out[y+delta_y][x+delta_x][channel] = 255
    
    def imwrite(self, fname):
        if self.img_out != None:
            cv2.imwrite(fname, self.img_out)
        else:
            cv2.imwrite(fname, self.img)
    
    def ptwrite(self, fstem):
        self.clusters[0].savetxt(fstem+".B")
        self.clusters[1].savetxt(fstem+".R")
    
