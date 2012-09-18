
#   For description of file see CSample doc string below.

import numpy as np
from math import fabs, fsum
from sys import exit
from copy import deepcopy

class CSample(object):
    """
    A circular sample window as used in ``ChESS: Quick and Robust Detection of
    Chess-board Features'' (Bennett & Lasenby).
    """
    
    def __init__(self, radius, image,
                 center=None, position=None):
        """
        Arguments: *radius* is 3 or 5, *image* is an OpenCV Mat, optional args
        are (x, y) tuples.
        
        Note: *center* takes precedence over conflicting *position*.
        """
        self.radius = radius
        self.width = radius + 1 + radius
        self.height = self.width
        self.image = image
        
        if(center!=None):
            self.center = center
            self.position = (center[0]-self.radius, center[1]-self.radius)
        elif(position!=None):
            self.position = position
            self.center = (position[0]+self.radius, position[1]+self.radius)
        else:
            self.position = (0, 0)
            self.center = (self.radius, self.radius)
        
        if(radius==3):
            self.points = [ ( 6,  3),
                            ( 5,  1),
                            ( 3,  0),
                            ( 1,  1),
                            ( 0,  3),
                            ( 1,  5),
                            ( 3,  6),
                            ( 5,  5)  ]
        elif(radius==5):
            self.points = [ (10,  5),
                            (10,  3),
                            ( 9,  1),
                            ( 7,  0),
                            ( 5,  0),
                            ( 3,  0),
                            ( 1,  1),
                            ( 0,  3),
                            ( 0,  5),
                            ( 0,  7),
                            ( 1,  9),
                            ( 3, 10),
                            ( 5, 10),
                            ( 7, 10),
                            ( 9,  9),
                            (10,  7)  ]
        else:
            print('Error: radius value' + radius + 'unsupported')
            exit()
    
    def __copy__(self):
        return CSample(self.radius, self.image, center=self.center)
    
    def __deepcopy__(self, memo):
        return CSample(
            deepcopy(self.radius),
            deepcopy(self.image),
            center=deepcopy(self.center)
        )
    
    def __str__(self):
        return 'csample(' + self.radius + ') centered at' + self.center
    
    def at3(self, x, y):
        """
        Returns intensity of pixel in *self.image* corresponding to (*x*, *y*) in
        sample window.
        """
        absx = x + self.position[0]
        absy = y + self.position[1]
        return fsum(self.image[absy][absx])
    
    def at2(self, point):
        """
        Returns intensity of pixel in *self.image* corresponding to tuple *point*
        in sample window.
        """
        return self.at3(point[0], point[1])
    
    def I(self, n):
        """
        Returns intensity of point *self.points[n]* in sample window.
        """
        return self.at2(self.points[n])
    
    def sum_response(self):
        """
        Returns the ``sum response'' of this sample, as described in ``ChESS:
        Quick and Robust Detection of Chess-board Features'' (Bennett & Lasenby).
        """
        half = len(self.points)/2
        quarter = half/2
        
        E = 0
        for n in range(quarter):
            E += fabs(
                (self.I(n) + self.I(n + half))
                - (self.I(n + quarter) + self.I(n + quarter + half))
            )
        
        return E
    
    def diff_response(self):
        """
        Returns the ``diff response'' of this sample, as described in ``ChESS:
        Quick and Robust Detection of Chess-board Features'' (Bennett & Lasenby).
        """
        half = len(self.points)/2
        
        E = 0
        for n in range(half):
            E += fabs(self.I(n) - self.I(n + half))
        
        return E
    
    def mean_response(self):
        """
        Returns the ``mean response'' of this sample, as described in ``ChESS:
        Quick and Robust Detection of Chess-board Features'' (Bennett & Lasenby).
        """
        neighbor = self.neighbor_mean()
        hlocal = self.horizontal_local_mean()
        vlocal = self.vertical_local_mean()        
        return max(fabs(neighbor - hlocal), fabs(neighbor - vlocal))
    
    def box_local_mean(self):
        """
        Returns the mean intensity of the nine pixels in the center of this sample.
        """
        middle = (self.radius, self.radius)
        E = 0
        E += self.at2(middle + (-1, -1))
        E += self.at2(middle + ( 0, -1))
        E += self.at2(middle + ( 1, -1))
        E += self.at2(middle + (-1,  0))
        E += self.at2(middle + ( 0,  0))
        E += self.at2(middle + ( 1,  0))
        E += self.at2(middle + (-1,  1))
        E += self.at2(middle + ( 0,  1))
        E += self.at2(middle + ( 1,  1))
        return E/9
    
    def cross_local_mean(self):
        """
        Returns the mean pixel intensity of a 3x3 plus sign in the center of this
        sample.
        """
        middle = (self.radius, self.radius)
        E = 0
        E += self.at2(middle)
        E += self.at2(middle + ( 1, 0))
        E += self.at2(middle + (-1, 0))
        E += self.at2(middle + ( 0, 1))
        E += self.at2(middle + ( 0,-1))
        return E/5
    
    def horizontal_local_mean(self):
        """
        Returns the mean intensity of a horizontal sequence of three pixels in the
        center of this sample.
        """
        middle = (self.radius, self.radius)
        E = 0
        E += self.at2(middle)
        E += self.at2(middle + ( 1, 0))
        E += self.at2(middle + (-1, 0))
        return E/3
    
    def vertical_local_mean(self):
        """
        Returns the mean intensity of a vertical sequence of three pixels in the
        center of this sample.
        """
        middle = (self.radius, self.radius)
        E = 0
        E += self.at2(middle)
        E += self.at2(middle + (0,  1))
        E += self.at2(middle + (0, -1))
        return E/3
    
    def neighbor_mean(self):
        """
        Returns a lazily-computed approximation of the ``neighbor mean response''
        of this sample, as described in ``ChESS: Quick and Robust Detection of
        Chess-board Features'' (Bennett & Lasenby).
        """
        E = 0
        N = 0
        M = (self.radius**2 + (self.radius+1)**2) / 2 # M for 'max' and also 'made up'
        for x in range(self.width):
            for y in range(self.height):
                sd = fabs(x - self.radius)**2 + fabs(y - self.radius)**2
                if(sd < M):
                    E += self.at3(x, y)
                    N += 1
        return E/N
    
    def response(self):
        """
        Returns the ``response'' of this sample, as described in ``ChESS: Quick
        and Robust Detection of Chess-board Features'' (Bennett & Lasenby). For
        our images use of ``mean response'' is unnecessary and so we omit it.
        """
        return self.sum_response() - self.diff_response() #- self.mean_response()
    
