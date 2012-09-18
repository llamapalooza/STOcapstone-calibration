#!/usr/bin/env python2.7

import cv2
import math
import numpy as np
from copy import deepcopy
from datetime import datetime
from sys import exit

def main():
    FILES=[]
    for i in range(28, 48):
        FILES.append("34"+str(i)+"L.ppm.no_MR.pgm")
    
    for FILE in FILES:
        print str(datetime.now()) + ": " + FILE
        
        k = kimage(FILE)
        k.kmeans()
        k.imwrite()

def sqdist(x1, y1, x2, y2):
    """
    Returns the squared distance between points (*x1*,*y1*) and (*x2*,*y2*).
    """
    dx = x2 - x1
    dy = y2 - y1
    return dx*dx + dy*dy

class kimage(object):
    """
    Performs kmeans clustering on an image, weighting samples (pixels) by
    intensity.
    """
    def __init__(self, fname, epsilon=None, maxiter=None):
        NOISE_THRESHOLD = 0.2 # discard samples (pixels) of <20% max intensity
        DEFAULT_EPSILON = 25 # stopping condition, center movement in pixels
        DEFAULT_MAXITER = 100 # stopping condition, maximum number of iterations
        # seeding of cluster centers depends upon image size: see use below
        
        self.infname = fname
        
        img = cv2.imread(self.infname)
        imgarr = np.asarray(img) # img and imgarr deleted after __init__
        
        self.rows = len(imgarr)
        self.cols = len(imgarr[0])
        
        if epsilon != None:
            self.epsilon = epsilon
        else:
            self.epsilon = DEFAULT_EPSILON
        
        if maxiter != None:
            self.maxiter = maxiter
        else:
            self.maxiter = DEFAULT_MAXITER
        
        # fake prevcenters give the illusion of not being finished clustering yet:
        self.prevcenters = [[self.epsilon+1, self.epsilon], [0, 0]]
        
        # initial clusters will be left and right sides of image:
        self.centers = [[0, 0], [self.cols - 1, 0]]
        
        self.clusters = self.empty_clusters()
        
        zmax = 0
        for y in range(self.rows):
            for x in range(self.cols):
                z = sum(imgarr[y][x])
                if z > zmax:
                    zmax = z
        
        for y in range(self.rows):
            for x in range(self.cols):
                z = sum(imgarr[y][x])
                if z > (zmax*NOISE_THRESHOLD):
                    self.clusters[self.label(x,y)].append([x, y, z])
    
    def label(self):
        """
        Assigns a point to the nearest cluster center. Returns the index of this
        center in *self.centers* (corresponds to cluster index in *self.clusters*).
        """
        minsqd = self.rows**2 + self.cols**2 # equivalent to infinity in this case
        label = None
        for c in range(len(self.centers)):
            sqd = sqdist(x, y, self.centers[c][0], self.centers[c][1])
            if sqd < minsqd:
                minsqd = sqd
                label = c
        if label == None:
            print "Error: unable to assign cluster to point" + str(x) + "," + str(y)
            exit()
        return label
    
    def empty_clusters(self):
        """
        Returns a list of empty clusters, one for center in *self.centers*.
        """
        ret = []
        for i in range(len(self.centers)):
            ret.append([])
        return ret
    
    def not_finished(self):
        """
        Returns True if a center has moved at least *self.epsilon* pixels since the
        last iteration of clustering.
        """
        for i in range(len(self.centers)):
            sqd = sqdist(self.centers[i][0], self.centers[i][1],
                         self.prevcenters[i][0], self.prevcenters[i][1])
            if sqd > self.epsilon**2:
                return True
        return False
    
    def cavg(self, c):
        """
        Returns weighted average of pixel coordinates in cluster with index *c*;
        weight for each pixel is its intensity.
        """
        N = 0
        Ex = 0
        Ey = 0
        for point in self.clusters[c]:
            x = point[0]
            y = point[1]
            z = point[2]
            N += z
            Ex += z*x
            Ey += z*y
        return Ex/N, Ey/N
    
    def new_centers(self):
        """
        Relocates cluster centers according to current clustering of samples.
        """
        self.prevcenters = deepcopy(self.centers)
        for c in range(len(self.clusters)):
            x, y = self.cavg(c)
            self.centers[c][0] = x
            self.centers[c][1] = y
        print "\tnew centers are:"
        for center in self.centers:
            print "\t\t(" + str(center[0]) + ", " + str(center[1]) + ")"
    
    def cluster(self):
        """
        Reclusters samples according to current locations of cluster centers.
        """
        new_clusters = self.empty_clusters()
        for cluster in self.clusters:
            for point in cluster:
                x = point[0]
                y = point[1]
                z = point[2]
                new_clusters[self.label(x,y)].append([x, y, z])
        self.clusters = new_clusters
    
    def kmeans(self):
        """
        Perform kmeans clustering until in one iteration no cluster center moves
        more than *self.epsilon* pixels.
        """
        i = self.maxiter
        self.init_clusters()
        while self.not_finished() and (i>0):
            self.cluster()
            self.new_centers()
            i -= 1
        if i == 0:
            print "\tmax iterations!"
    
    def outfname(self):
        """
        Returns the name to be used for any output image files.
        """
        return str(self.infname) + str(len(self.centers)) + "-means.png"
    
    def imwrite(self):
        """
        Writes current clustering to a file to be displayed as an image.
        """
        NUM_CHANNELS = 3 # cluster i printed on channel i modulo NUM_CHANNELS
        DISPLAY_CENTERS = True # each center a white pixel in output image
        
        outimage = np.zeros((self.rows, self.cols, NUM_CHANNELS))
        
        for c in range(len(self.clusters)):
            for point in self.clusters[c]:
                x = point[0]
                y = point[1]
                z = point[2]
                outimage[y][x][c % NUM_CHANNELS] = z
        
        if DISPLAY_CENTERS:
            for center in self.centers:
                x = center[0]
                y = center[1]
                outimage[y][x][0] = 255
                outimage[y][x][1] = 255
                outimage[y][x][2] = 255
        
        cv2.imwrite(self.outfname(), outimage)


if __name__ == '__main__':
    main()


