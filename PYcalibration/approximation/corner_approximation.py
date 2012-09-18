import cv2
import numpy as np

import cluster as cl
import homography as hm
from coord import homogenous, ctoh, htoc

def approximate_corners(IMAGE_IN, IMAGE_OUT=None, PTS_OUT=None, HOM_OUT=None):
    """
    Approximates the locations of corners on each of two calibration boards in ChESS
    response image with filename *IMAGE_IN*. If requested, an image is written to
    *IMAGE_OUT* for visual inspection, a matrix of (x, y) image coordinates in the
    shape of each board is np.save()'d to *PTS_OUT*0 and *PTS_OUT*1, respectively,
    and a homography from [x, y, 1]^T (with x in range(width) and y in range(height))
    to each board's image coordinates is np.savetxt()'d to *HOM_OUT*[0,1].npy, again
    respectively.
    """
    # read in image and disregard pixels of very slight intensity
    image_in = np.asarray(cv2.imread(IMAGE_IN))
    nrows = len(image_in)
    ncols = len(image_in[0])
    image_out = np.zeros((nrows, ncols, 3), dtype=np.int)
    
    points = [ [x, y, image_in[y][x][0]]
               for y in range(nrows)
               for x in range(ncols)
               if image_in[y][x][0] > 15 ]
    
    # 2-means cluster to find the boards
    x_min = min([ point[0] for point in points ])
    x_max = max([ point[0] for point in points ])

    boards = cl.kmeans(points, [[x_min, 0], [x_max, 0]], 5)
    
    # process each board separately
    for b, board in enumerate(boards):
        # do some filtering to (hopefully!) get one point per corner
        board = _reduce(board)
        
        # guess homography and point order
        if HOM_OUT != None:
            THIS_HOM_OUT = HOM_OUT + str(b)
        else:
            THIS_HOM_OUT = None
        board = _reshape(board, THIS_HOM_OUT)
        
        # output
        if PTS_OUT != None:
            np.save(PTS_OUT + str(b), board)
        if IMAGE_OUT != None:
            _imwrite(image_out, board, b)
    
    # output
    if IMAGE_OUT != None:
        cv2.imwrite(IMAGE_OUT, image_out)

def _reduce(points3):
    """
    If *points3* is a set of points representing the ChESS response of an image
    segment containing a single calibration board, returns an undordered set of
    points which contains exactly one point for each corner on the board.
    """
    reduced = cl.high_pass_filter(points3, threshold=0.2)
    reduced = cl.local_maxima(reduced, radius=12)
    reduced = cl.geographic_inliers(reduced, factor=2)
    reduced = cl.high_pass_filter(reduced, threshold=0.4)
    return reduced

def _reshape(points2, HOM_OUT=None):
    """
    Returns a matrix containing two-dimensional representations of corner points in
    *points2*, in a shape which hopefully corresponds to that of the calibration
    board on which the corners are located. The associated homography is constrained
    to map the corners of a pre-image board to the observed corners, and the
    board dimensions must be co-divisors of len(*points2*). If requested,
    np.savetxt()'s this homography to *HOM_OUT*.
    """
    corners = cl.corners(points2)
    npts = len(points2)
    
    min_error = None
    H = None
    height = None
    width = None
    ordered = None
    for (rows, cols) in [ (div, npts/div)
                          for div in range(2, npts/2 + 1)
                          if npts % div == 0 ]:
        h = hm.homography(_box(rows, cols), [ ctoh(p) for p in corners ])
        ordering, error = hm.best_into_mapping(_board(rows, cols),
                                               h,
                                               [ ctoh(p) for p in points2 ])
        
        if min_error == None or error < min_error:
            min_error = error
            H = h
            height = rows
            width = cols
            ordered = ordering
    
    print "    sum of squared error:", min_error
    
    if HOM_OUT != None:
        np.savetxt(HOM_OUT, H)
    # return the homography and a matrix of the points in the shape of the board
    return np.reshape(np.asarray([ htoc(p)
                                   for p in ordered ]),
                      (height, width, -1))

def _imwrite(image, board, channel):
    """
    Adds a visual representation of calibration board *board* to image *image* with
    respect to channel *channel* in a completely arbitrary fashion. The first corner
    appears in white, and following corners progress from color *channel* to color
    *channel*-1.
    """
    def _plot(p, c, z):
        # Plot a blob that is more readily visible than a single pixel.
        x = p[0]
        y = p[1]
        for delta_y in range(-5,6):
            for delta_x in range(-5,6):
                image[y+delta_y][x+delta_x][c] = z
    
    # color-code corners
    for nrow, row in enumerate(board):
        for ncol, point in enumerate(row):
            _plot(point, channel, max(0, 255 - 25*nrow - 8*ncol))
            _plot(point, channel - 1, min(255, 25*nrow + 8*ncol))
    
    # first corner in white
    for channel in range(3):
        _plot(board[0][0], channel, 255)

def _box(h, w):
    """
    A sequence of points representing the four corners of a board of height *h* and
    width *w* positioned at (0, 0) and in the first quadrant.
    """
    return [ homogenous(  0,   0),
             homogenous(w-1,   0),
             homogenous(w-1, h-1),
             homogenous(  0, h-1) ]

def _board(h, w):
    """
    A sequence of points representing all corners (by row) on a board of height *h*
    and width *w* positioned at (0, 0) and in the first quadrant.
    """
    return [ homogenous(x, y) for y in range(h) for x in range(w) ]

