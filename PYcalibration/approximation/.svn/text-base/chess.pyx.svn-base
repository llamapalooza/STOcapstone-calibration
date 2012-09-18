
import cv2
import numpy as np
cimport numpy as np
cimport cython
DTYPE = np.int
ctypedef np.int_t DTYPE_t

@cython.boundscheck(False)
def cython_ChESS(infname, outfname, normalize=True):
    """
    A regrettable unrolled loop implementation of ChESS feature detector as described
    in ``ChESS: Quick and Robust Detection of Chess-board Features'' (Bennett &
    Lasenby), but with the exception that this implementation disregards the so-called
    ``mean response.'' If requested, the ChESS response image is normalized.
    """
    cdef int r = 5
    cdef np.ndarray[DTYPE_t, ndim=3] img = np.asarray(cv2.imread(infname), dtype=DTYPE)
    cdef int rows = len(img)
    cdef int cols = len(img[0])
    cdef np.ndarray[DTYPE_t, ndim=2] img_in = np.empty((rows, cols), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] img_out = np.zeros((rows, cols), dtype=DTYPE)
    cdef int x, y
    cdef double NORMALIZE_TO = 250.0
    cdef:
        int value, max_value
        int I_0
        int I_1
        int I_2
        int I_3
        int I_4
        int I_5
        int I_6
        int I_7
        int I_8
        int I_9
        int I_10
        int I_11
        int I_12
        int I_13
        int I_14
        int I_15
        int S_0
        int S_1
        int S_2
        int S_3
        int D_0
        int D_1
        int D_2
        int D_3
        int D_4
        int D_5
        int D_6
        int D_7
    
    for y in range(rows):
        for x in range(cols):
            img_in[y,x] = img[y,x,0] + img[y,x,1] + img[y,x,2]
    
    max_value = 0
    for y in range(r, rows-r):
        for x in range(r, cols-r):
            value = 0
            I_0  = img_in[y+5,x  ]
            I_1  = img_in[y+5,x-2]
            I_2  = img_in[y+4,x-4]
            I_3  = img_in[y+2,x-5]
            I_4  = img_in[y  ,x-5]
            I_5  = img_in[y-2,x-5]
            I_6  = img_in[y-4,x-4]
            I_7  = img_in[y-5,x-2]
            I_8  = img_in[y-5,x  ]
            I_9  = img_in[y-5,x+2]
            I_10 = img_in[y-4,x+4]
            I_11 = img_in[y-2,x+5]
            I_12 = img_in[y  ,x+5]
            I_13 = img_in[y+2,x+5]
            I_14 = img_in[y+4,x+4]
            I_15 = img_in[y+5,x+2]
            S_0  = I_0  + I_8  - I_4  - I_12
            S_1  = I_1  + I_9  - I_5  - I_13
            S_2  = I_2  + I_10 - I_6  - I_14
            S_3  = I_3  + I_11 - I_7  - I_15
            D_0  = I_0  - I_8
            D_1  = I_1  - I_9
            D_2  = I_2  - I_10
            D_3  = I_3  - I_11
            D_4  = I_4  - I_12
            D_5  = I_5  - I_13
            D_6  = I_6  - I_14
            D_7  = I_7  - I_15
            
            if S_0 < 0:
                S_0 *= -1
            if S_1 < 0:
                S_1 *= -1
            if S_2 < 0:
                S_2 *= -1
            if S_3 < 0:
                S_3 *= -1
            
            if D_0 < 0:
                D_0 *= -1
            if D_1 < 0:
                D_1 *= -1
            if D_2 < 0:
                D_2 *= -1
            if D_3 < 0:
                D_3 *= -1
            if D_4 < 0:
                D_4 *= -1
            if D_5 < 0:
                D_5 *= -1
            if D_6 < 0:
                D_6 *= -1
            if D_7 < 0:
                D_7 *= -1
            
            value += S_0
            value += S_1
            value += S_2
            value += S_3
            value -= D_0
            value -= D_1
            value -= D_2
            value -= D_3
            value -= D_4
            value -= D_5
            value -= D_6
            value -= D_7
            
            if value > 0:
                img_out[y,x] = value
                if value > max_value:
                    max_value = value
            else:
                img_out[y,x] = <DTYPE_t>0
    
    if normalize:
        img_out *= (NORMALIZE_TO/<double>max_value)

    cv2.imwrite(outfname, img_out.astype(np.uint8))

