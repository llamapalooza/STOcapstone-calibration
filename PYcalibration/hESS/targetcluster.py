from geom2d import Point, get_x, get_y, get_z, set_x, set_z, sqdist, A_2x, segments

def identify_boards(clusters):
    """
    todo
    """
    for cluster in clusters:
            cluster.highpass_filter()
            cluster.suppress_nonmaxima()
            cluster.order_points()

def kmeans_tc(points, centers, epsilon=5, max_iter=15):
    """
    todo
    """
    clusters = []
    for center in centers:
        clusters.append(TargetCluster(center))
    
    converging = True
    iteration = 0
    while converging and iteration < max_iter:
        # clear current labels
        for cluster in clusters:
            cluster.clear()
        
        # label points
        for point in points:
            label(point, clusters)
        
        # recalculate cluster centers
        for cluster in clusters:
            cluster.calc_center()
        
        iteration += 1
        
        # check for convergence
        converging = False
        for cluster in clusters:
            if cluster.d_center() > epsilon**2:
                converging = True
    
    return clusters

def label(point, clusters):
    """
    Adds a *point* to the nearest cluster in *clusters*.
    """
    min_sqdist = None
    closest_cluster = None
    
    for cluster in clusters:
        d = cluster.sqdist_to(point)
        if min_sqdist == None or d < min_sqdist:
            min_sqdist = d
            closest_cluster = cluster
    
    closest_cluster.add(point)

class TargetCluster(object):
    def __init__(self, center=None):
        """
        todo
        """
        self.points = []
        self.prevcenter = None
        self.center = center
        self.corners = None
    
    def size(self):
        """
        Returns the number of points in this cluster.
        """
        return len(self.points)
    
    # kmeans clustering:
    
    def sqdist_to(self, point):
        """
        Returns the squared distance of a point from the center of this cluster.
        """
        return sqdist(self.center, point)
    
    def add(self, point):
        """
        Adds a point to this cluster. Asks no questions.
        """
        self.points.append(point)
    
    def clear(self):
        """
        Removes all points from this cluster.
        """
        self.points = []
        self.corners = None
    
    def calc_center(self):
        """
        Finds the average coordinates of *self.points*, weighted by pixel intensity,
        and saves them in *self.center*.
        """
        N = 0
        Ex = 0
        Ey = 0
        for point in self.points:
            x = get_x(point)
            y = get_y(point)
            z = get_z(point)
            N += z
            Ex += z*x
            Ey += z*y
        self.prevcenter = self.center
        self.center = Point(Ex/N, Ey/N)
    
    def d_center(self):
        """
        Returns the most recent change in center coordinates as the squared distance
        between current and previous centers or *None* if no previous center exists.
        """
        if self.prevcenter == None:
            return None
        else:
            return sqdist(self.prevcenter, self.center)
    
    # corner point location:
    
    def highpass_filter(self, threshold=0.2):
        """
        todo
        """
        z_max = max([ get_z(point) for point in self.points ])
        z_cutoff = z_max * threshold
        
        highpoints = []

        for point in self.points:
            if get_z(point) > z_cutoff:
                highpoints.append(point)
        
        self.points = highpoints
    
    def suppress_nonmaxima(self, r=7):
        """
        Removes from this cluster all points within *r* pixels of a point with
        greater intensity. In the case of a tie, only the ``last-encountered''
        tied point is preserved.
        """
        local_maxima = []
        
        for p in self.points:
            z_p = get_z(p)
            is_max = True
            tie = False
            
            for q in [ other for other in self.points if other != p ]:
                z_q = get_z(q)
                if sqdist(p, q) < r**2:
                    if z_q > z_p:
                        is_max = False
                    elif z_q == z_p:
                        is_max = False
                        tie = True
            
            if is_max:
                # count it
                local_maxima.append(p)
            elif tie:
                # eliminate tie
                set_z(p, 0)
        
        self.points = local_maxima
    
    # corner point identification:
    
    def order_points(self):
        self._calc_corners()
    
    def _calc_corners(self):
        """
        Locates four corners to approximate the convex hull of *self.points* and
        saves them as *self.corners*.
        """
        corners = None
        max_area = 0
        for P in self._all_quads():
            area = A_2x(P)
            if area > max_area:
                max_area = area
                corners = P
        self.corners = corners
    
    def _all_quads(self):
        """
        Returns a list of all quadrilaterals with vertices from *self.points*.
        Vertex order is significant, so some quadrilaterals are ``equivalent.''
        """
        return [ [p0, p1, p2, p3]
                    for p0 in self.points
                    for p1 in self.points
                        if p1 != p0
                    for p2 in self.points
                        if p2 != p0 and p2 != p1
                    for p3 in self.points
                        if p3 != p0 and p3 != p1 and p3 != p2 ]
