import numpy as np

class Point:
    def __init__(self, x, y, edge_id=None):
       self.x = x
       self.y = y
       self.edge_id = edge_id

    def __getitem__(self, it):
        if it:
            return self.y
        elif not it:
            return self.x
        else:
            pass
    
    def __setitem__(self, key, value):
        if key:
            self.y = value
        elif not key:
            self.x = value
        else:
            pass
    
    def __str__(self) -> str:
        return '(' + str(self.x) + ',' + str(self.y) + ')'


def euclidean(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

def on_the_right(p1, p2, p3):
    return (p2[0] - p1[0])*(p3[1] - p1[1]) - (p2[1] - p1[1])*(p3[0] - p1[0]) < 0

def bisector(p1, p2):
    if abs(p1[1] - p2[1]) < 1e-9:
        return None, (p1[0] + p2[0])/2
    else:
        a = -(p2[0] - p1[0]) / (p2[1] - p1[1])
        b = (p1[1] + p2[1])/2 - a*(p1[0] + p2[0])/2
        return a, b

def bisectors_are_convergent(bis1, bis2, p1, p2, p3, eps):
    mid12 = ((p1[0] + p2[0])/2, (p1[1] + p2[1])/2)
    mid23 = ((p2[0] + p3[0])/2, (p2[1] + p3[1])/2)

    if abs(p1[1] - p2[1]) < 1e-9:
        r1 = (mid12[0], p1[1] + eps) if on_the_right(p1, p2, (mid12[0], p1[1] + eps)) else (mid12[0], p1[1] - eps)
    else:
        r1 = (mid12[0] + eps, mid12[1] + bis1[0] * eps) if on_the_right(p1, p2, (mid12[0] + eps, mid12[1] + bis1[0]* eps)) \
            else (mid12[0] - eps, mid12[1] - bis1[0]* eps)

    if abs(p2[1] - p3[1]) < 1e-9:
        r2 = (mid23[0], p2[1] + eps) if on_the_right(p2, p3, (mid23[0], p2[1] + eps)) else (mid23[0], p2[1] - eps)
    else:
        r2 = (mid23[0] + eps, mid23[1] + bis2[0]* eps) if on_the_right(p2, p3, (mid23[0] + eps, mid23[1] + bis2[0]* eps)) \
            else (mid23[0] - eps, mid23[1] - bis2[0]* eps)
    hold = euclidean(r1, r2) < euclidean(mid12, mid23) 
    return hold

def get_parabola_coefficients(point, line, eps=0):
    a = 1 / (2 * (point[1] - line + eps))
    b = -2 * point[0] * a
    c = (point[0]**2 + point[1]**2 - line**2) * a
    return a, b, c

def get_parabola_intersection_coordinates(left_point, right_point, line):
    if abs(left_point[1] - line) < 1e-9:
        return left_point
    elif abs(right_point[1] - line) < 1e-9:
        return right_point
    else:
        a1, b1, c1 = get_parabola_coefficients(left_point, line)
        a2, b2, c2 = get_parabola_coefficients(right_point, line)
        if abs(a1 - a2) < 1e-9:
            x = (left_point[0] + right_point[0])/2
        else:
            x = (-(b1 - b2) + np.sqrt((b1-b2) ** 2 - 4 * (a1 - a2) * (c1 - c2))) / (2 * (a1 - a2))
        return Point(x, a1*x**2 + b1*x + c1)

