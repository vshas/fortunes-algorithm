import numpy as np
from utils import *
from binary_heap import Heap
from interval_red_black_tree import IntervalRBTree
import math
import matplotlib.pyplot as plt


class ExtendedIntervalRBTree(IntervalRBTree):
    """
    Extension of the IntervalRBTree object such that the elements in the leafs depend on the sweep_line object parameter. This creates, in a
    sense, dynamic objects. Since, the particularities of the whole object can be altered with one attribute.
    """
    def __init__(self, items=None, sweep_line=None):
        super().__init__(items)
        self.line = sweep_line

    def update_sweep_line(self, line):
        self.line = line
        x = self._root
        self._update_arcs(x)
    
    def _update_arcs(self, x):
        x.item.update_sweep_line(self.line)
        if x.right != self._nil:
            self._update_arcs(x.right)
        if x.left != self._nil:
            self._update_arcs(x.left)
    
    def get_beach_line(self, bounds):
        if self._root is not None:
            return self._get_line(self._root, [], [], bounds)
    
    def _get_line(self, x, px, py, bounds):
        if x.left != self._nil:
            px, py  = self._get_line(x.left, px, py, bounds)
        
        hx, hy = x.item.get_line(bounds)
        px.extend(hx)
        py.extend(hy)

        if x.right != self._nil:
            px, py = self._get_line(x.right, px, py, bounds)
        
        return px, py


class Arc:
    def __init__(self, low, high, source=None, line=None, left_edge=None, right_edge=None, prv=None, nxt=None, circle_event=None):
        self.low = low
        self.high = high
        self.source = source
        self.line = line
        self.left_edge = left_edge
        self.right_edge = right_edge
        self.prv = prv
        self.nxt = nxt
        self.circle_event = circle_event

    def __lt__(self, other):
        return (self[0] - other[0] < -1e-12) or (abs(self[1] - other[0]) < 1e-12)

    def __getitem__(self, it):
        if it:
            if type(self.high) == list:
                return get_parabola_intersection_coordinates(*self.high, self.line)[0]
            else:
                return self.high
        else:
            if type(self.low) == list:
                return get_parabola_intersection_coordinates(*self.low, self.line)[0]
            else:
                return self.low

    def __setitem__(self, key, value):
        if key:
            self.high = value
        else:
            self.low = value

    def __str__(self):
        left = self[0] if type(self.low) == list else -np.Inf
        right = self[1] if type(self.high) == list else np.Inf
        return '[' + str(left) + ',' + str(right) + ')'

    def update_sweep_line(self, line):
        self.line = line
    
    def get_line(self, bounds):
        eps = 1e-9 if abs(self.source[1] - self.line) < 1e-9 else 0
        a, b, c = get_parabola_coefficients(self.source, self.line, eps)
        
        if type(self.low) == list:
            eps = 1e-9 if abs(self.low[0][1] - self.line) < 1e-9 else 0
            a_l, b_l, c_l = get_parabola_coefficients(self.low[0], self.line, eps)
            x_low = (-(b_l - b) + np.sqrt((b_l - b) ** 2 - 4 * (a_l - a) * (c_l - c))) / (2 * (a_l - a))
        else:
            x_low = -np.inf

        if type(self.high) == list:
            eps = 1e-9 if abs(self.high[1][1] - self.line) < 1e-9 else 0
            a_h, b_h, c_h = get_parabola_coefficients(self.high[1], self.line, eps)
            x_high = (-(b - b_h) + np.sqrt((b - b_h) ** 2 - 4 * (a - a_h) * (c - c_h))) / (2 * (a - a_h))
        else:
            x_high = np.inf
        
        x_low = np.maximum((-b - np.sqrt(b**2 - 4 * a * (c - bounds[3]))) / (2 * a), x_low)
        x_high = np.minimum((-b + np.sqrt(b**2 - 4 * a * (c - bounds[3]))) / (2 * a), x_high)
        x = np.linspace(x_low, x_high, 100)
        y = a*x**2 + b*x + c
        return x, y
    

class Edge:
    def __new__(cls, left_edge, right_edge):
        if left_edge.twin == right_edge and right_edge.twin == left_edge:
            return super(Edge, cls).__new__(cls)
        else:
            raise ValueError
    
    def __init__(self, left_edge, right_edge, line=None):
        self.left_edge = left_edge
        self.right_edge = right_edge
        self._line = line

    def __iter__(self):
        self._n = -1
        return self

    def __next__(self):
        self._n += 1
        if self._n < 2:
            return self[self._n]
        else:
            raise StopIteration

    def __getitem__(self, it):
        if it:
            return self.left_edge
        elif not it:
            return self.right_edge
        else:
            pass
    
    def __str__(self):
        return f'({str(self.left_edge)}, {str(self.right_edge)})'

    @property
    def line(self):
        self._line

    @line.setter
    def line(self, line):
        self.left_edge.line = line
        self.right_edge.line = line
        self._line = line

    @property
    def definite(self):
        return (self.left_edge.definite and self.right_edge.definite)

    def get_line_coordinates(self):
        return (self.left_edge.sink[0], self.right_edge.sink[0]), (self.left_edge.sink[1], self.right_edge.sink[1])
    

class HalfEdge:
    def __init__(self, sink=None, prv=None, nxt=None, twin=None, ridge_point=None, line=None):
        self._sink = sink
        self.prv = prv
        self.nxt = nxt
        self.twin = twin
        self.ridge_point = ridge_point
        self.line = line

    def __str__(self):
        return '(' + str(self.sink[0]) + ',' + str(self.sink[1]) + ')'

    @property
    def definite(self):
        if type(self._sink) == Point:
            return True
        else:
            return False

    @property
    def sink(self):
        if type(self._sink) == list:
            return get_parabola_intersection_coordinates(*self._sink, self.line) # hier gaat hij huilen
        else:
            return self._sink

    @sink.setter
    def sink(self, sink):
        self._sink = sink

    @sink.deleter
    def sink(self):
        del self._sink

    def get_line(self):
        return (self.sink[0], self.twin.sink[0]), (self.sink[1], self.twin.sink[1])


class EdgeList:
    """
    Datastructure for storing edges and vertices.
    """
    def __init__(self, edges=[], vertices=[], sweep_line=None):
        self.edges = edges
        self.vertices = vertices
        self.line = sweep_line
    
    def __getitem__(self, it):
        return self.edges[it]

    def update_sweep_line(self, line):
        self.line = line
        for edge in self.edges:
            edge.line = self.line
    
    def get_selected_edges(self, definite_ids):
        return tuple(edge for edge in self.edges if id(edge) not in definite_ids)


class Event:
    """
    Object corresponding to an event in Fortune's Algorithm.
    """
    def __init__(self, key, point, is_site=True, is_valid=True, vanishing_arc=None):
        self.key = key
        self.point = point
        self.is_site = is_site
        self.is_valid = is_valid
        self.vanishing_arc = vanishing_arc

    def __repr__(self):
        return "{}({!r})".format(self.__class__.__name__, self.__dict__)


class VoronoiDiagram:
    def __init__(self, points, bounds=None, verbose=False, number_of_ticks=None, speed=None):
        self.points = points
        self._points = [Point(*point) for point in points]
        self.bounds = bounds if bounds is not None else self._set_bounds()
        self.verbose = verbose
        self.plot_params = self._set_plot_params(number_of_ticks, speed)
        if self.points is not None:
            edge_list = self.fit()
            self._postprocessing(edge_list)
        self.sweep_line = None 

    def _set_plot_params(self, number_of_ticks, speed):
        params= {}
        params['fig'] = None
        params['ax'] = None
        params['ticks'] = self._set_ticks(number_of_ticks)
        params['speed'] = speed
        params['remaining_tick'] = None
        return params

    def _set_bounds(self):
        a = 0.1
        x_min = min([p[0] for p in self.points])
        x_max = max([p[0] for p in self.points])
        y_min = min([p[1] for p in self.points])
        y_max = max([p[1] for p in self.points])
        return (math.floor(x_min - a * (x_max - x_min)), math.ceil(x_max + a * (x_max - x_min)),
                  math.floor(y_min - a * (y_max - y_min)), math.ceil(y_max + a * (y_max - x_min)))

    @staticmethod
    def _new_halve_edges(left_sink, right_sink, left_ridge_point, right_ridge_point):
        left_edge = HalfEdge(sink=left_sink, ridge_point=left_ridge_point)
        right_edge = HalfEdge(sink=right_sink, ridge_point=right_ridge_point)
        left_edge.twin, right_edge.twin = right_edge, left_edge
        if left_edge.ridge_point.edge_id is None:
            left_edge.ridge_point.edge_id = id(left_edge)
        if right_edge.ridge_point.edge_id is None:
            right_edge.ridge_point.edge_id = id(right_edge)
        return left_edge, right_edge

    @staticmethod
    def _set_neighbors(prv_edge, nxt_edge):
        nxt_edge.prv = prv_edge
        prv_edge.nxt = nxt_edge 

    def fit(self):
        """
        This function calculates the Voronoi diagram for points on a 2d plane. 
        """
        queue = Heap([Event(key=point[1], point=point) for point in self._points])
        status_structure = ExtendedIntervalRBTree()
        edge_list = EdgeList()

        # Get the first site event. 
        self.sweep_line, event = queue.extract_max()
        self.start = self.sweep_line
        left_edge, right_edge = HalfEdge(), HalfEdge()
        status_structure.insert(Arc(low=-np.Inf, high=np.Inf, line=self.sweep_line, source=event.point, left_edge=left_edge, 
                                right_edge=right_edge))
        
        if self.verbose:
            self._plot(status_structure, edge_list)

        while len(queue):
            # extract event from queue
            self.sweep_line, event = queue.extract_max()
            if not event.is_valid:
                continue
            
            if self.verbose:
                self._plot(status_structure, edge_list)

            status_structure.update_sweep_line(self.sweep_line)

            if event.is_site:
                self._handle_site_event(queue, status_structure, edge_list, event.point)
            else:
                self._handle_circle_event(queue, status_structure, edge_list, event.vanishing_arc, event.point)
        
        if self.verbose:
            self.sweep_line -= (self.bounds[3] - self.bounds[2]) 
            self._plot(status_structure, edge_list)
        
        plt.show(block=True)
        return edge_list

    def _partition_arc(self, atop_arc, point, edge_list):
        """
        Partitions the original (atop_arc) into a new arc corresponding to the event point, and two arcs corresponding to the left and
        right side of the original (atop_arc) respectively.
        """
        # construct new edges for the new arc.
        left_edge, right_edge = VoronoiDiagram._new_halve_edges(left_sink=[atop_arc.source, point], right_sink=[point, atop_arc.source],
                                                                left_ridge_point=atop_arc.source, right_ridge_point=point)

        # mid_arc is the completely new arc corresponding to the event point.
        mid_arc = Arc(low=[atop_arc.source, point], high=[point, atop_arc.source], source=point, line=self.sweep_line, left_edge=left_edge,
                      right_edge=right_edge)

        # left_arc is the left part of the original arc (atop_arc).
        left_arc = Arc(low=atop_arc.low, high=[atop_arc.source, point], source=atop_arc.source, line=self.sweep_line,
                       left_edge=atop_arc.left_edge, right_edge=left_edge, prv=atop_arc.prv, nxt=mid_arc)
        
        # right_arc is the right part of the original arc (atop_arc).
        right_arc = Arc(low=[point, atop_arc.source], high=atop_arc.high, source=atop_arc.source, line=self.sweep_line,
                        left_edge=right_edge, right_edge=atop_arc.right_edge, prv=mid_arc,
                        nxt=atop_arc.nxt)

        # link the newly created arcs together.
        mid_arc.prv, mid_arc.nxt = left_arc, right_arc
        if atop_arc.prv is not None:
            atop_arc.prv.nxt = left_arc
        if atop_arc.nxt is not None:
            atop_arc.nxt.prv = right_arc

        # bookkeeping of the output data.
        new_edge = Edge(left_edge, right_edge)
        left_edge.ridge_point.edge_id = id(new_edge)
        right_edge.ridge_point.edge_id = id(new_edge)
        edge_list.edges.append(new_edge)

        return left_arc, mid_arc, right_arc

    def _handle_site_event(self, queue, status_structure, edge_list, event_point):
        """
        Updates the status_structure datastructure according to the site event event point given by event_point. This implies deconstructing
        one arc in the status_structure and reconstructing it with one additional arc.
        """
        #  find the arc above the event point
        atop_arc = status_structure.search((event_point[0], event_point[0]))

        # remove atop arc and split into 3 pieces.
        status_structure.delete(atop_arc)
        if atop_arc.circle_event is not None:
            atop_arc.circle_event.is_valid = False
        left_arc, mid_arc, right_arc = self._partition_arc(atop_arc, event_point, edge_list)

        # insert the three new arcs into the status structure to fix the chasm in the beachline.
        for arc in (left_arc, mid_arc, right_arc):
            status_structure.insert(arc)

        # check for potential circle events
        if atop_arc.prv is not None:
            circle_event = self._potential_circle_event(atop_arc.prv.source, left_arc.source, mid_arc.source, left_arc)
            if circle_event is not None:
                left_arc.circle_event = circle_event
                queue.insert(circle_event)
        if atop_arc.nxt is not None:
            circle_event = self._potential_circle_event(mid_arc.source, right_arc.source, atop_arc.nxt.source, right_arc)
            if circle_event is not None:
                right_arc.circle_event = circle_event
                queue.insert(circle_event)

    def _potential_circle_event(self, p1, p2, p3, vanishing_arc):
        """
        Check if the 3 input point could potentially elicit a circle event in the future. The potential resulting circle event could be rendered
        unvalid if some other event changes the local dynamics.
        """
        # get the parameters of the bisectors between consecutive points.
        eps=1e-3
        a1, b1 = bisector(p1, p2)
        a2, b2 = bisector(p2, p3)
        if a1 is None and a2 is None:
            return None

        # if both bisectors aim towards the instersection point of the bisectors,
        if bisectors_are_convergent((a1, b1), (a2, b2), p1, p2, p3, eps):
            # then calculate the intersection point.
            if a1 is None:
                x = b1
                y = a2 * x + b2
            elif a2 is None:
                x = b2
                y = a1 * x + b1
            else:
                x = (b1 - b2) / (a2 - a1)
                y = a1 * x + b1
            # The point that will trigger the circle event lies below the corresponding y coordinate.
            y_hat = y - euclidean((x, y), p1)
            if y_hat < self.sweep_line:
                return Event(key=y_hat, point=Point(x, y), is_site=False, vanishing_arc=vanishing_arc)
        return None

    def _handle_circle_event(self, queue, status_structure, edge_list, vanishing_arc, event_point):
        """
        Updates the status_structue according to a circle event with event point given by event_point. The vansihing_arc disappears and the
        neighboring arcs are tight togeter accordingly.
        """
        # fix chasm in the beach line (symmetric)
        vanishing_arc.prv.high[1] = vanishing_arc.nxt.source
        vanishing_arc.nxt.low[0] = vanishing_arc.prv.source
        vanishing_arc.prv.nxt = vanishing_arc.nxt
        vanishing_arc.nxt.prv = vanishing_arc.prv

        # set all future circle events where the current vanishing arc is the vanishing arc to invalid
        if vanishing_arc.prv.circle_event is not None:
            vanishing_arc.prv.circle_event.is_valid = False
        if vanishing_arc.nxt.circle_event is not None:
            vanishing_arc.nxt.circle_event.is_valid = False
        edge_list.vertices.append(event_point)

        # set new tracing out edges
        vanishing_arc.left_edge.sink = event_point
        vanishing_arc.right_edge.sink = event_point
        left_edge, right_edge = VoronoiDiagram._new_halve_edges(left_sink=[vanishing_arc.prv.source, vanishing_arc.nxt.source], 
                                                                right_sink=event_point, left_ridge_point=vanishing_arc.prv.source, 
                                                                right_ridge_point=vanishing_arc.nxt.source)
        # create links between the edges and append to edge_list.
        VoronoiDiagram._set_neighbors(vanishing_arc.prv.right_edge, left_edge)
        VoronoiDiagram._set_neighbors(vanishing_arc.nxt.left_edge, vanishing_arc.prv.right_edge.twin)
        VoronoiDiagram._set_neighbors(right_edge, vanishing_arc.nxt.left_edge.twin) 
        edge_list.edges.append(Edge(left_edge, right_edge))

        # update the edges kept by the arcs.
        vanishing_arc.nxt.left_edge = left_edge
        vanishing_arc.prv.right_edge = left_edge

        # check for potential circle events
        if vanishing_arc.prv.prv is not None:
            circle_event = self._potential_circle_event(vanishing_arc.prv.prv.source,
                                                                  vanishing_arc.prv.source,
                                                                  vanishing_arc.nxt.source, vanishing_arc.prv)
            if circle_event is not None:
                vanishing_arc.prv.circle_event = circle_event
                queue.insert(circle_event)

        if vanishing_arc.nxt.nxt is not None:
            circle_event = self._potential_circle_event(vanishing_arc.prv.source, vanishing_arc.nxt.source,
                                                                  vanishing_arc.nxt.nxt.source, vanishing_arc.nxt)
            if circle_event is not None:
                vanishing_arc.nxt.circle_event = circle_event
                queue.insert(circle_event)
        status_structure.delete(vanishing_arc)

    def _plot(self, status_structure, edge_list):
        """
        Umbrella function for plotting the progression of fortunes algorithm. The plotting of Fortune's Algorithm is carried out according to 
        increments (ticks). Before an event is handled in the main loop (.fit()) the plot is progressed from the previous sweep_line to the 
        current sweep_line according to small increments.
        """
        # when the function is called for the first time, initialize.
        if self.plot_params['fig'] is None:
            self._plot_initialize()

        # if there is a valid increment remaining from the previous iteration then plot this increment.
        if self.plot_params['remaining_tick'] is not None:
            if self.sweep_line < self.plot_params['remaining_tick']:
                self._plot_state(status_structure, edge_list, self.plot_params['remaining_tick'])
                self.plot_params['remaining_tick'] = None
        
        # Progress the plot until the current increment is smaller than the current sweep_line. 
        for tick in self.plot_params['ticks']:
            if tick >= self.sweep_line:
                self._plot_state(status_structure, edge_list, tick)
            else:
                # keep the remaining increment for the following iteration.
                self.plot_params['remaining_tick'] = tick
                return
 
    def _plot_initialize(self):
        """
        Initialize all plot related objects.
        """
        plt.ion()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axis(self.bounds)
        ax.set_xlim(self.bounds[:2])
        ax.set_ylim(self.bounds[2:])
        ax.scatter([p[0] for p in self.points], [p[1] for p in self.points], s=2, c='k')
        self.plot_params['fig'] = fig
        self.plot_params['ax'] = ax
        self.plot_params['x'] = np.linspace(self.bounds[0], self.bounds[1], 100)
        self.plot_params['y'] = np.array([self.bounds[3]]*100)
        self.plot_params['definite_ids'] = ()
    
    def _plot_state(self, status_structure, edge_list, line):
        """
        Plots the state of Fortune's Algorithm at a particular time (increment). The details of status_structure and edge_list depend on the
        corresponding virtual sweepline. That is, the line corresponds to an increment set in thet parrent function (_plot()).
        """
        # Update the datastructures to the virtual sweepline.
        status_structure.update_sweep_line(line)
        edge_list.update_sweep_line(line) 

        # get the x and y coordinates of the beachline.
        px, py = status_structure.get_beach_line(self.bounds)

        # explicitly remove the previous beachline and sweepline (from the previous increment) from the 'ax' object.
        if self.plot_params['ax'].lines:
            del self.plot_params['ax'].lines[-2:]

        # explicitly remove the edges of the previous voronoi diagram.
        k = len(self.plot_params['ax'].lines) - len(self.plot_params['definite_ids'])
        del self.plot_params['ax'].lines[-k:]
        
        # add the edges of the current voronoi diagram to the plot object and keep track of the definite edges.
        edges = edge_list.get_selected_edges(self.plot_params['definite_ids'])
        new_definites = tuple(edge for edge in edges if edge.definite)
        for edge in new_definites:
            self.plot_params['ax'].plot(*edge.get_line_coordinates(), 'k', linewidth=1)
        self.plot_params['definite_ids'] = self.plot_params['definite_ids'] + tuple(id(edge) for edge in new_definites)

        # plot the non definite edges.
        for edge in tuple(set(edges) ^ set(new_definites)):
            self.plot_params['ax'].plot(*edge.get_line_coordinates(), 'r', linewidth=1)

        # set the new beachline and sweepline to the plot object (corresponding to the current increment).
        self.plot_params['ax'].plot(px, py, 'k', linewidth=1)
        self.plot_params['ax'].plot(np.linspace(self.bounds[0], self.bounds[1], 100), np.array([line]*100), 'k', linewidth=1)

        # fill the area from the topline to the beach line (and update the top line).
        xf = np.concatenate((px, self.plot_params['x'][::-1]))
        yf = np.concatenate((py, self.plot_params['y'][::-1]))
        self.plot_params['ax'].fill(xf, yf, '#FFA500', alpha=0.1)
        plt.pause(self.plot_params['speed'])
        plt.draw()
        self.plot_params['x'] = px
        self.plot_params['y'] = py

    def _set_ticks(self, num):
        """
        Set the increments.
        """
        start = max(p[1] for p in self.points)
        dx = ((start - self.bounds[2])*2) / num # it was 1.35
        ticks = iter([-dx * k + start for k in range(num+1)])
        return ticks
    
    def _postprocessing(self, edge_list):
        """
        Process the resulting information obtained from Fortune's Algorithm in such a way that it corresponds to the data in the Voronoi object
        in the 'scipy.spatial' library.
        """
        self._vertices = edge_list.vertices
        self._edges = edge_list.edges
        print(len(self._vertices))

        # initialize lookup tables
        vertex_dict = {id(vertex): ix for ix, vertex in enumerate(self._vertices)}
        edge_dict = {id(edge): ix for ix, edge in enumerate(self._edges)}
        point_dict = {id(point): ix for ix, point in enumerate(self._points)}

        # get ridge vertices (need to check but seems oke, also need to check if ridge points correspond to ridge vertices)
        self.ridge_vertices = [sorted([vertex_dict[id(he.sink)] if he.definite else -1 for he in edge]) for edge in self._edges]
        self.ridge_points = np.array([[point_dict[id(he.ridge_point)] for he in edge] for edge in self._edges])

        # get regions
        self.regions = []
        point_region = []
        for ix, p in enumerate(self._points):
            point_region.append(point_dict[id(p)])
            edge = self._edges[edge_dict[p.edge_id]]
            if edge.left_edge.ridge_point == p:
                half_edge = edge.left_edge
            else:
                half_edge = edge.right_edge

            if ix == 5:
                print(f'p: {p}')
                print(f'edge: {half_edge}')
                print(f'nxt: {half_edge.nxt}')

            self.regions.append(VoronoiDiagram._construct_region(half_edge, vertex_dict)) 
        
        self.points = np.array([[p[0],p[1]] for p in self._points])
        self.vertices = np.array([[v[0],v[1]] for v in self._vertices])
        self.point_region = np.array(point_region)

    @staticmethod 
    def _construct_region(half_edge, vertex_dict):
        """
        construct a region as in 'scipy.spatial'.
        """
        region = []
        current_edge = half_edge
        while current_edge.definite:
            region.append(vertex_dict[id(current_edge.sink)])
            current_edge = current_edge.nxt
            if current_edge == half_edge:
                return region[::-1]

        region.append(-1)
        current_edge = current_edge.prv
        while current_edge is not None:
            region.append(vertex_dict[id(current_edge.sink)])
            current_edge = current_edge.prv
        
        return region[::-1]

