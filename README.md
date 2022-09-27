# Fortunes-algorithm

Python implementation of Fortune's Algorithm. Which is an algorithm for obtaining the Voronoi diagram of a set of points (in 2d). This algorithm has a worst-case complexity of $O(n \ln n)$. The file <tt>voronoi_diagram.py</tt> contains the <tt>Voronoi</tt> class which is similar to the <tt>scipy.spatial.Voronoi</tt> class. Even though the worst-case complexity is optimal, this implentation is not optimized for speed. Basically, the only reason to use the code is to get a better sense of how Fortune's Algorithm works and if you like visuals. By setting <tt>verbose=True</tt> this implementation gives a progress plot of the algorithm.

![Example of progress plot at a specific increment.](https://github.com/vshas/fortunes-algorithm/blob/main/example.png?raw=true)

<img src="https://github.com/vshas/fortunes-algorithm/blob/main/example.png?raw=true" width="480" height="480">
