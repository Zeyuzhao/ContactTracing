# Generate fractional solution from problem
from ortools.linear_solver import pywraplp
from ortools.linear_solver.pywraplp import Variable, Constraint, Solver, Objective

from graph_tool.all import *

g = Graph()
ID = {} # id to vertex number
vertexID = g.new_vertex_property("int") # vertex to id
delimiter = ','
# load graph data from edge list
with open("../data/mont/mon1k.csv", "r") as f:
    for i, line in enumerate(f):
        split = line.split(delimiter)
        id1 = int(split[0])
        id2 = int(split[1])

        print("line {}: {} {}".format(i, id1, id2))
        # Retrieve vertices or create a new vertex
        if id1 not in ID:
            v1 = g.add_vertex()
            vertexID[v1] = id1
            ID[id1] = v1
        else:
            v1 = ID[id1]

        if id2 not in ID:
            v2 = g.add_vertex()
            vertexID[v2] = id2
            ID[id2] = v2
        else:
            v2 = ID[id2]

        g.add_edge(id1, id2)
        if (i > 10):
            break
g.vertex_properties["vertexID"] = vertexID

# display graph
graph_draw(g, vertex_size=1, edge_pen_width=1.2, output="output/mont.pdf")





