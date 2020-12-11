from ortools.linear_solver import pywraplp
from ortools.linear_solver.pywraplp import Variable, Constraint, Solver, Objective

from graph_tool.all import *

g = load_graph_from_csv("../data/mont/mon10k.csv",
                        directed=True)
print("done loading")
# Draw Graph
graph_draw(g, vertex_size=1, edge_pen_width=.8, output="../output/mont10k.pdf")