#%%
from ortools.linear_solver import pywraplp
from ortools.linear_solver.pywraplp import Variable, Constraint, Objective

solver = pywraplp.Solver.CreateSolver("GLOP")
x = [solver.NumVar(0, 1, f"x{u}") for u in range(10)]
y = [solver.NumVar(0, 1, f"y{u}") for u in range(10)]
z = solver.NumVar(0, solver.infinity(), "z")

weights = [i for i in range(10)]

total_constraint = sum(x) == 5
z_definition_constraint = (sum([i*j for (i, j) in zip(x, weights)]) == z)

solver.Add(total_constraint)
solver.Add(z_definition_constraint)

#%%
num_exposed = solver.Objective()
num_exposed.SetCoefficient(z, 1)
num_exposed.SetMaximization()
solver.Solve()
#%%
print([i.solution_value() for i in x])
print(f"sum dual: {total_constraint.dual_value()}")
print(f"z dual: {z_definition_constraint.dual_value()}")
# %%
