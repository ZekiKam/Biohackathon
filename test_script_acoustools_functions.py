import torch
from acoustools.Utilities import create_points, propagate_abs, device, TOP_BOARD
from acoustools.Solvers import wgs
from acoustools.Visualiser import Visualise, ABC

# -------------------------------------------------
# 1. Setup (Full-stack: SETUP)
# -------------------------------------------------

board = TOP_BOARD  # standard 16x16 array

# Create a horizontal "channel" of points
N = 20
x_positions = torch.linspace(-0.04, 0.04, N)

points = torch.zeros((1, 3, N), device=device)
points[0, 0, :] = x_positions      # x varies → horizontal
points[0, 1, :] = 0.0              # y fixed
points[0, 2, :] = 0.05             # z height above board

# -------------------------------------------------
# 2. Solver (Full-stack: SOLVER)
# -------------------------------------------------

# WGS gives more uniform pressure across multiple points
activation = wgs(points, board=board)
print(activation.device)  # Should be (1, 16, 16)

# -------------------------------------------------
# 3. Visualisation plane (Full-stack: ANALYSIS)
# -------------------------------------------------

# XZ plane slice at y = 0
A, B, C = ABC(0.08, plane='xz', origin=torch.tensor((0,0,0), device=device))


# -------------------------------------------------
# 4. Visualise
# -------------------------------------------------

Visualise(
    A, B, C,
    activation,
    points=points,
    colour_functions=[propagate_abs],
    colour_function_args=[{'board': board}], 
    res=(300, 300),
    cmaps=["hot"],
)