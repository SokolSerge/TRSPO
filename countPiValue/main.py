import time
from mpi4py import MPI
import random

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

num_points = 10000

# Main process generates points
if rank == 0:
    start_time = time.time()
    points = [(random.random(), random.random()) for _ in range(num_points)]
    # Divide the points among the processes
    chunks = [points[i::size] for i in range(size)]
else:
    chunks = None

start_time_local = time.time()
# Scatter the points to the processes
points = comm.scatter(chunks, root=0)

# Each process counts the number of points inside the unit circle
pointsInCircle = sum(x * x + y * y <= 1 for x, y in points)

localPiEstimate = 4 * pointsInCircle / len(points)
localTime = time.time() - start_time_local

# Gather the results from all processes
results = comm.gather((localPiEstimate, localTime), root=0)

global_pi_estimate = sum(result[0] for result in results) / size
global_time = time.time() - start_time
print("Rank: ", rank, "\nNumber of points: ", num_points, "\nTotal time: ", global_time,
      "\nPi value: ", global_pi_estimate)
