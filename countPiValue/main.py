import numpy as np
from mpi4py import MPI

if __name__ == '__main__':
    n_points = int(sys.argv[1])
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    processes_num = comm.Get_size()