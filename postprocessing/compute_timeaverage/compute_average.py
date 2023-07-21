#NOTE: nfiles must be divisible by number of processors
from mpi4py import MPI
import numpy as np
import xarray as xr
import h5py
from function_definitions import *
from input_variables import *

def main():
    comm = MPI.COMM_WORLD
    rank = comm.rank
    nproc = comm.size

    for sim in sim_list:
        filepath = "%s/%s/output/instantaneous-fields" % (folderpath, sim)
        file_path  = "%s/u-%s.cbd" % (filepath, num_list[0])
        u = read_cbd(file_path)
        xnodes, ynodes, znodes = get_dimensions(u)

        chunk, remdr = compute_MPI_params(nfiles, nproc, rank)
        avg = initialize_avg_fields(xnodes, ynodes, znodes)
        first_ord_fields, second_ord_fields, indv_fields = seperate_orders()

        for num in num_list[chunk*rank:chunk*(rank + 1) + int(remdr)]:
            if rank == 0:
                if int(num) % 10 == 0:
                    print("Current sim %s, Iterations completed %s/%d" % (sim, num, len(num_list[0:chunk])))
            f = read_vel_fields(indv_fields, filepath, num)
            avg = add_to_average(first_ord_fields, second_ord_fields, f, avg)

        avg = divide_by_numfiles(chunk, remdr, avg)

        recv = empty_recv_data()
        if rank == 0:
            recv = initialize_recv_data(nproc, xnodes, ynodes, znodes, recv)

        comm.Barrier()
        gather_vars(comm, avg, recv)
    
        if rank == 0:
            avg = compute_average(recv)
            write_HDF5(destination, sim, avg, u, h5py)




if __name__ == '__main__':
    main()
