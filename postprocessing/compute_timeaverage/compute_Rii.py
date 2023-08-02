#NOTE: nfiles must be divisible by number of processors
from mpi4py import MPI
import numpy as np
import xarray as xr
import h5py
import os

# Function definitions 
def read_data_cbd(file_path):
    '''
    read_cbd: reads the cbd file data
    Arguments:
    file_path-- path where the file is located
    Return--
    xr_data-- data_matrix: 3-D matrix of size (N_x, N_y, N_z)
          x-coord: 1-D array of  x-coordinates size (N_x,1)
          y-coord: 1-D array of  y-coordinates size (N_y,1)
          z-coord: 1-D array of  z-coordinates size (N_z,1)
    '''
    f = open(file_path, "rb")
    id = np.fromfile(f, dtype=np.int64, count=1)[0]
    if id == 288230376151834571:
        T = np.float32
    elif id == 576460752303546315:
        T = np.float64
    else:
        raise ValueError("Invalid ID")
    N = tuple(np.fromfile(f, dtype=np.int64, count=3))
    xmin = tuple(np.fromfile(f, dtype=np.float64, count=3))
    xmax = tuple(np.fromfile(f, dtype=np.float64, count=3))
    data1 = np.fromfile(f, dtype=T, count=np.prod(N)).reshape(N, order="F")
    f.close()
    return data1

def read_coords_cbd(file_path):
    '''
    read_cbd: reads the cbd file data
    Arguments:
    file_path-- path where the file is located
    Return--
    xr_data-- data_matrix: 3-D matrix of size (N_x, N_y, N_z)
          x-coord: 1-D array of  x-coordinates size (N_x,1)
          y-coord: 1-D array of  y-coordinates size (N_y,1)
          z-coord: 1-D array of  z-coordinates size (N_z,1)
    '''
    f = open(file_path, "rb")
    id = np.fromfile(f, dtype=np.int64, count=1)[0]
    if id == 288230376151834571:
        T = np.float32
    elif id == 576460752303546315:
        T = np.float64
    else:
        raise ValueError("Invalid ID")
    N = tuple(np.fromfile(f, dtype=np.int64, count=3))
    xmin = tuple(np.fromfile(f, dtype=np.float64, count=3))
    xmax = tuple(np.fromfile(f, dtype=np.float64, count=3))
    x1, x2, x3 = (np.fromfile(f, dtype=np.float64, count=n) for n in N)
    f.close()
    return x1, x2, x3

def initialize_MPI(MPI):
    comm = MPI.COMM_WORLD
    rank = comm.rank
    nproc = comm.size
    
    return comm, rank, nproc

def compute_MPI_params(nfiles, nproc, rank):

    #This function returns chunk size handeled by each processor
    chunk = int(nfiles / nproc)      # Amount of tsteps handled by each processor
    #remdr = (nfiles % nproc) * int((rank + 1) / nproc)   # Remaining amount of tsteps handled by the last processor incase tsteps is not divisible by nproc
    remdr = nfiles % nproc if rank == nproc - 1 else 0
    return chunk, remdr

def read_vel_field(input_path, var_name, num):
    f = read_data_cbd('%s/%s-%s.cbd' % (input_path, var_name, num))
    return f

def read_vel_field_coords(input_path, var_name, num):
    x, y, z = read_coords_cbd('%s/%s-%s.cbd' % (input_path, var_name, num))
    return x, y, z

def add_to_average(avg, f):
    avg += f       
    return avg

def divide_by_numfiles(chunk, remdr, avg):
    numfiles = chunk + int(remdr)
    avg /= numfiles    
    return avg

def get_resolution(var):
    return var.shape

def compute_average(rank, chunk, remdr, input_path, var_name, num_list):
    var = read_vel_field(input_path, var_name, num_list[0])
    resolution = get_resolution(var)
    tavg_var = np.zeros(resolution)

    for num in num_list[chunk*rank:chunk*(rank + 1) + int(remdr)]:
        var = read_vel_field(input_path, var_name, num)
        tavg_var = add_to_average(tavg_var, var)

    tavg_var = divide_by_numfiles(chunk, remdr, tavg_var)
    
    return tavg_var, resolution

def initialize_recv_buffer(nproc, resolution):
    recv = np.empty((nproc, resolution[0], resolution[1], resolution[2]))   
    return recv

def gather_var(comm, avg, recv):
    comm.Gather(avg, recv, root=0) 

def compute_final_average(recv):
    final_avg = np.mean(recv, axis=0) 
    return final_avg

def write_HDF5(input_path, var_name, output_path, avg, num, h5py):

    # Getting the coordinate data
    xcoord, ycoord, zcoord = read_vel_field_coords(input_path, var_name, num)

    hf = h5py.File(output_path, 'w')
    hf.create_dataset('%s_rii' % var_name, data=avg)
    hf.create_dataset('x-coord', data=xcoord)
    hf.create_dataset('y-coord', data=ycoord)
    hf.create_dataset('z-coord', data=zcoord)
    hf.close()

def autocorrelation2d_horizontal_i(q, resolution, qmean):
    norm=1.0/resolution[0]/resolution[1]
    R = np.zeros(resolution)
    for jk in range(resolution[2]):
        q_ = q[:,:,jk]-qmean[jk]
        q_hat = np.fft.fft2(q_)
        q_hat_conj = q_hat.conjugate()
        R_ = np.fft.ifft2(q_hat*q_hat_conj)*norm
        R[:,:,jk] = R_
    return R

def h5_in_xarray(filepath, field):
    hf = h5py.File(filepath, 'r')
    data = hf.get('%s' % field)
    x1 = hf.get('x-coord')
    x2 = hf.get('y-coord')
    x3 = hf.get('z-coord')
    xr_data = xr.DataArray(
        np.array(data),
        dims = ("x", "y", "z"),
        coords = { "x" : np.array(x1),
                  "y" : np.array(x2),
                  "z" : np.array(x3)
        }
    )

    return xr_data

def compute_autocorrelation(rank, chunk, remdr, input_path, var_name, var_mean, num_list):
    var = read_vel_field(input_path, var_name, num_list[0])
    resolution = get_resolution(var)
    Rii_avg = np.zeros(resolution)

    for num in num_list[chunk*rank:chunk*(rank + 1) + int(remdr)]:
        var = read_vel_field(input_path, var_name, num)
        Rii_var = autocorrelation2d_horizontal_i(var, resolution, var_mean)
        Rii_avg = add_to_average(Rii_avg, Rii_var)

    Rii_avg = divide_by_numfiles(chunk, remdr, Rii_avg)
    
    return Rii_avg, resolution

# Input parameters to be specified by the users
folderpath = "/data/shared/atharva_chanFlowData/pressure_corrected_DomainAnn_longerx"
sim_list = os.listdir(folderpath)
#sim_list = [match for match in sim_list if "hxsx83" in match and "_16x24x96_restart2" in match]
input_path = "%s/%s/output/instantaneous-fields" % (folderpath, sim_list[0])
var_name_list = ["u", "v"]
files = os.listdir("%s" % input_path)
# getting num list (of u-{num}.cbd)
# this part of the code will depend on how your output is saved.
num_list = [match[2:-4] for match in files if "v" in match]
#num_list = [match[3:-4] for match in files if "v" in match]
num_list = ["{:05d}".format(int(num)) for num in num_list]
num_list.sort()
num_list = ["{:03d}".format(int(num)) for num in num_list]
#num_list = num_list[2000:]
nfiles = len(num_list)

# code
comm, rank, nproc = initialize_MPI(MPI)
chunk, remdr = compute_MPI_params(nfiles, nproc, rank)

for sim in sim_list:
    input_path = "%s/%s/output/instantaneous-fields" % (folderpath, sim)
    h5_mean_path = '/home/as2204/TACC_postprocess/longerx_averages/%s.h5' % sim

    for var_name in var_name_list:
        output_path = "/home/as2204/TACC_postprocess/longerx_averages/r%s%s_%s.h5" % (var_name, var_name, sim)
        var_mean = h5_in_xarray(h5_mean_path, '%s' % var_name)
        var_mean_z = var_mean.mean(dim=['x', 'y']).values
        Rii_avg, resolution = compute_autocorrelation(rank, chunk, remdr, input_path, var_name, var_mean_z, num_list)

        recv = None
        if rank == 0:
            recv = initialize_recv_buffer(nproc, resolution)

        comm.Barrier()
        gather_var(comm, Rii_avg, recv)

        if rank == 0:
            Rii_avg = compute_final_average(recv)
            write_HDF5(input_path, var_name, output_path, Rii_avg, num_list[0], h5py)
