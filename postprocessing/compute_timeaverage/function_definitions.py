import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from scipy.fft import fft
from input_variables import *

def read_cbd(file_path):
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
    data1 = np.fromfile(f, dtype=T, count=np.prod(N)).reshape(N, order="F")
    #dic = {"data":data1, "x-coord":x1, "y-coord":x2, "z-coord":x3, "diagonal_points":[xmin,xmax]}
    xr_data = xr.DataArray(
        data1,
        dims = ("x", "y", "z"),
        coords = { "x" : x1,
                  "y" : x2,
                  "z" : x3
        }
    )
    f.close()
    return xr_data

def h5_in_xarray(filepath):
    hf = h5py.File(filepath, 'r')
    data = hf.get('data')
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

def create_xarray(u):

    # This function creates an xarray that has same data size and coordinates aligned with the input xarray u
    new_array = xr.DataArray(
        np.empty(u.shape),
        dims = ("x", "y", "z"),
        coords = { "x" : u['x'],
              "y" : u['y'],
              "z" : u['z']
        }
    )

    return new_array


def numpy2xarray(u_numpy, u):

    # This function creates an xarray that has same data size and coordinates aligned with the input xarray u
    new_array = xr.DataArray(
        u_numpy,
        dims = ("x", "y", "z"),
        coords = { "x" : u['x'],
              "y" : u['y'],
              "z" : u['z']
        }
    )

    return new_array

def intp_w_to_uvp(w, u):

    # This function interpolates w to uvp nodes
    w_intp = create_xarray(u)
    w_intp[:, :, :-1] = w.interp(z = u['z'][:-1])
    w_intp[:, :, u['z'].size - 1] = 0

    return w_intp

def get_dimensions(u):

    #This function returns the grid size of the input xarray
    xnodes, ynodes, znodes = u.shape

    return xnodes, ynodes, znodes

def compute_MPI_params(nfiles, nproc, rank):

    #This function returns chunk size handeled by each processor
    chunk = int(nfiles / nproc)      # Amount of tsteps handled by each processor
    remdr = (nfiles % nproc) * int((rank + 1) / nproc)   # Remaining amount of tsteps handled by the last processor incase tsteps is not divisible by nproc

    return chunk, remdr

def seperate_orders():
    first_ord_fields = []
    second_ord_fields = []
    for field in field_list:
        if len(field) == 1:
            first_ord_fields.append(field)
        elif len(field) == 2:
            second_ord_fields.append(field)
        else:
            raise 'Error: All fields in field list must of order 1 or 2'
        
    indv_fields = list(set("".join(field_list)))
    return first_ord_fields, second_ord_fields, indv_fields

def read_vel_fields(indv_fields, filepath, num):
    f = {}
    for field in indv_fields:
        f["%s" % field] = read_cbd('%s/%s-%s.cbd' % (filepath, field, num))
    if "w" in indv_fields:
        f["w"] = intp_w_to_uvp(f["w"], f["u"])
    return f

def initialize_avg_fields(xnodes, ynodes, znodes):
    avg = {}
    for field in field_list:
            avg["%s" % field] = np.zeros((xnodes, ynodes, znodes))
            
    return avg

def compute_firstorder_avg(field, field_avg):
    field_avg += field
    return field_avg

def compute_secondorder_avg(field1, field2, field_avg):
    field_avg += field1 * field2
    return field_avg

def add_to_average(first_ord_fields, second_ord_fields, f, avg):
    
    for field in first_ord_fields:
        avg["%s" % field] = compute_firstorder_avg(f["%s" % field], avg["%s" % field])
    
    for field in second_ord_fields:
        avg["%s" % field] = compute_secondorder_avg(f["%s" % field[0]], f["%s" % field[1]], avg["%s" % field])
        
    return avg

def divide_by_numfiles(chunk, remdr, avg):
    numfiles = chunk + int(remdr)
    for field in field_list:
        avg["%s" % field] /= numfiles
        
    return avg

def empty_recv_data():
    recv = {}
    for field in field_list:
        recv["%s" % field] = None
        
    return recv

def initialize_recv_data(nproc, xnodes, ynodes, znodes, recv):
    for field in field_list:
        recv["%s" % field] = np.empty((nproc, xnodes, ynodes, znodes)) 
        
    return recv

def gather_vars(comm, avg, recv):
    
    for field in field_list:
        comm.Gather(avg["%s" % field].values, recv["%s" % field], root=0) 
    
def compute_average(recv):
    final_avg = {}
    for field in field_list:
        final_avg["%s" % field] = np.mean(recv["%s" % field], axis=0) 
    
    return final_avg

def write_HDF5(destination, sim, avg, u, h5py):

    fp = "%s/%s.h5" % (destination, sim)
    hf = h5py.File(fp, 'w')
    for field in field_list:
        hf.create_dataset('%s' % field, data=avg["%s" % field])
    hf.create_dataset('x-coord', data=u['x'].values)
    hf.create_dataset('y-coord', data=u['y'].values)
    hf.create_dataset('z-coord', data=u['z'].values)
    hf.close()
