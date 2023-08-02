import os
# List of simulations for which you want to compute average
folderpath = "/home/as2204/motosims/scalars"
destination = "/home/as2204/motosims/scalars/averages"
sim_list = os.listdir("%s" % folderpath)
#sim_list = [match for match in sim_list if "hxsx" in match and "_4x" in match]
sim_list = [match for match in sim_list if "longer" in match and "homogen" not in match]

files = os.listdir("%s/%s/output/instantaneous-fields" % (folderpath, sim_list[0]))
num_list = [match[2:-4] for match in files if "v" in match]
num_list = ["{:05d}".format(int(num)) for num in num_list]
num_list.sort()
num_list = ["{:03d}".format(int(num)) for num in num_list]
#num_list = num_list[1000:]

# Variables for which you want to compute average
field_list = ["u", "v", "w", "uu", "uv", "uw", "vv", "vw", "ww"]

nfiles = len(num_list)
#print(sim_list)
