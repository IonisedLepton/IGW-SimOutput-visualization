import igwtools as igwt
import numpy as np

L,H = igwt.read_startup()
x,z = igwt.igwread('bin_vgrid')
init_data = igwt.t0(var='U',start=0)

clim_min = min(init_data.flatten())
clim_max = max(init_data.flatten())


lo_list = [clim_min]
hi_list = [clim_max]

for frame in range(1,96):
    data = igwt.t0(var='U',start=frame)

    lo = min(data.flatten())
    hi = max(data.flatten())
    print('frame: ',frame)
    if lo < min(lo_list):
        lo_list.append(lo)
    if hi > max(hi_list):
        hi_list.append(hi)

print('min U is: ',min(lo_list))
print('max U is: ',max(hi_list))


