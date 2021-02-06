"""
Script for debugging the colorbar problem with u animations
"""
import igwtools as igwt
import numpy as np
import matplotlib.pyplot as plt
import imageio
import glob
import os

def kclim():

    L,H = igwt.read_startup()
    x,z = igwt.igwread('bin_vgrid')
    init_data = igwt.t0(var='U',start=0)

    clim_min = min(init_data.flatten())
    clim_max = max(init_data.flatten())


    lo_list = [clim_min]
    hi_list = [clim_max]

    for frame in range(1,50):
        data = igwt.t0(var='U',start=frame)

        lo = min(data.flatten())
        hi = max(data.flatten())
        if lo < min(lo_list):
            lo_list.append(lo)
        if hi > max(hi_list):
            hi_list.append(hi)

    clim = (min(lo_list),max(hi_list))

    return clim

def plotsnap(x,z,data,clim,plot_type,fig=None,ax=None):

    if not fig or not ax:
        fig,ax = plt.subplots(figsize=(13,7))

    xlim = (-50,200)
    zlim = (-0.2,0)
    cmap = 'jet'
    n_contours = 100
    
    ax.fill_between(x[0,:],np.min(z),z[0,:],facecolor='#65391a',linewidth=0.25)

    levels = np.linspace(clim[0],clim[1],n_contours)

    if plot_type == 'filled contour':
        cs = ax.contourf(x,z,data,levels=levels,cmap = cmap)
    elif plot_type == 'contour':
        cs = ax.contour(x,z,data,levels=levels,cmap = cmap)

    cbar = fig.colorbar(cs)

    return fig,ax

def frame_gen():
    frames = []
    time,dt = igwt.times()
    var = 'U'
    varfile = 'bin_U'
    start = 0
    end = 50
    clim = kclim()
    x,z = igwt.igwread('bin_vgrid')
    plot_type = 'contour'

    print('clim: ',clim)

    for iframe in range(0,50):

        data = igwt.t0(var='U',start=iframe)
        fname  = 'frame'+str(iframe)+'.png'
        fig,ax = plotsnap(x,z,data,clim,plot_type)
        fig.savefig(fname)

        frames.append(imageio.imread(fname))
        plt.close()

    return np.array(frames)
                    
def gif():
    gif_name = 'anim_U_contour_dc.gif'
    frames = frame_gen()
    imageio.mimsave(gif_name,frames)

    for file in glob.glob('frame*'):
        os.remove(file)



gif()
# x,z = igwt.igwread('bin_vgrid')
# data = igwt.t0(var='U',start=10)
