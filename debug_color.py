"""
Script for debugging the colorbar problem with u animations
"""
import igwtools as igwt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import imageio
import glob
import os
import json
import cmocean

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

def plotsnap(x,z,data,clim,plot_type,fig=None,ax=None, **kwargs):

    cbar_norm = kwargs.pop('cbar_norm')
    n_contours = kwargs.pop('n_contours', 100)
    cmap = kwargs.pop('cmap','RdBu')

    if cbar_norm['norm_type'] == 'SymLogNorm':
        linthresh = cbar_norm['linthresh']
        linscale = cbar_norm['linscale']
        base = cbar_norm['base']
        cbar_norm = colors.SymLogNorm(linthresh = linthresh,
                                    linscale = linscale,
                                    vmin = clim[0],
                                    vmax = clim[1],
                                    base=base)
    if not fig or not ax:
        fig,ax = plt.subplots(figsize=(13,7))

    xlim = (-50,200)
    zlim = (-0.2,0)
    # normalize the colorbar
    if cbar_norm is None:
        cbar_norm = colors.SymLogNorm(linthresh=0.05, linscale=1,
                                    vmin=clim[0], vmax=clim[1], base=10)
    
    ax.fill_between(x[0,:],np.min(z),z[0,:],
                    facecolor='#65391a', linewidth=0.25)

    levels = np.linspace(clim[0],clim[1],n_contours)

    if plot_type == 'filled contour':
        cs = ax.contourf(x,z,data,levels=levels,
                        cmap = cmap, norm=cbar_norm)
        cs2 = ax.contour(cs,levels=levels,cmap=cmap)
    elif plot_type == 'contour':
        cs = ax.contour(x,z,data,levels=levels,cmap = cmap)
    else:
        raise ValueError ("Unsupported Plot Type")

    cbar = fig.colorbar(cs, ax=ax, extend='both')

    if plot_type=='filled contour':
        cbar.add_lines(cs2)

    return fig,ax

def subtract_vertical_avg(data,var='u'):
    
    x,z = igwt.igwread('bin_vgrid')
    # subtract vertical average from u
    Q = np.trapz(data[:,0],z[:,0])

    # TODO: not incredibly efficient - optimize the code below
    for i in range(np.shape(x)[1]):
        u_avg = Q/(-np.min(z[:,i]))
        data[:,i] = data[:,i] - u_avg
    
    return data

    
def single_plot(iframe):
    var = 'U'
    varfile = 'bin_U'
    clim = kclim()
    x,z = igwt.igwread('bin_vgrid')
    plot_type = 'filled contour'
    L, H = igwt.read_startup()

    print('clim: ',clim)

    data = igwt.t0(var,start=iframe)
    #  data = subtract_vertical_avg(data,var)

    fname  = 'plot_U_contourf_'+iframe+'_ncontours25_contouradded.png'
    fig,ax = plotsnap(x,z,data,clim,plot_type)
    fig.savefig(fname)

    plt.close()



def frame_gen(**kwargs):
    frames = []
    time,dt = igwt.times()
    var = 'U'
    varfile = 'bin_U'
    start = kwargs.pop('start_frame',0)
    end = kwargs.pop('end_frame',50)
    clim = kclim()
    x,z = igwt.igwread('bin_vgrid')
    plot_type = kwargs.pop('plot_type', 'filled_contour')
    L, H = igwt.read_startup()

    print('clim: ',clim)

    for iframe in range(start,end):

        data = igwt.t0(var,start=iframe)
        data = subtract_vertical_avg(data,var)


        fname  = 'frame'+str(iframe)+'.png'
        fig,ax = plotsnap(x,z,data,clim,plot_type, **kwargs)
        fig.savefig(fname)

        frames.append(imageio.imread(fname))
        plt.close()

    return np.array(frames)
                    
def gif(**kwargs):
    dirname = kwargs.pop('dirname', None)
    gif_name = kwargs.pop('gif_name', 'noname.gif')

    if dirname is not None:
        gif_name = os.path.join(dirname,gif_name)

    frames = frame_gen(**kwargs)
    imageio.mimsave(gif_name,frames)

    for file in glob.glob(os.path.join(dirname,'frame*')):
        os.remove(file)


def bgVals(i,j):
    vals = []
    for iframe in range(80):
        data = igwt.t0(var='U',start=iframe)
        vals.append(data[i,j])
    return min(vals),max(vals)

def bgValsSub(i,j):
    vals = []
    for iframe in range(80):
        data = igwt.t0(var='U',start=iframe)
        data = subtract_vertical_avg(data,'U')
        vals.append(data[i,j])

    return min(vals),max(vals)

def logDict(dict_to_log,fpath):
    # TODO: Modify to avoid rewriting same files
    with open(fpath,'w') as file:
        file.write(json.dumps(param_set))


# gif()
# single_plot(iframe=25)

test_params = [
            {
                'dirname':'anims_U_contourf_dc',
                'gif_name':'cmap_RdBu__cbar_linthresh0dot05_linscale1.gif',
                'start_frame' : 0,
                'end_frame' : 70,
                'plot_type' : 'filled contour',
                'n_contours' : 100,
                'cmap' : 'RdBu',
                'cbar_norm':
                            {
                                'norm_type' : 'SymLogNorm',
                                'linthresh' : 0.05,
                                'linscale' : 1,
                                'base' : 10
                            }

            }
        ]


for param_set in test_params:
    gif(**param_set)
    logging_fpath = os.path.join(param_set['dirname'],'README.json')
    logDict(param_set,logging_fpath)



