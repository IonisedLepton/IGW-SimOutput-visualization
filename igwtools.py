"""
    Module: igwtools

This module contains functions that can process and produce animations or plots of the binary outputs of the IGW Simulation

Functions:
	igwread() : Reads binaries and returns np array
	igwwrite() : Reads data and saves it in a format that igwread() can process
	plotsnap() : Generalized plotting function for a single binary output frame 
    slidingmean() : Computes and returns the sliding mean of a 1D array
    slidingmean2() : Computes and returns the sliding mean of a 2D array
    fcount() : counts and returns the number of binary frames of an output variable
    t0() : read in the first binary file of an output variable and returns np array of the data
    read_startup() : read the startup_graphics file and return L,H
    frame_gen() : Generate plots of all the binary frames of an output variable
    gif() : Produce gif of the plots of an output variable
    times() : read the binary file containing time information and return np arrays 
    cleanopts() : Helper function. Reformat a dictionary containing information about the plots to produce 

This module is mainly used in:
    interactive.py
"""

import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import inspect
import imageio


def igwread(filename, seq=None):
    """
    Loads IGW binary output. The file "startup_graphics" that is written
    by IGW should be present with the output files.

    To load a specific file, use the one parameter form:
          D = igwread('bin_D5')
       Jinv = igwread('bin_inv_Jac')
        x,z = igwread('bin_vgrid')
      px,pz = igwread('bin_pressure9')
    sgpbotx = igwread('bin_sgpbotx')

    You can also let the function append the snapshot number, such as inside
    of a loop, by using the two parameter form:
          D = igwread('bin_D',5)
      px,pz = igwread('bin_pressure',9)


      Parameters:
        filename: str
            binary file name, such as 'U35' or just the output variable such as, 'U'.
        
        seq: int
            Frame number of the binary file to be read such as 35. Only required if output variable passed to filename

     Returns:
        numpy.array, tuple of numpy.array
    """

    # Filename
    path = ""
    if seq is None: fname = filename
    else: fname = filename + str(seq)
    if not os.path.isfile(fname):

        for root, dirs, files in os.walk(os.getcwd()):
            if fname in files:
                fname = os.path.join(root, fname)
                path =  root + "/"
                break

    # Get I, J
    with open(path + 'startup_graphics') as f:
        line = f.readline().split()
        I, J = int(line[0]), int(line[1])

    # Load binary data
    data = np.fromfile(fname, dtype=np.float64, count=-1)
    n = len(data)

    # Reshape and return data indexed as z,x
    if n == I+1: # bin_sgpbotx, bin_sgpbotz
        return data.reshape(I+1, 1).swapaxes(0, 1)

    if n == I*(J+2): # bin_{D,U,V,W}, bin_inv_Jac, Uback_vs_z, ...
        return data.reshape(I, J+2).swapaxes(0, 1)

    if n == 2*I*(J+2): # bin_pressure, bin_vgrid
        data1 = data[0:n//2].reshape(I, J+2).swapaxes(0, 1)
        data2 = data[n//2: ].reshape(I, J+2).swapaxes(0, 1)
        return data1, data2

def igwwrite(fname,data):
    """
    Writes out an array in the same format as igw
    such that we can load it later with igwread()

    Parameters:
        fname: str
        Save name of file

        data: np.array
        data to be saved

    """
    with open(fname,'wb') as f:
        for ii in range(data.shape[1]):
            f.write(data[:,ii].copy())

def plotsnap(data, x, z, plot_name=None, units=('km','km'), clim=None, xlim=None, zlim=None, colorbarticks=7,
             cmap='seismic', grid=True, dpi=400, figsize=(6,4), plot_type='contour', contour_minmax=None, n_contours=None):
    """
    Generalized plotting function for IGW snapshots
    It creates a pseudocolour plot of data
    units for (x,z) can be 'm' or 'km', we don't handle any other values
    xlim and zlim are consistent with units specified

    Main Parameters:
        data: np.array
        array containing the data to be plotted

        x: 2D numpy.array
        array containing x grid coordinates

        z: 2D numpy.array
        array containing z grid coordinates

    Return:
        pc,cb
        matplotlib handles for further tweaking of the plots
    """
    # Check units
    if units[0] == 'km': x /= 1000
    if units[1] == 'km': z /= 1000

    # Shade in topography
    plt.fill_between(x[0, :], np.min(z), z[0, :], facecolor='#65391a', linewidth=0.25)
    plt.grid(grid)

    if plot_type == 'pcolormesh':

        pc = plt.pcolormesh(x, z, data, cmap=cmap)
        cb = plt.colorbar(format='%5.3f')
        pc.set_rasterized(True)
        cb.solids.set_edgecolor("face")
        cb.solids.set_rasterized(True)

    else:

        if n_contours is None: N = 50
        else: N = n_contours
        if contour_minmax is not None:

            cmin = min(contour_minmax)
            cmax = max(contour_minmax)
            N = np.linspace(cmin, cmax, N)

        if plot_type == 'contour':

            pc = plt.contour(x, z, data, N, cmap=cmap)
            cb = plt.colorbar(format='%5.3f')

        elif plot_type == 'filled contour':

            pc = plt.contourf(x, z, data, N, cmap=cmap)
            cb = plt.colorbar(format='%5.3f')

    # Color scale limits
    if clim is not None:
        if type(clim) is not tuple:   # One value means centered about zero
            CL, CH = -clim, clim
        if type(clim) is tuple:
            CL, CH = clim[0], clim[1]
        pc.set_clim(CL,CH)
        cb.set_clim(CL,CH)
        cb.set_ticks(np.linspace(CL,CH, colorbarticks))

    # Axis limits
    if xlim is not None: plt.gca().set_xlim([xlim[0],xlim[1]])
    if zlim is not None: plt.gca().set_ylim([zlim[0],zlim[1]])

    # Axis labels
    plt.xlabel(r'$x$ ({})'.format(units[0]))
    plt.ylabel(r'$z$ ({})'.format(units[1]))

    # Figure size and output
    plt.gcf().set_size_inches(figsize[0], figsize[1]) # set figure size here
    plt.tight_layout(h_pad=0.0, pad=2.0)

    if plot_name is not None:
        plt.savefig(plot_name, format='png', dpi=dpi)

    # Return the handles for further tweaking outside of this function
    return pc, cb

def slidingmean(t,d,w):
    """
    Sliding mean, computed as:  sm(t) = (1/w) * \int_{t-w}^{t} d(t') dt'
    - t and d should be length N vectors, w is the averaging window width


    Parameters:
        t: array
        axis array

        d: array
        data array

        w: float
        weight

    Returns:
        slidingmean
    """
    from slidingmean_c import slidingmean_c as smean
    tt, dd = t.copy(), d.copy()  # Interfacing with C routines requires care...
    ww = np.array([float(w)])
    return smean(tt,dd,ww)

def slidingmean2(t,d,w):
    """
    Sliding mean for a 2d array where we work along axis 1
    That is, input d has dimensions MxN
             input t has dimensions N
             output  has dimensions MxN

    Parameters:
        t: array
        axis array

        d: array
        data array

        w: float
        weight

    Returns:
        slidingmean
    """
    from slidingmean_c import slidingmean_c as smean
    tt, dd = t.copy(), d.copy()  # Interfacing with C routines requires care...
    ww = np.array([float(w)])
    out = np.copy(d)
    for j in range(d.shape[0]):
        out[j,:] = smean(tt,dd[j,:],ww)
    return out

def fcount(var):
    """
    Function that counts all of the binary data files associated
            with the input variable in the current working directory.

    Parameters:
        var: str
        output variable name, such as 'U'

    Returns:
    count of the number of files in the directory
    """
    path = ""
    if not os.path.isfile("bin_vgrid"):

        for root, dirs, files in os.walk(os.getcwd()):
            if "bin_vgrid" in files:
                path = root + "/"
                break

    Nfiles = 0
    for f in glob.glob(path + 'bin_' + var + '*'):
        Nfiles += 1

    return Nfiles

def t0(var, start=None):
    """
    Reads the first binary file of an output variable

    Parameters:
    var: str
        The output variable whose binary is to be read, such as 'U'
    start: int, Optional
        If a different frame is to be read, then the index of that frame

    Returns:
        np.array
        The inputted binary data of the file

    Main Dependencies:
    igwread()
    """
    if start is None: start = 0
    else: start = start

    vfile = 'bin_' + var
    data = igwread(vfile, start)

    return data

def read_startup():
    """
    Read the startup_graphics file

    Returns
        L,H
    """
    path = ""
    if not os.path.isfile("startup_graphics"):

        for root, dirs, files in os.walk(os.getcwd()):
            if "startup_graphics" in files:
                path = root + "/"
                break

    ##~~ Read startup_graphics to get height & length of domain~~
    with open(path + 'startup_graphics') as f:

        lines = f.readlines()
        L = float(lines[1].split()[0])
        H = float(lines[1].split()[1])

    return  L, H

def frame_gen(**kwargs):
    """
    Generate an array of snapshots of the different frames(binaries) of an output variable. Helper function

    Output Variables currently supported:
    U
    V
    W
    pressure (for pressure, the user needs to input either 'pressurex' or 'pressurez' in the kwargs dict to specify x or z gradient)
    D
    Used in:
        gif()

    Main Dependencies:
        plotsnap()
        igwread()
        t0()
        read_startup()
        times()

    Parameters:
        kwargs: dict
            dictionary of specifications of the plots to be produced

    Returns:
        np.array
            array of objects returned by imageio.imread(). Each element in the array corresponding to one frame
    """
    frames = []
    time, dt = times()

    var = kwargs['var']
    
    # bin_pressure contains both x and z gradient data.
    if 'pressure' in var:
        if var[-1] is 'x':
            axis = 0 
        elif var[-1] is 'z':
            axis =1
        var=var[:-1]

    varfile = 'bin_' + var
    start = kwargs['start']
    end = kwargs['end']
    t_units = kwargs['display_time']
    plot_type = kwargs['plot_type']
    N = kwargs['n_contours']
    cmm = kwargs['contour_minmax']

    if start is None: start = 0
    if end is None: end = fcount(var)
    elif end > fcount(var): end = fcount(var)

    nframes = end - start
    init_data = t0(var, start=start)

    data = np.copy(init_data)
    x, z = igwread('bin_vgrid')
    if var == 'D':

        clim1 = max(init_data.flatten())
        clim2 = min(init_data.flatten())
        kwargs['clim'] = (clim2, clim1)
        init_data = 0 * init_data

    elif var == 'U':

        # L, H = read_startup()
        # Q = np.trapz(data[:, 0], z[:, 0])


        # for i in range(np.shape(x)[1]):

        #     u_avg = Q/(-np.min(z[:, i]))
        #     data[:,i] = data[:,i] - u_avg
        pass


    elif var == 'V':
        
        L, H = read_startup()
        Q = np.trapz(data[:, 0], z[:, 0])
	
	
        for i in range(np.shape(x)[1]):

            v_avg = Q/(-np.min(z[:, i]))
            data[:,i] = data[:,i] - v_avg
    
    elif var =='pressure':

        data = init_data[axis]
        clim1 = max(data.flatten())
        clim2 = min(data.flatten())
        kwargs['clim'] = (clim2,clim1)
        init_data = 0*data
    
    plotargs = cleanopts(kwargs)
    # pc, cb = plotsnap(data - init_data, x, z, **plotargs)
    pc, cb = plotsnap(data, x, z, **plotargs)

    if t_units == "frames": plt.title(var + ' at T = 0000')
    elif t_units == "hh:mm:ss": plt.title(var + ' at T = 00:00:00')
    elif t_units == 's': plt.title(var + ' at T = 0000 s')
    elif t_units == "tidal period": plt.title(var + ' at T = 0000')

    plt.savefig('frame0.png')
    print('frame0.png')
    frames.append(imageio.imread('frame0.png'))
    for i in range(start + 1, nframes):
       
        if var != 'pressure':
            # data = igwread(varfile, i) - init_data
            data = igwread(varfile, i)
        else:
            data = igwread(varfile,i)[axis] - init_data[axis]

        fname = 'frame' + str(i) + '.png'

        if plot_type == "pcolormesh":

            pc.set_array(data[:-1, :-1].ravel())

        else:

            if N is None: N = 50
            if cmm is not None:
                cmin = min(cmm)
                cmax = max(cmm)
                N = np.linspace(cmin, cmax, N)

            if plot_type == "filled contour":

                ax = plt.gca()
                ax.collections.pop()
                pc = plt.contourf(x, z, data, N, cmap='jet')

            elif plot_type == "contour":

                ax = plt.gca()
                ax.collections = []
                # pc, cb = plotsnap(data, x, z, **plotargs)
                pc = plt.contour(x, z, data, N, cmap='jet')

            N = kwargs['n_contours']

        if t_units == "frames":

            plt.title(var + ' at T = {:04d}'.format(i))

        elif t_units == "hh:mm:ss":

            t = time[i]
            d = int(t // 86400)
            h = int(t % 86400 // 3600)
            m = int(t % 3600 // 60)
            s = int(t % 60)
            if d > 0: plt.title(var + ' at T = {:02d}:{:02d}:{:02d}:{:02d}'.format(d, h, m, s))
            else: plt.title(var + ' at T = {:02d}:{:02d}:{:02d}'.format(h, m, s))

        elif t_units == "seconds":

            l = len(str(time[-1]))
            plt.title(var + ' at T = {:0{width}.2f} s'.format(time[i], width=l))

        elif t_units == "tidal period":

            l = len(str(time[-1] / 44712.0))
            plt.title(var + ' at T = {:0{width}.2f} s'.format(time[i]/ 44712.0, width=l))

        plt.savefig(fname)
        print(fname)
        frames.append(imageio.imread(fname))

    return np.array(frames)

def gif(**kwargs):
    """
    Create gif of snapshots of an output variable.Saves the gif in the directory of this module. Helper function
    Used in:
        interactive.interactive()

    Main Dependencies:
        frame_gen()

    Parameters:
        kwargs: dict
            dictionary of specifications of the plots to be produced
    
    """
    gif_name = kwargs['gif_name']
    if gif_name is None:
        gif_name = 'IGW_output.gif'

    frames = frame_gen(**kwargs)
    imageio.mimsave(gif_name,frames)

    ##~~Deletes the .png files that were saved since we don't need them anymore~~
    for file in glob.glob('frame*.png'):
        os.remove(file)

def times():
    """
    Read the binaries containing the time information for each frame

    Returns:
        t,dt
        np.arrays containing t,dt information
    """

    path = ""
    if not os.path.isfile("time0"):

        for root, dirs, files in os.walk(os.getcwd()):
            if "time0" in files:
                path = root + "/"
                break

    t = []
    dt = []
    for file in glob.glob(path + 'time' + '*'):

        f = open(file, 'r')
        line = f.readline()
        data = line.split()
        t.append(float(data[0]))
        dt.append(float(data[1]))
        f.close()

    t = np.array(t)
    dt = np.array(dt)
    i = np.argsort(t)
    t = t[i]
    dt = dt[i]

    return t, dt

def cleanopts(opts):
    """
    Generates a copy of dict opts that only has parameters that plotsnap() accepts. Helper function
    """
    opts2 = opts.copy()
    delkeys = []

    argspec = inspect.signature(plotsnap)
    for key in opts2.keys():

        if key not in argspec.parameters.keys():

            delkeys.append(key)

    for key in delkeys:

        del opts2[key]

    return opts2
