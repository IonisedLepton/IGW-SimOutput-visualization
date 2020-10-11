# These two lines may be needed -- it depends on how your python is set up
# (it selects an alternate graphics backend from the default because
#  the default was not working on my computer...)
import matplotlib
# matplotlib.use('TkAgg')

# Load packages
import matplotlib.pyplot as plt
import numpy as np
from igwtools import *
from IPython import embed
from builtins import input      # Python 2/3 compatibility

def interactive():
    """
    First pass at an interactive IGW exploring script
    """

    # Create a dictionary with all the options
    # opts = dict()
    # opts.update({'var': 'bin_U', 'seq': 120, 'subseq': 0})
    # opts.update({'units': ('km','km'), 'clim': 0.03, 'xlim':None, 'zlim':None})
    # opts.update({'colorbarticks': 7, 'cmap':'seismic', 'grid':True})
    # opts.update({'dpi':400, 'figsize':(6.5, 3.5)})

    opts = {
        'var' : 'D',
        'plot_name' : None,
        'units' : ('km', 'km'),
        'clim' : None,
        'xlim' : None,
        'zlim' : None,
         'xlim' : (-50.0, 200),
         'zlim' : (-0.2, 0),
        'colorbarticks' : 7,
        'cmap' : 'jet',
        'grid' : True,
        'dpi' : 400,
        'figsize' : (13.0, 7.0),
        'gif_name' : None,
        'start' : 0,
        'end' : 60,
        'display_time' : 'hh:mm:ss',
        'plot_type' : 'filled contour',
        'contour_minmax' : None,
        'n_contours' : 50
    }

    # Enter a user input loop
    proceed = True
    while proceed:

        print("")
        opts, action = getaction(opts)
        if action == 'quit':

            proceed = False

        elif action == 'embed':

            embed()

        elif action == 'plot':

            print("Updating plot")

            # Get the data
            vs  = igwread('bin_' + opts['var'], opts['end'])     # var at seq
            vss = igwread('bin_' + opts['var'], opts['start'])  # var at subseq

            if opts['var'] == 'D':
                clim1 = max(vss.flatten())
                clim2 = min(vss.flatten())
                opts['clim'] = (clim2, clim1)
                vss = 0 * vss

            data = vs - vss  # difference
            x, z = igwread('bin_vgrid')

            # Produce the plot
            plt.clf()
            pc, cb = plotsnap(data, x, z, **cleanopts(opts))
            plt.show()

        elif action == 'animate':

            print("Generating animation...")
            plt.clf()
            gif(**opts)


def getaction(opts):
    """
    Asks the user for an action
    """
    keys = list(opts.keys())
    keys.sort()
    for i, k in enumerate(keys):

        print("{:2d}.{:>14}: {}".format(i+1, k, opts[k]))

    print("Select a paramater to update, or (q)uit, (p)lot, (e)mbed, (a)nimate:")
    a = input("> ")

    if a is 'q': action="quit"
    elif a is 'p': action="plot"
    elif a is 'e': action="embed"
    elif a is 'a': action="animate"

    elif a.isdigit():

        opts = getnewopt(opts, keys[int(a)-1])
        action = None

    else:

        action = None

    return opts, action


def getnewopt(opts,key):
    """
    Update an option with user input
    """
    try:

        if key in ['var', 'cmap', 'plot_name', 'gif_name']:

            v = input("New value for {} > ".format(key))
            opts[key] = v

        elif key in ['units']:

            v = input("New value for {} (two strings) > ".format(key))
            tmp = v.split()
            opts[key] = tmp[0], tmp[1]

        elif key in ['clim']:

            v = input("New value for {} (one or two floats) > ".format(key))
            tmp = v.split()
            if len(tmp)==1: opts[key] = float(tmp[0])
            else: opts[key] = float(tmp[0]), float(tmp[1])

            # elif len(tmp) == 2: opts[key] = float(tmp[0]), float(tmp[1])
            # elif tmp[0].lower() == "none": opts[key] = None

        elif key in ['xlim', 'zlim', 'figsize', 'contour_minmax']:

            v = input("New value for {} (two floats) > ".format(key))
            tmp = v.split()
            opts[key] = float(tmp[0]), float(tmp[1])

        elif key in ['colorbarticks', 'dpi', 'end', 'start', 'n_contours']:

            v = input("New value for {} (one int) > ".format(key))
            opts[key] = int(v)

        elif key in ['grid']:

            v = input("New value for {} (boolean, T or F) > ".format(key))
            if v.upper()[0] is 'T': opts[key] = True
            if v.upper()[0] is 'F': opts[key] = False

        elif key in ['display_time']:

            v = input('Choose from (f)rames, (h)h:mm:ss, (s)econds or (t)idal period >')
            if v == 'f': opts[key] = 'frames'
            elif v == 'h': opts[key] = 'hh:mm:ss'
            elif v == 's': opts[key] = 'seconds'
            elif v == 't': opts[key] = 'tidal period'

        elif key in ['plot_type']:

            v = input('Choose from (f)illed contour, (c)ontour, (p)colormesh >')
            if v == 'f': opts[key] = 'filled contour'
            elif v == 'c': opts[key] = 'contour'
            elif v == 'p': opts[key] = 'pcolormesh'

    except:

        print("Update failed, try again\n")

    return opts

if __name__ == "__main__":
    interactive()
