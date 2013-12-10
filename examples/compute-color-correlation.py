#! /opt/local/bin/python
# encoding: utf-8
"""
colormap.py

Created by Sebastian Werner on 2012-10-09.

"""

import sys
import os
import numpy as np
from numpy.ma.core import exp
import Image
import scipy as sp
import scipy.ndimage
from scipy.constants.constants import pi
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import pylab as pl
import argparse
from scipy.signal import argrelextrema
import ImageFilter
import operator
from matplotlib.backends.backend_pdf import PdfPages
import datetime
import re
from qmaptools import *


############### Settings 
mpl.rcParams["font.size"] = '10'
mpl.rcParams['axes.titlesize'] = 'medium'
#mpl.rcParams['text.usetex'] = True

# create some colormaps from black to color.
cdict = {'red':   ((0.0,  0.0, 0.0),(1.0,  1.0, 1.0)),
         'green': ((0.0,  0.0, 0.0),(1.0,  0.0, 0.0)),
         'blue':  ((0.0,  0.0, 0.0),(1.0,  0.0, 0.0))}
red_cmap = mpl.colors.LinearSegmentedColormap('red_cmap',cdict,256)
cdict = {'green':   ((0.0,  0.0, 0.0),(1.0,  1.0, 1.0)),
         'blue': ((0.0,  0.0, 0.0),(1.0,  0.0, 0.0)),
         'red':  ((0.0,  0.0, 0.0),(1.0,  0.0, 0.0))}
green_cmap = mpl.colors.LinearSegmentedColormap('green_cmap',cdict,256)
cdict = {'blue':   ((0.0,  0.0, 0.0),(1.0,  1.0, 1.0)),
         'green': ((0.0,  0.0, 0.0),(1.0,  0.0, 0.0)),
         'red':  ((0.0,  0.0, 0.0),(1.0,  0.0, 0.0))}         
blue_cmap = mpl.colors.LinearSegmentedColormap('blue_cmap',cdict,256)


# General settings
from warnings import simplefilter,filterwarnings
simplefilter("ignore", np.ComplexWarning)
filterwarnings("ignore")
mpl.rcParams['figure.dpi'] = 300


############### Argument Parser
parser = argparse.ArgumentParser(description='Compute correlation coefficient for a RGB image.')
parser.add_argument('infile', metavar='IMAGE.ext', nargs='+', type=argparse.FileType('r'), help='(.png|bmp|.gif|.jpg|.tif) file with RGB intensity.')
parser.add_argument('--verbose', '-v', help="increase number of debug messages",action='count')
parser.add_argument('--basename','-b',type=str,help="Filename of OUTFILEs. Required for multiple files to process.")
parser.add_argument('--title',type=str,help="Main title string.")
parser.add_argument('--radius',type=int, help="which radius for filtering (0 means - do not filter)",default=0)
parser.add_argument('--resize',type=int, help="Resize incoming raw data to new size (nearest)",default=0)
parser.add_argument('--pixelsize',type=float,help="What is the pixelsize in nm of our map?")
parser.add_argument('--red',type=str,help="Element corresponding to red?",default="$Mn$")
parser.add_argument('--green',type=str,help="Element corresponding to green?",default="$Co$")
parser.add_argument('--blue',type=str,help="Element corresponding to blue?",default="$Si$")


# principal component analysis
parser_pca = parser.add_argument_group("statistical analysis options")
parser_pca.add_argument('--descriptors',action='store_true', help="Determine co-location descriptors (2d CC, SSIM)")
parser_pca.add_argument('--scatter',action='store_true', help="Perform scatter analysis")
parser_pca.add_argument('--heatmap',action='store_true', help="Perform heat-map analysis")
parser_pca.add_argument('--pca',action='store_true', help="Perform primary component analysis")
parser_pca.add_argument('--powerspectrum',action='store_true', help="Perform power spectrum/fft analysis")
parser_pca.add_argument('--histogram',action='store_true', help="Include a histogram")
parser_pca.add_argument('--cutoff',type=int, help="Cutoff value for noise (0..255)",default=70)
parser_pca.add_argument('--colocation',action='store_true', help="Perform co-location analysis for pixels with value greater than cutoff")
parser_pca.add_argument('--numbins',type=int, help="set dimensions for heatmap",default=25)

# image options
parser_overview = parser.add_argument_group("overview options")
parser_overview.add_argument('--haadf',action='store_true', help="include the STEM HAADF image.")
parser_overview.add_argument('--haadfext',type=str, help="extension from the basename supplied: IMAGEpostfix.(jpg|png)", default="-HAADF")
parser_overview.add_argument('--map',action='store_true', help="include the EDS/EELS map.")
parser_overview.add_argument('--mapext',type=str, help="extension from the basename supplied: IMAGEpostfix.(jpg|png)", default="-MAP")
parser_overview.add_argument('--composite',choices=['rg','rb','gb',False], default=False, help="Which channel composite shall be shown (False for no composite)")
parser_overview.add_argument('--threshold',choices=['rg','rb','gb',False], default=False, help="Which a composite after applying the noise-cutoff (False for none)")

#parser_overview.add_argument('--eftem',action='store_true', help="include an EF-TEM image (read from FILEe.(jpg|png) )")
#parser_overview.add_argument('--map-extension',type=str, help="extension from the basename supplied: IMAGEpostfix.(jpg|png)", default="-EFTEM")

# spectroscopy options
parser_spec = parser.add_argument_group("spectroscopy options")
parser_spec.add_argument('--eels',action='store_true', help="include the integral EELS plot (read from FILE.txt)")
parser_spec.add_argument('--eds',action='store_true', help="include an EDS plot (read from FILE.txt)")
parser_spec.add_argument('--smooth',type=int, help="smoothing width for EELS/EEDS", default=35)
parser_spec.add_argument('--order',type=int, help="peak search filter parameter", default=30)


args = parser.parse_args()

# we got multiple files
if len(args.infile) > 1:
    mode = 'multi'
    if not args.basename:
        raise parser.error("When supplying multiples files for process, a basename using '-b NAME' must be specified.")
    else:
        basename = args.basename
else:
    mode = "single"
    basename = os.path.splitext(args.infile[0].name)[0]

if args.title:
    maintitle = args.title
else:
    maintitle = ""

if (not args.haadf) and (not args.map) and (not args.composite) and (not args.threshold):
    raise parser.error("Either a HAADF, map, thresholdmap or composite must be included must be specified.")


### prepare conditionition matrix
numbins = args.numbins
# now eliminate some part from the heatmap
conditioning_matrix = np.ones((numbins,numbins))
a = 0
b = 0
r = 2
y,x = np.ogrid[-a:numbins-a, -b:numbins-b]
mask = x*x + y*y <= r*r
conditioning_matrix[mask] = 0.0
mask1 = x*x + y*y <= (r+1)**2
conditioning_matrix[mask ^ mask1] = 0.1
mask2 = x*x + y*y <= (r+2)**2
conditioning_matrix[mask1 ^ mask2] = 0.4
mask3 = x*x + y*y <= (r+3)**2
conditioning_matrix[mask2 ^ mask3] = 0.6


# create results file
logfile = open(basename+"-analysis.csv","w")
logfile.write("file,RG cc,RG 2d cc,RG SSIM,RB cc,RB 2d cc,RB SSIM,GB cc,GB 2d cc,GB SSIM")
if args.pca:
    logfile.write(",PCA center x, PCA center y,rotation,gamma 1,sigma 1,gamma 2,sigma 2\n")
else:
    logfile.write("\n")

all_r = np.array([])
all_g = np.array([])

############### MAIN
# main loop to cycle over the files
for file in args.infile:
    
    ############### Prep data structures, plots & pdf
    # we come in with a basename like: "13-08-21_#48_CoMn0.01_1"
    file_basename = os.path.splitext(file.name)[0]
    
    if (file_basename[-1] == "n") or (file_basename[-1] == "r") or (file_basename[-1] == "c"):
        file_basename = file_basename[:-1]
    
    if args.radius > 0:
        file_basename += "_r=%i" % (args.radius)
    if maintitle != '':
        file_title = "%s (%s)" % (maintitle, file_basename)
    else:
        file_title = file_basename
    text = []
    pca = []
    
    # set up pdf
    pdf = PdfPages(os.path.splitext(file.name)[0]+"-analysis.pdf")
    d = pdf.infodict()
    d['Title'] = file_title
    d['Author'] = "Sebastian Werner"
    d['Subject'] = 'Data analysis for EELS/EDS maps'
    d['CreationDate'] = datetime.datetime.today()
    
    # set up the pages
    f = plt.figure("regular")
    
    # now styled text
    #file_title = "%s (\\textit{%s})" % (maintitle, file_basename)

    f.text(0.5,0.98, file_title, transform=f.transFigure, ha='center', va='center', style='italic', size=14)

    #if args.radius > 0:
    #    f2 = plt.figure("filtered")
    #    f2.text(0.5,0.98,file_title+" (filtered)", transform=f.transFigure, ha='center', va='center', style='italic', size=14)
       
    
    ############### Open image
    # read image into PIL
    img = Image.open(file).convert('RGB')
    print "Importing %s (%i x %i)" % (file, img.size[0], img.size[1])
    
    # filter if needed
    if args.radius > 0:
        print "Analysis of a blurred image with r = %i" % (args.radius)
        img = img.filter(MyGaussianBlur(radius=args.radius))
        logfile.write("%s," %(file_basename))        
    else:
        print "Analyzing raw image"
        logfile.write("Blurred %s r=%i," %(file_basename,args.radius))
        
    # import channels by color
    arr = PIL2array(img)
    r = importchan(arr,0)
    g = importchan(arr,1)
    b = importchan(arr,2)

    # check if channels actually have data...
    Has = {"r" : False, "g" : False, "b": False}
    channels = ['r','g','b']

    if r.max() > 25:
        Has["r"] = True
    if g.max() > 25:
        Has["g"] = True
    if b.max() > 40:
        Has["b"] = True  
    
    ############### ANALYSIS WORKER ROUTINE  
    
    # output PDF is a grid of 3x2
    # 1st row:
    #   right: RGB input
    #   middle: HAADF, composite or threshold image
    #   left:   heatmap or parity plot
    # 2nd row:
    #   in case "eels" or "eds" are selected: 
    #     left: histogram or powerspectrum or pca or correlation plot
    #     middle + right: spectrum
    #   otherwise:
    #     right: Histogram of input RGB
    #     middle: powerspectrum or pca or empty
    #     left:   correlation plot (Greg's) or empty
    
    
    #### 1st row: left
    subplot_1st_left = plt.subplot2grid((2,3),(0,0),aspect="equal")
    # place RGB input
    if args.map:
         draw_image(subplot_1st_left,file_basename,which="map")
    else:
        subplot_1st_left.imshow(img,interpolation="nearest")
    subplot_1st_left.set_axis_off()
    subplot_1st_left.set_title("input")
        
    # Run simple correlations on the map
    # save results and place text
    result = do_correlation(r,g,b,logfile=logfile)
    f.text(0.19, 0.56, result, size=10, ha='center', va='center')
    
    
    #### 1st row: middle
    subplot_1st_center =plt.subplot2grid((2,3),(0,1),aspect="equal")
    
    if args.haadf:
        draw_image(subplot_1st_center,file_basename,which="haadf")
        
    elif args.composite:
        if args.composite == "rg":
            temp = np.dstack((r,g,np.zeros((img.size[0],img.size[1]),dtype=np.int8))).astype(np.uint8)
            subplot_1st_center.set_title(args.red+" + "+args.green)
        elif args.composite == "rb":
            temp = np.dstack((r,np.zeros((img.size[0],img.size[1]),dtype=np.int8),b)).astype(np.uint8)
            subplot_1st_center.set_title(args.red+" + "+args.blue)
        elif args.composite == "gb":
            temp = np.dstack((np.zeros((img.size[0],img.size[1]),dtype=np.int8),g,b)).astype(np.uint8)
            subplot_1st_center.set_title(args.green+" + "+args.blue)

        subplot_1st_center.imshow(temp,interpolation="nearest")
        subplot_1st_center.set_axis_off()
            
    elif args.threshold:
        draw_threshold(subplot_1st_center, r, g, b, threshold_r=args.cutoff, threshold_g=args.cutoff) 
    
    #### 1st row: right
    subplot_1st_left = plt.subplot2grid((2,3),(0,2),aspect="equal")
    
    if args.heatmap:
        draw_heatmap(subplot_1st_left, r, g, label_x=args.red, label_y=args.green, numbins = args.numbins, conditioning_matrix = conditioning_matrix)
    elif args.scatter:
        draw_scatterplot(subplot_1st_left, r, g, label_x=args.red, label_y=args.green ,color_dots=np.dstack((r,g,b)).reshape((-1,3)))
    
    
    #### 2nd row: 
    subplot_2nd_left = plt.subplot2grid((2,3),(1,0),aspect="equal")
        
    if (args.eels or args.eds):
        
        #### left
        if args.colocation:
            draw_colocation(subplot_2nd_left,img)
        elif args.pca:
            draw_pca(subplot_2nd_left,r,g)
        elif args.powerspectrum:
            draw_powerspectrum(subplot_2nd_left,r,g,b)
        else:
            draw_histogram(subplot_2nd_left,r,g,b)
        
        #### center + right
        subplot_spectroscopy = plt.subplot2grid((2,3),(0,1),colspan=2)
        if args.eels:
            draw_eels(subplot_spectroscopy,file_basename)
        elif args.eds:
            draw_eds(subplot_spectroscopy,file_basename)
        
    else:
        #### left
        #if args.histogram:
        draw_histogram(subplot_2nd_left,r,g,b)

        #### center
        subplot_2nd_center = plt.subplot2grid((2,3),(1,1),aspect="equal")
        if args.powerspectrum:
            draw_powerspectrum(subplot_2nd_center,r,g,b) #,cutoff=args.cutoff)
        elif args.pca:
            draw_pca(subplot_2nd_center,r,g)
        
        #### right
        subplot_2nd_right = plt.subplot2grid((2,3),(1,2),aspect="equal")
        if args.colocation:
            draw_colocation(subplot_2nd_right,img,cutoff=args.cutoff)

    logfile.write('\n')
    
    ############### Place the descriptive text
    if args.pixelsize:
        f.text(0.84,0.53, "Pixel size $%1.2f \\times\\ %1.2f\\, nm^2$" % (args.pixelsize,args.pixelsize), size=10, ha='center', va='center')

    ############### Save data and finish up
    # finish layout
    f.tight_layout()
    
    plt.draw()
    
    # save PDF
    f.savefig(pdf,format='pdf',dpi=300) 
    
    #plt.show()
    
    plt.close("all")
    pdf.close()



# now run the overall analysis on ALL files
if False: #len(args.infile) > 1:
    
    # prepare the summary file
    pdf = PdfPages(basename+"-summary.pdf")
    
    d = pdf.infodict()
    d['Title'] = file_title
    d['Author'] = "Sebastian Werner"
    d['Subject'] = 'Data analysis for EELS/EDS maps'
    d['CreationDate'] = datetime.datetime.today()

    #prepare conditionition matrix - twice the resolution
    numbins *= 2
    # now eliminate some part from the heatmap
    conditioning_matrix = np.ones((numbins,numbins))
    a = 0
    b = 0
    r = 8
    y,x = np.ogrid[-a:numbins-a, -b:numbins-b]
    mask = x*x + y*y <= r*r
    conditioning_matrix[mask] = 0.1
    mask1 = x*x + y*y <= (r+1)**2
    conditioning_matrix[mask ^ mask1] = 0.25
    mask2 = x*x + y*y <= (r+2)**2
    conditioning_matrix[mask1 ^ mask2] = 0.5
    mask3 = x*x + y*y <= (r+3)**2
    conditioning_matrix[mask2 ^ mask3] = 0.75      
    

    heatmap, xedges, yedges = np.histogram2d(all_r, all_g, bins=numbins, range=[[0, 1], [0, 1]])
    f3 = plt.figure()
    f3.suptitle('Overall analysis (%s)' % (maintitle), fontsize=20)
    heat_plot = f3.add_subplot(121)

    # apply a the conditioning matrix
    heatmap = np.multiply(heatmap,conditioning_matrix)    

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    norm = plt.cm.colors.Normalize(vmin=0,vmax=1)

    # rotate to match the imshow style
    heatmap = heatmap.transpose()[::-1]
    qqq = heat_plot.imshow(heatmap/heatmap.max(),cmap=plt.cm.jet, interpolation="nearest", aspect="equal", extent=extent, rasterized=True)#, norm=norm)

    cbar = plt.colorbar(qqq,shrink=0.6) #.ax.tick_params(axis='y', direction='out')
    cbar.ax.tick_params(labelsize=10) 

    heat_plot.autoscale_view(True, False, False)
    heat_plot.set_xlabel("Relative $Mn$ intensity")
    heat_plot.set_xlim(0,1)
    heat_plot.set_xticks([0,0.5,1])#,1)
    heat_plot.set_ylabel("Relative $Co$ intensity")
    heat_plot.set_ylim(0,1)
    heat_plot.set_yticks([0,0.5,1.0])#,1)
    heat_plot.plot([0,1],[0,1],"k-",linewidth=0.5,alpha=0.5)

    # Principal component analysis
    pca_plot = f3.add_subplot(122)
    pca_plot.set_aspect(1)
    pca_plot.set_xlim(-3,3)
    pca_plot.set_ylim(-3,3)
    results = mpl.mlab.PCA(np.array([all_r, all_g]).T )

    center = results.mu
    rot90 = np.array([[0,-1],[1,0]])
    angle = np.arccos( results.Wt[0][0] )*180 / np.pi

    print "PCA center at (%.3f,%.3f): f1 %.1f s1 %.2f, f2 %.1f s2 %.2f, rot %.1f" % (center[0],center[1],results.fracs[0]*100,results.fracs[1]*100,results.sigma[0], results.sigma[1],angle)

    pca_plot.scatter(results.Y[:,0],results.Y[:,1],color="r",alpha=0.5,s=0.2,rasterized=True)
    pca_plot.set_title("$\Gamma_1$ %2.1f%% ($\sigma_1$ = %.3f)\n$\Gamma_2$ %2.1f%% ($\sigma_2$ = %.3f)" % (results.fracs[0]*100, results.sigma[0], results.fracs[1]*100, results.sigma[1]),fontsize=10)

    plt.draw()
    f3.tight_layout()
    f3.savefig(pdf,dpi=300,format='pdf')
    pdf.close()

logfile.close()










