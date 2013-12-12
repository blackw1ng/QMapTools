import sys
import os
import numpy as np
from numpy.ma.core import exp
import scipy as sp
import scipy.ndimage
from scipy.constants.constants import pi
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import pylab as pl
from scipy.signal import argrelextrema
import operator
from matplotlib.backends.backend_pdf import PdfPages
import datetime
import re

from .image_importer import *
from .inpaint import *

############### Library functions
def correlate1d(a,b):
    """" Calculate a 1d cross correlation"""
    return np.sum( (a - np.mean(a)) * (b - np.mean(b)) ) / ((a.size - 1) * np.std(a) * np.std(b) )

def manhattan(a,b):
    """Calculate the Taxicab or Manhattan norm.
    
    http://en.wikipedia.org/wiki/Norm_(mathematics)#Taxicab_norm_or_Manhattan_norm
    """
    return(np.sum(abs(a-b)))        

def zeronorm(a,b):
    """http://en.wikipedia.org/wiki/Norm_(mathematics)#Zero_norm"""
    diff = a - b
    return(np.linalg.norm(diff.ravel(), 0)  )
def compute_ssim(img_mat_1, img_mat_2):
    """Compute the Structure Similary Index between two images
    
    See:  Z. Wang, A. C. Bovik, H. R. Sheikh and E. P. Simoncelli, "Image quality assessment: From error visibility to structural similarity," IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600-612, Apr. 2004.
    Implementation after https://ece.uwaterloo.ca/~z70wang/research/ssim/
    """
    #Variables for Gaussian kernel definition
    gaussian_kernel_sigma=1.5
    gaussian_kernel_width=11
    gaussian_kernel=np.zeros((gaussian_kernel_width,gaussian_kernel_width))
    
    #Fill Gaussian kernel
    for i in range(gaussian_kernel_width):
        for j in range(gaussian_kernel_width):
            gaussian_kernel[i,j]=\
            (1/(2*pi*(gaussian_kernel_sigma**2)))*\
            exp(-(((i-5)**2)+((j-5)**2))/(2*(gaussian_kernel_sigma**2)))

    #Convert image matrices to double precision (like in the Matlab version)
    img_mat_1=img_mat_1.astype(np.float)
    img_mat_2=img_mat_2.astype(np.float)
    
    #Squares of input matrices
    img_mat_1_sq=img_mat_1**2
    img_mat_2_sq=img_mat_2**2
    img_mat_12=img_mat_1*img_mat_2
    
    #Means obtained by Gaussian filtering of inputs
    img_mat_mu_1=scipy.ndimage.filters.convolve(img_mat_1,gaussian_kernel)
    img_mat_mu_2=scipy.ndimage.filters.convolve(img_mat_2,gaussian_kernel)
        
    #Squares of means
    img_mat_mu_1_sq=img_mat_mu_1**2
    img_mat_mu_2_sq=img_mat_mu_2**2
    img_mat_mu_12=img_mat_mu_1*img_mat_mu_2
    
    #Variances obtained by Gaussian filtering of inputs' squares
    img_mat_sigma_1_sq=scipy.ndimage.filters.convolve(img_mat_1_sq,gaussian_kernel)
    img_mat_sigma_2_sq=scipy.ndimage.filters.convolve(img_mat_2_sq,gaussian_kernel)
    
    #Covariance
    img_mat_sigma_12=scipy.ndimage.filters.convolve(img_mat_12,gaussian_kernel)
    
    #Centered squares of variances
    img_mat_sigma_1_sq=img_mat_sigma_1_sq-img_mat_mu_1_sq
    img_mat_sigma_2_sq=img_mat_sigma_2_sq-img_mat_mu_2_sq
    img_mat_sigma_12=img_mat_sigma_12-img_mat_mu_12;
    
    #c1/c2 constants
    #First use: manual fitting
    c_1=6.5025
    c_2=58.5225
    
    #Second use: change k1,k2 & c1,c2 depend on L (width of color map)
    l=255
    k_1=0.01
    c_1=(k_1*l)**2
    k_2=0.03
    c_2=(k_2*l)**2
    
    #Numerator of SSIM
    num_ssim=(2*img_mat_mu_12+c_1)*(2*img_mat_sigma_12+c_2)
    #Denominator of SSIM
    den_ssim=(img_mat_mu_1_sq+img_mat_mu_2_sq+c_1)*\
    (img_mat_sigma_1_sq+img_mat_sigma_2_sq+c_2)
    #SSIM
    ssim_map=num_ssim/den_ssim
    index=np.average(ssim_map)

    return index
def azimuthalAverage(image, center=None):
    """
    Calculate the azimuthally averaged radial profile.
    image - The 2D image
    center - The [x,y] pixel coordinatepsd2D = np.abs( F2 )**2
            is used as the center. The default is None, which
            then uses the center of the image (including fractional pixels).
    Credit to: http://www.astrobetter.com/fourier-transforms-of-images-in-python/
    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)
    
    if not center:
        center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])
        
    r = np.hypot(x - center[0], y - center[1])
    
    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]
    
    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)
    
    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]       # location of changed radius
    nr = rind[1:] - rind[:-1]        # number of radius bin
    
    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]
    
    radial_prof = tbin / nr
    
    return radial_prof
def easypca(A,numpc=0):
 """ performs principal components analysis 
     (PCA) on the n-by-p data matrix A
     Rows of A correspond to observations, columns to variables. 

 Returns :  
  score : 
    the principal component scores; that is, the representation 
    of A in the principal component space. Rows of SCORE 
    correspond to observations, columns to components.
  coeff :
    is a p-by-p matrix, each column containing coefficients 
    for one principal component.
  eigenvalues : 
    a vector containing the eigenvalues 
    of the covariance matrix of A.
  contributions :
    a vector containing the percentage contributions of each of
    the principal components
  
 Inspired by:
    http://glowingpython.blogspot.com/2011/07/principal-component-analysis-with-numpy.html
    http://glowingpython.blogspot.it/2011/07/pca-and-image-compression-with-numpy.html
 """    
 
 # subtract the mean (along columns)
 M = (A-np.mean(A.T,axis=1)).T
 
 # computing eigenvalues and eigenvectors of covariance matrix
 [eigenvalues,coeff] = np.linalg.eig(np.cov(M))
 
 # how many eigenvectors do we have?
 p = np.size(coeff,axis=1)
 
 # sorting the eigenvalues
 idx = np.argsort(eigenvalues)
 # in ascending order 
 idx = idx[::-1]       
 # sorting eigenvectors according to the sorted eigenvalues
 coeff = coeff[:,idx]
 # sorting eigenvalues
 eigenvalues = eigenvalues[idx]
 # cutting down some of those eigenvectors 
 if numpc < p and numpc > 0:
   coeff = coeff[:,range(numpc)] # cutting some PCs
   eigenvalues = eigenvalues[:,range(numpc)]
  
 # projection of the data in the new space 
 projected = np.dot(coeff.T,M) 
 
 # and finally contribute contributions.
 contributions = np.cumsum(eigenvalues)/np.sum(eigenvalues)
 
 return projected,coeff,eigenvalues,contributions
 
 
class mlabPCA:
    def __init__(self, a):
        """
        compute the SVD of a and store data for PCA.  Use project to
        project the data onto a reduced set of dimensions

        Inputs:

          *a*: a numobservations x numdims array

        Attrs:

          *a* a centered unit sigma version of input a

          *numrows*, *numcols*: the dimensions of a

          *mu* : a numdims array of means of a

          *sigma* : a numdims array of atandard deviation of a

          *fracs* : the proportion of variance of each of the principal components
          
          *s* : eigenvalues of the decomposition

          *Wt* : the weight vector for projecting a numdims point or array into PCA space

          *Y* : a projected into PCA space


        The factor loadings are in the Wt factor, ie the factor
        loadings for the 1st principal component are given by Wt[0]
        This row is also the 1st eigenvector

        """
        n, m = a.shape
        if n<m:
            raise RuntimeError('we assume data in a is organized with numrows>numcols')

        self.numrows, self.numcols = n, m
        self.mu = a.mean(axis=0)
        self.sigma = a.std(axis=0)

        a = self.center(a)

        self.a = a

        U, s, Vh = np.linalg.svd(a, full_matrices=False)
        
        # Note: .H indicates the conjugate transposed / Hermitian.
        
        # The SVD is commonly written as a = U s V.H.
        # If U is a unitary matrix, it means that it satisfies U.H = inv(U).
        
        # The rows of Vh are the eigenvectors of a.H a.
        # The columns of U are the eigenvectors of a a.H. 
        # For row i in Vh and column i in U, the corresponding eigenvalue is s[i]**2.
        
        
        # export rotation matrix
        # its rows are eigenvectors.
        self.Wt = Vh
        
        # save the transposed coordinates
        Y = np.dot(Vh, a.T).T
        self.Y = Y
        
        # save the eigenvalues
        self.s = s**2
        
        # and now the contribution of the individual components
        vars = self.s/float(len(s))
        self.fracs = vars/vars.sum()

    def project(self, x, minfrac=0.):
        'project x onto the principle axes, dropping any axes where fraction of variance<minfrac'
        x = np.asarray(x)

        ndims = len(x.shape)

        if (x.shape[-1]!=self.numcols):
            raise ValueError('Expected an array with dims[-1]==%d'%self.numcols)


        Y = np.dot(self.Wt, self.center(x).T).T
        mask = self.fracs>=minfrac
        if ndims==2:
            Yreduced = Y[:,mask]
        else:
            Yreduced = Y[mask]
        return Yreduced



    def center(self, x):
        'center the data using the mean and sigma from training set a'
        return (x - self.mu)/self.sigma



    @staticmethod
    def _get_colinear():
        c0 = np.array([
            0.19294738,  0.6202667 ,  0.45962655,  0.07608613,  0.135818  ,
            0.83580842,  0.07218851,  0.48318321,  0.84472463,  0.18348462,
            0.81585306,  0.96923926,  0.12835919,  0.35075355,  0.15807861,
            0.837437  ,  0.10824303,  0.1723387 ,  0.43926494,  0.83705486])

        c1 = np.array([
            -1.17705601, -0.513883  , -0.26614584,  0.88067144,  1.00474954,
            -1.1616545 ,  0.0266109 ,  0.38227157,  1.80489433,  0.21472396,
            -1.41920399, -2.08158544, -0.10559009,  1.68999268,  0.34847107,
            -0.4685737 ,  1.23980423, -0.14638744, -0.35907697,  0.22442616])

        c2 = c0 + 2*c1
        c3 = -3*c0 + 4*c1
        a = np.array([c3, c0, c1, c2]).T
        return a
    



def smooth(x,window_len=5,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    window_len += (1 - window_len % 2)
    
    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y[(window_len/2-1):-(window_len/2+1)]


def do_correlation(r,g,b,logfile=False):
    """Calculation and log correlation metrics on given (r,g,b) image.
    
    r,g,b:     numpy array with pixel intensities (r,g or b)
    logfile:   file pointer to logfile.
    
    TODO: Needs major re-write as this is just poor scrap code :(
    """
    # apply std deviation correlation normalization
    rn = normalize(np.copy(r))
    gn = normalize(np.copy(g))
    bn = normalize(np.copy(b))
    
    Has = {"r" : False, "g" : False, "b": False}
    if r.max() > 25:
        Has["r"] = True
    if g.max() > 25:
        Has["g"] = True
    if b.max() > 40:
        Has["b"] = True
        
    # autocorrelation as standard 
    if Has['g']:
        gg = signal.fftconvolve(gn,gn[::-1, ::-1],mode="same")
    if Has['r']:
        rr = signal.fftconvolve(rn,rn[::-1, ::-1],mode="same")
    
    cc2 = '2D cc: '
    ssim = 'SSIM: '
    
    if Has['r'] and Has['g']:
        temp = correlate1d(g,r)
        print "1d Correlation of g to r is cc: %.3f norm: %i zero: %i" % (temp, manhattan(g,r), zeronorm(g,r))
        if logfile:
            logfile.write("%.4f," % (temp))
    
        rg = signal.fftconvolve(rn,gn[::-1, ::-1],mode="same")
        temp = float(rg.max()/gg.max())
        print "2d correlation of g to r is %.3f" % temp 
        cc2 += "rg = %2.1f%% " % (temp * 100)
        if logfile:
            logfile.write("%.4f," % (temp))
    
        temp = compute_ssim(r,g)
        print "SSIM correlation of g to r is %.3f" % temp
        ssim += "rg = %2.1f%% " % (temp * 100)
        if logfile:
            logfile.write("%.4f," % (temp))
    else:
        if logfile:
            logfile.write("NaN,NaN,NaN," % (temp))
        
    
    if Has['r'] and Has['b']:
        temp = correlate1d(r,b)
        print "1d Correlation of b to r is cc: %.3f norm: %i zero: %i" % (temp, manhattan(r,b), zeronorm(r,b))
        if logfile:
            logfile.write("%.4f," % (temp))
        
        rb = signal.fftconvolve(rn,bn[::-1, ::-1],mode="same")
        temp = float(rb.max()/rr.max())
        print "2d correlation of b to r is %.3f" % temp
        cc2 += "rb = %2.1f%% " % (temp * 100)
        if logfile:
            logfile.write("%.4f," % (temp))
    
        temp = compute_ssim(r,b)
        print "SSIM correlation of b to r is %.3f" % temp
        ssim += "rb = %2.1f%%" % (temp * 100)
        if logfile:
            logfile.write("%.4f," % (temp))
    else:
        if logfile:
            logfile.write("NaN,NaN,NaN," % (temp))
    
    if Has['g'] and Has['b']:    
        temp = correlate1d(g,b)
        print "1d Correlation of g to b is cc: %.3f norm: %i zero: %i" % (temp, manhattan(g,b), zeronorm(g,b))
        if logfile:
            logfile.write("%.4f," % (temp))
        
        gb = signal.fftconvolve(gn,bn[::-1, ::-1],mode="same")
        temp = float(gb.max()/gg.max())
        print "2d correlation of g to b is %.3f" % temp
        cc2 += "gb = %2.1f%% " % (temp * 100)
        if logfile:
            logfile.write("%.4f," % (temp))
    
        temp = compute_ssim(b,g)    
        print "SSIM correlation of g to b is %.3f" % temp
        ssim += "gb = %2.1f%%" % (temp * 100)
        if logfile:
            logfile.write("%.4f," % (temp))
    else:
        if logfile:
            logfile.write("NaN,NaN,NaN," % (temp))

    return(cc2+"\n"+ssim)



def draw_scatterplot(subplot,component_x,component_y,label_x="$A$",label_y="$B$",color_dots=False, normalized=True):
    """Produce a parity-plot for a given set of two components. 
    
        subplot:        Subplot object to draw in.
        component_x:    numpy array with pixel intensities (r,g or b)
        component_y:    numpy array with pixel intensities (r,g or b)
        label_x:        Label for the component X to be placed on x-axis
        label_y:        Label for the component Y to be placed on y-axis
        color_dots:     Colors dots using the supplied dstacked(r,g,b) or numpy array of the image
        normalized:     Show 0..255 or 0..1 as scaling factor.
    """

    # xy_plot = plt.subplot2grid((2,3),(row,0),axisbg="black")
    print "\tScatter parity plot..."
    
    subplot.set_title("Parity plot")
    subplot.set_axis_bgcolor("black")
    
    if normalized:
        component_x = np.copy(component_x)/255.0
        component_y = np.copy(component_y)/255.0
        graph_max = 1.0
        label_prefix = "Relative "
    else:
        graph_max = 255.0
        label_prefix = "Absolute "
    
    if type(color_dots) != type(False):
        graph_color = (color_dots/255.0).reshape(-1,3).tolist()
    else:
        # no color tuple given, so just white little dots
        graph_color="w"
    
    subplot.scatter(component_x.tolist(),component_y.tolist(),s=1,color=graph_color,rasterized=True)
    
    subplot.plot([0,graph_max],[0,graph_max],"r-",linewidth=0.5)
    subplot.set_xlabel(label_prefix+label_x+" intensity")
    subplot.set_xlim(0,graph_max)
    subplot.set_xticks([0,graph_max/2.0,graph_max])
    subplot.set_ylabel("Relative "+label_y+" intensity")
    subplot.set_ylim(0,graph_max)
    subplot.set_yticks([0,graph_max/2.0,graph_max])
    

def draw_heatmap(subplot, component_x,component_y, label_x="$A$", label_y="$B$", normalized=True, cutoff=0, contour=False, enlarged=False, interpolate=False, remove_zero=False,return_results=False, numbins=20, conditioning_matrix=np.ones((20,20))):
    """Produce a heatmap-plot for a given set of two components. 
    
        subplot:        Subplot object to draw in.
        component_x:    numpy array with pixel intensities (r,g or b)
        component_y:    numpy array with pixel intensities (r,g or b)
        label_x:        Label for the component X to be placed on x-axis
        label_y:        Label for the component Y to be placed on y-axis
        normalized:     Scale intensities by count or to 0.1 scale - both x/y and colorbar.
        countour:       Use countourplot instead of heatmap.
        cutoff:         Lowest intensity to show (0..255)
        return_array:   return the masked individual array (for mass processing)
    """
    
    print "\tHeatmap parity plot..."
    #heat_plot = plt.subplot2grid((2,3),(row,1),aspect="equal") 

    if normalized:
        component_x = np.copy(component_x)/255.0
        component_y = np.copy(component_y)/255.0
        cutoff /= 255.0
        graph_max = 1
        label_prefix = "Relative "
    else:
        graph_max = 255
        label_prefix = "Absolute "
    
    # ignore the black background noise
    masked_x = component_x[(component_x >= cutoff) & (component_y >= cutoff)].flatten() 
    masked_y = component_y[(component_x >= cutoff) & (component_y >= cutoff)].flatten()
    
    # create the 2D histogram
    heatmap, xedges, yedges = np.histogram2d(masked_x, masked_y, bins=numbins, range=[[0, graph_max], [0, graph_max]])
    
    # apply a the conditioning matrix
    heatmap = np.multiply(heatmap,conditioning_matrix)    
    
    
    
    
    # now we mask those out, to ignore them
    if remove_zero or interpolate:
        # now we remove the zeros
        heatmap[heatmap == 0] = "NaN"
        heatmap = np.ma.masked_invalid(heatmap)
        
    if interpolate:
        heatmap = replace_nans(heatmap, 2, 0.1, 1,"idw")

        
    if normalized:
        heatmap = heatmap/heatmap.max()
        norm = plt.cm.colors.Normalize(vmin=0,vmax=1)
        
    if contour:
        X, Y = 0.5*(xedges[1:]+xedges[:-1]), 0.5*(yedges[1:]+yedges[:-1])
        qqq = subplot.contourf(X, Y, heatmap, cmap=plt.cm.jet)
        #qqq = plt.pcolor(xedges, yedges, heatmap)
        
    else:
        # rotate to match the imshow style
        heatmap = heatmap.T[::-1]
        extent = [xedges[0], xedges[-1],yedges[0], yedges[-1]]
        # show the heatmap
        qqq = subplot.imshow(heatmap ,cmap=plt.cm.jet, interpolation="nearest", aspect="equal", extent=extent, rasterized=True)
        
    # place the colorbar
    if not enlarged:
        cbar = plt.colorbar(qqq,shrink=0.5) 
        cbar.ax.tick_params(labelsize=10) #.ax.tick_params(axis='y', direction='out')
    else:
        cbar = plt.colorbar(qqq)
        
    # label & scale
    subplot.set_title("Heatmap analysis")
    subplot.autoscale_view(True, False, False)
    subplot.set_xlabel(label_prefix+label_x+" intensity")
    subplot.set_xlim(0,graph_max)
    subplot.set_xticks([0,graph_max/2.0,graph_max])#,1)
    subplot.set_ylabel(label_prefix+label_y+" intensity")
    subplot.set_ylim(0,graph_max)
    subplot.set_yticks([0,graph_max/2.0,graph_max])#,1)
    
    # plot the parity line in the end
    subplot.plot([0,graph_max],[0,graph_max],"w-",linewidth=0.5,alpha=0.5)
    
    if return_results:
        return(masked_x,masked,y)
        


def draw_pca(subplot,component_x,component_y,logfile=False,return_results=False,enlarged=False):
    """Produce a heatmap-plot for a given set of two components. 
    
        subplot:        Subplot object to draw in.
        component_x:    numpy array with pixel intensities (r,g or b)
        component_y:    numpy array with pixel intensities (r,g or b)
        return_results:   return the masked individual array (for mass processing)
    """
    
    print "Statistical analysis...."
    
    # set plot
    subplot.set_aspect(1)
    subplot.set_xlim(-5,5)
    subplot.set_ylim(-5,5)
    
    ## TODO: implement 3D mode...
    #pca_plot = Axes3D(f)
    #pca_plot.scatter(rN.tolist(),gN.tolist(), bN.tolist() ,color=c.tolist(),s=5)
    #pca_plot.set_zlabel("Relative $O$ intensity")
    #pca_plot.set_zlim(0,1)
    
    # run the analysis
    #results = mpl.mlab.PCA(np.array([rN.flatten(),gN.flatten()]).T )
    results = mpl.mlab.PCA(np.array([component_x.flatten(),component_y.flatten()]).T )
    
    center = results.mu
    rot90 = np.array([[0,-1],[1,0]])
    angle = np.arccos( results.Wt[0][0] )*180 / np.pi
    
    print "\tPCA center at (%.3f,%.3f): f1 %.1f s1 %.2f, f2 %.1f s2 %.2f, rot %.1f" % (center[0],center[1], results.fracs[0]*100,results.sigma[0]/255.0, results.fracs[1]*100, results.sigma[1]/255.0,angle)
    
    #contributions = np.cumsum(results.fracs)/np.sum(result.fracs)
    
    #subplot.scatter(results.a[:,0],results.a[:,1], color="b",alpha=0.5,s=2,rasterized=True)
    
    if enlarged:
        fs = 16
        ps= 6
    else:
        fs = 10
        ps = 0.1
        subplot.set_title("$\Gamma_1$ %2.1f%% ($\sigma_1$ = %.3f)\n$\Gamma_2$ %2.1f%% ($\sigma_2$ = %.3f)" % (results.fracs[0]*100, results.sigma[0]/255.0, results.fracs[1]*100, results.sigma[1]/255.0),fontsize=fs)
        
    subplot.scatter(results.Y[:,0], results.Y[:,1], color="r", alpha=0.5,s=ps,rasterized=True)
    
   
    
    if logfile:
        logfile.write("%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f" % (center[0], center[1], angle, results.fracs[0],  results.sigma[0]/255.0, results.fracs[1], results.sigma[1]/255.0))
  
    ## mark some stuff in the XY plot
    #subplot.plot( *zip( center, results.Wt.dot( np.array( [0, results.sigma[0] ] ) ) + center ), linewidth=1, color="g")
    #subplot.plot( *zip( center, rot90.dot( results.Wt.dot( np.array([0, results.sigma[1] ] ) ) ) + center ), linewidth=1, color="g")
    #subplot.add_patch( mpl.patches.Ellipse( center, results.sigma[0], results.sigma[1], angle=angle, alpha=0.2, color="g") )
    
    if return_results:
        return results

def draw_image(subplot,file_basename,which="haadf",postfix=""):
    
    # try opening and inserting appropriate file
    if which == "haadf":
        image_title = "HAADF overview"
        if postfix == "":
            postfix = "h"
    elif which == "map":
        image_title = "RGB map"
    else:
        print 'Not implemented.'
        raise 
    
    # try to open the map - try .png first, then .jpg
    try:
        temp = Image.open(file_basename+postfix+".png")
    except IOError:
        try:
            temp = Image.open(file_basename+postfix+".jpg")
        except IOError:
            print which+' file '+file_basename+postfix+".(png|jpg) not found."
            raise
    
    subplot.set_title(image_title)
    subplot.set_axis_off()
    subplot.imshow(temp,interpolation="bicubic") 

def draw_eds(subplot,file_basename):
    pass
def draw_powerspectrum(subplot,r,g,b,cutoff=0):
    
    r_fft = np.fft.fft2(r)
    r_fft = np.fft.fftshift(r_fft)
    r_psd2D = np.abs( r_fft )**2
    r_psd1D = azimuthalAverage(r_psd2D)
    
    g_fft = np.fft.fft2(g)
    g_fft = np.fft.fftshift(g_fft)
    g_psd2D = np.abs( g_fft )**2
    g_psd1D = azimuthalAverage(g_psd2D)
    
    subplot.set_title("1D Powerspectrum")
    subplot.semilogy( g_psd1D, color="g",linewidth=2 )
    subplot.semilogy( r_psd1D, color="r",linewidth=2 )
    subplot.set_xlabel("Spatial frequency $k$ / $nm^{-1}$")
    subplot.set_ylabel("Power density / -")
    #subplot.set_xlim(0,130)
    
    if cutoff > 0:
        rx = np.copy(r)
        rx[ r < cutoff ] = 0

        gx = np.copy(g)
        gx[ g < cutoff ] = 0
        
        rx_fft = np.fft.fft2(rx)
        rx_fft = np.fft.fftshift(rx_fft)
        rx_psd2D = np.abs( rx_fft )**2
        rx_psd1D = azimuthalAverage(rx_psd2D)
        
        gx_fft = np.fft.fft2(gx)
        gx_fft = np.fft.fftshift(gx_fft)
        gx_psd2D = np.abs( gx_fft )**2
        gx_psd1D = azimuthalAverage(gx_psd2D)
        
        subplot.semilogy( gx_psd1D, color="#008800", linewidth=1 )
        subplot.semilogy( rx_psd1D, color="#880000", linewidth=1 )
        
    
    
def draw_histogram(subplot,r,g,b):
    subplot.set_title("Component histogram")
    subplot.set_aspect("auto")
    
    if r.max() > 25:
        n, bins, patches = subplot.hist(r[r>1].flatten(), 50, normed=1, histtype='stepfilled', facecolor="r",alpha=0.5)
    if g.max() > 25:
        n, bins, patches = subplot.hist(g[g>1].flatten(), 50, normed=1, histtype='stepfilled', facecolor="g",alpha=0.5)
    if b.max() > 40:
        n, bins, patches = subplot.hist(b[b>1].flatten(), 50, normed=1, histtype='stepfilled', facecolor="b",alpha=0.5)
        
    subplot.set_xlim(0,255)
    #subplot.set_ylim(0,0.5)
    subplot.set_xlabel("Channel absolute intensity")
    subplot.set_ylabel("Probability")
def draw_threshold(subplot,r,g,b,threshold_r=0,threshold_g=0,threshold_b=0):

    subplot.set_title("Threshold view")
    subplot.set_axis_off()
    
    rx = np.copy(r)
    rx[ r <= threshold_r ] = 0

    gx = np.copy(g)
    gx[ g <= threshold_g ] = 0
    
    bx = np.copy(b)
    bx[ b <= threshold_b ] = 0
    
    temp = np.dstack( (rx,gx,bx) ).astype(np.uint8)
    
    subplot.imshow(temp,interpolation="bicubic",aspect="equal")
    
def draw_colocation(subplot,img,label_r="$Mn$",label_g="$Co$",label_b="$O$",normalized=True,cutoff=70):
    """Generate a probability of co-location plot.
    
    Calculates the percentage of which a given 'green' intensity also has 
    'red' present.
    
    img:        PIL image object
    label_r:    Element label for red channel
    label_g:    Element label for green channel
    label_b:    Element label for blue channel
    normalized: Normalize axes
    cutoff:     Absolute pixel intensity to count from (ignore noise)
    """
    
    R_count = np.zeros(256)
    G_count = np.zeros(256)
    
    numbins = 32
    
    #totalRedPixels = len(r[r>red_cutoff])

    for pixel in img.getdata():
        
        if ((pixel[0] == 0) and (pixel[1] == 0)):
            # reject pure black
            continue
            
        else:
            # check if red value is higher than cutoff (noise)
            if pixel[0] > cutoff:
                # count red HIT at the green intensity of the pixel
                R_count[pixel[1]] += 1
            
            # count green HIT at the green intensity of the pixel
            G_count[pixel[1]] += 1
    
    if normalized:
        graph_max = 1.0
        label_prefix = "Normalized "
    else:
        graph_max = 256.0
        label_prefix = "Absolute "
    
    #print R_count[0:10].astype(int)
    #print G_count[0:10].astype(int)
    
    #results = np.array(R_count,dtype=float) / np.array(G_count,dtype=float)
    
    # now re-binning in a fast way: from a 256 array down to a 256/numbins array
    # by: reshaping it to an array with x: numbins-columns and y: 256/numbins
    # values which then get summed...
    R_binned = R_count.reshape([numbins, len(R_count) / numbins]).sum(1) 
    G_binned = G_count.reshape([numbins, len(G_count) / numbins]).sum(1) 
    #results = results.reshape([numbins, len(results)/numbins]).mean(1)
    
    # now reject small number statistics
    #R_binned[G_binned < 30] = "NaN"
    
    # now normalize by pixels in that very bin
    results = R_binned / G_binned
    
    #print results
    
    # the data now also has a good number of "NaN" values from division by 0...
    # and now we also make the 100% ones to NaN
    #results[results == 1] = "NaN"
    # now we mask those out, to ignore them
    results = np.ma.masked_invalid(results)
    
    # construct plot x-axis
    x_axis = np.arange(0,graph_max,graph_max/float(numbins))
    
    # and plot
    subplot.bar(x_axis, results, facecolor="r", edgecolor='none',width=1/float(numbins))
    
    masked = np.zeros(numbins)
    masked[np.ma.getmaskarray(results)] = 1
    subplot.bar(x_axis, masked, facecolor="#cccccc", edgecolor='none',width=1/float(numbins))
    
    # unbinned raw.
    #subplot.bar(np.arange(0,256, 1 ), np.array(R_count,dtype=float) / np.array(G_count,dtype=float), facecolor="r", width=0.1, edgecolor='none') 
    
    subplot.set_aspect("auto")
    subplot.set_title("Colocation probability")
    subplot.set_xlim(0,graph_max)
    subplot.set_ylim(0,1)
    subplot.set_xlabel(label_prefix+label_g+" intensity")
    subplot.set_ylabel("Fraction of "+label_g+" also having "+label_r)
        
def draw_eds(subplot):
    pass
def draw_eels(subplot,file_basename,smooth):
    """Generate a plot from EELS data saved by CSI.
    Currently highlights just Co, Mn and O... generic code to be added.
    
    subplot:        Subplot object to draw in.
    file_basename:  Filename (no suffix) to read from (adds .txt)
    smooth:         Window size for Hanning type smoothing 
    """
    data = pl.csv2rec(file_basename+".txt", delimiter="\t", names=["eV","raw","ev2","ref","ev3","bg","ev4","sign"])

    
    #eels_plot.set_title("EELS spectrum of %s" % (sys.argv[2]))
    subplot.set_ylabel("Counts / $A.U.$")
    subplot.set_xlabel("Electron energy loss / $\Delta eV$ ")
    subplot.set_ylim(data["raw"].min()*0.3,data["raw"].max()*1.15)
    subplot.plot(data["eV"],data["raw"],label="raw data",color="grey")

    smoothed = smooth(data["raw"],window_len=smooth,window="hanning")

    subplot.plot(data["eV"],smoothed,color="k",label="smoothed data")

    # find peaks
    peak_index = argrelextrema(smoothed,np.greater,order=30)[0].tolist()[1:]

    ranges = [(535,580), (630,670), (770,810)]  

    # annotate each peak

    for peak in peak_index:
        if any(lower <= data["eV"][peak] <= upper for (lower, upper) in ranges):
            subplot.annotate('%.1f' % (data["eV"][peak]),style='italic',size=11, xy=(data["eV"][peak], data["raw"][peak]), xycoords="data", textcoords="offset points", xytext=(0, 25), horizontalalignment='left', verticalalignment='top', arrowprops=dict(facecolor='black',arrowstyle="->",shrinkB=7,connectionstyle="arc3"))

    # mark regions
    subplot.axvspan(535,580,color="b",alpha=0.2) # o
    subplot.annotate("O\n$K$-edge", xy=(554,eels_plot.get_ylim()[0]), xycoords="data", va="bottom", ha="center", size=8)
    subplot.axvspan(630,670,color="r",alpha=0.2) # mn
    subplot.annotate("Mn\n $L_{2,3}$-edge", xy=(647,eels_plot.get_ylim()[0]), xycoords="data", va="bottom", ha="center", size=8)
    subplot.axvspan(770,810,color="g",alpha=0.2) # Co
    subplot.annotate("Co\n$L_{2,3}$-edge", xy=(790,eels_plot.get_ylim()[0]), xycoords="data", va="bottom", ha="center",size=8)
    subplot.set_title("Integral EELS spectrum")
