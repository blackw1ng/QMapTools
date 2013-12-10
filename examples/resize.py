#! /opt/local/bin/python
import sys
import os
import Image
import argparse
import numpy as np
import argparse
from qmaptools import *


parser = argparse.ArgumentParser(description='Quick image resizer.')
parser.add_argument('infile', metavar='IMAGE.ext', nargs='+', type=argparse.FileType('r'), help='(.png|bmp|.gif|.jpg|.tif) file with RGB intensity.')
parser.add_argument('--filter',choices=['nearest','bicubic','antialias'], help="Which filter to use for resizing.",default='nearest')
parser.add_argument('--resize',type=int,help="Resize incoming raw data to new size (nearest)",default=0)
parser.add_argument('--combine',action='store_true',help="Combine _Xn images.")

args = parser.parse_args()

# The filter argument can be one of NEAREST (use nearest neighbour), BILINEAR (linear interpolation in a 2x2 environment), BICUBIC (cubic spline interpolation in a 4x4 environment), or ANTIALIAS (a high-quality downsampling filter). If omitted, or if the image has mode "1" or "P", it is set to NEAREST.

if args.filter == "bicubic":
    resizefilter = Image.BICUBIC
elif args.filter == "antialias":
    resizefilter = Image.ANTIALIAS
else:
    resizefilter = Image.NEAREST

for file in args.infile:
	
    img = Image.open(file)
    file_basename = os.path.splitext(file.name)[0]
    
    
    if args.combine:
        print "Combining %s" % (file_basename)
        try:
            imgMn = Image.open(file_basename+"_Mn.png")
        except:
            print "Mn error:", sys.exc_info()
            
        try:
            imgCo =  Image.open(file_basename+"_Co.png")
        except:
            print "Co error:", sys.exc_info()
            
        try:
            Mn = importchan(PIL2array(imgMn),0)
            Co = importchan(PIL2array(imgCo),1)
            stacked = np.dstack((Mn,Co,np.zeros(Mn.shape,dtype=np.int8))).astype(np.uint8)
            stackedimg = array2PIL(stacked,Mn.shape)
            
            if file_basename[-1] == "n":
                file_basename = file_basename[:-1]
            
            stackedimg.save(file_basename+"c.png")
            print "\tWrote %s" % (file_basename+"c.png")
        except:
             print "Unexpected error:", sys.exc_info()[0]
            
    else:    
        print "Importing %s (%i x %i)" % (file, img.size[0], img.size[1])
        if file_basename[-1] == "n":
            file_basename = file_basename[:-1]
            
        if args.resize > 0:
            img.thumbnail((args.resize,args.resize),resizefilter)
            print "\tResizing to (%i x %i) using %s" % (img.size[0], img.size[1], args.filter)
            img.save(file_basename+"r.png")
