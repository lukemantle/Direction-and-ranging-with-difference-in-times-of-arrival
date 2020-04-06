'''
Author: Luke Mantle
Model for T2 project
2020-Jan-23

Mathematical model for apparatus
Takes location of three microphones and sound source and generates TDOAs for
each pair of microphones. Then, based on microphone positions and TDOAs (and
not the sound source location), draws parametric functions for each pair of
microphones and generates probability density function ("heatmap") describing
the location of the sound source.

Parameters found towards the bottom of the program
'''

import numpy as np
import matplotlib.pyplot as plt

def makePlot(xBoundL=-50,xBoundR=50,yBoundL=-50,yBoundR=50, # plot boundaries
             xPixScal=1,yPixScal=1,                         # pixel scaling
             micSize=1,                                     # mic. dot size
             errConst=2,                                    # error constant
             xMic=[0,1],yMic=[0,0],                         # mic. positions
             source=[0,0],                                  # source pos. (x,y)
            ):     
    
    # set up grid for plot:
    # create x and y axes of plot (equivalent to meshgrid)
    x = np.arange( xBoundL, xBoundR, xPixScal )
    y = np.arange( yBoundL, yBoundR, yPixScal )
    
    # tile each axis (because x-axis needs to be same for each row
    # of y-axis and vice versa)
    X = np.tile( x,   [y.size,1])
    Y = np.tile( y,   [x.size,1])
    Y = np.rot90(Y)	# rotate y-axis
    
    d12 = getDelay(source[0],source[1],xMic[0],yMic[0],xMic[1],yMic[1])
    d13 = getDelay(source[0],source[1],xMic[0],yMic[0],xMic[2],yMic[2])
    d23 = getDelay(source[0],source[1],xMic[1],yMic[1],xMic[2],yMic[2])

    
    # assign graph
    data = function(X, Y,
                    xMic[0],yMic[0],xMic[1],yMic[1],xMic[2],yMic[2],
                    d12,d13,d23,
                    errConst)

    plt.imshow(data,extent=[xBoundL,xBoundR,yBoundL,yBoundR])
    
    # make contours
    plt.contour(X,Y,
                curve(X,Y,xMic[0],yMic[0],xMic[1],yMic[1]),
                [d12],colors="red")                        # mics 1, 2
    plt.contour(X,Y,
                curve(X,Y,xMic[0],yMic[0],xMic[2],yMic[2]),
                [d13],colors="red")                        # mics 1, 3
    plt.contour(X,Y,
                curve(X,Y,xMic[1],yMic[1],xMic[2],yMic[2]),
                [d23],colors="red")                        # mics 2, 3
    
    # plot microphones
    plt.plot(xMic,yMic,marker='o',linestyle='None',
             color='red',markersize=micSize)

    plt.show()		# make figure pop up in new window

# if function is error from curve, define curve to be plotted:
def curve(x,y,x1,y1,x2,y2):
    return np.sqrt((x-x1)**2+(y-y1)**2)-np.sqrt((x-x2)**2+(y-y2)**2)

# gaussian curve for error:
def gauss(theoretical, measured, errConst):
	# constants for gauss:
	return np.exp(-(((theoretical-measured)/errConst)**2))

# define function to be plotted:
def function(xFunc, yFunc, x1, y1, x2, y2, x3, y3, d12, d13, d23, errConst):
	return  ( gauss(curve(xFunc, yFunc,x1,y1,x2,y2),d12,errConst)
            * gauss(curve(xFunc, yFunc,x1,y1,x3,y3),d13,errConst)
            * gauss(curve(xFunc, yFunc,x2,y2,x3,y3),d23,errConst)
            )

def getDelay(xs,ys,x1,y1,x2,y2):
    return np.sqrt((x1-xs)**2+(y1-ys)**2)-np.sqrt((x2-xs)**2+(y2-ys)**2)
            

makePlot(	
			xBoundL=-20,         # left bound of plot
            xBoundR=20,          # right bound of plot
            yBoundL=-20,         # bottom bound of plot
            yBoundR=20,          # top bound of plot
            xPixScal=0.1,         # size of x-pixels
            yPixScal=0.1,      # size of y-pixels
            xMic=[0,10,5],     # mic positions [x1, x2, x3]
            yMic=[0,0,10],    # mic positions [y1, y2, y3]
            source=[0,4,.001],       # source position [x, y]
            micSize=8,          # display size of microphones
            errConst=5        # width of gauss curve for heatmap
            
		)