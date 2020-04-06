'''
Author: Luke Mantle
2020

Created for T2 project

Reads set of data points and creates scatter plot of all the data points'
intersection points along with a heatmap and set of curves based on the
points' distribution.
'''

import numpy as np
import matplotlib.pyplot as plt
import collections
import scipy.spatial as spatial
import scipy.spatial.distance as dist
import scipy.cluster.hierarchy as hier

labelContours = False

# plot boundaries
xBoundL=-1
xBoundR=1
yBoundL=-1
yBoundR=1

# pixel scaling
xScalCoarse=0.01
yScalCoarse=xScalCoarse
xScalFine=0.001
yScalFine=xScalFine                       
micSize=3           # mic. dot size  
sourceSize=3
pointSize=2

# gaussian curve for error:
def gauss(x, sigma, mu):
    y = (1 / (sigma * np.sqrt(2*np.pi))) * np.exp(-1/2 * ((x - mu) / sigma) ** 2)
    return y

def getDelay(xs,ys,x1,y1,x2,y2):
    return np.sqrt((x1-xs)**2+(y1-ys)**2)-np.sqrt((x2-xs)**2+(y2-ys)**2)

# if function is error from curve, define curve to be plotted:
def curve(x,y,x1,y1,x2,y2):
    return np.sqrt((x-x1)**2+(y-y1)**2)-np.sqrt((x-x2)**2+(y-y2)**2)

# read data
fileNameIn = "source 6"
lines = open(fileNameIn, "r").readlines()
linesNoBreak = []
for i in lines:
    if i[-1:] == "\n":
        linesNoBreak.append(i[:-1])     # remove \n from each line
    else:
        linesNoBreak.append(i)

nData = len(linesNoBreak)               # number of data points
inArr = np.zeros((nData,11))
i = 0
while i < nData:
    lineArr = np.array(linesNoBreak[i].split(',')).astype(np.float)
    inArr[i,:] = lineArr
    i += 1

# get mic. positions: assume mic positions in first line
# are mic positions for whole data set
cmic = 0.001    # convert mic. positions (0.001 converts from mm to m)
firstLineSplit = linesNoBreak[0].split(",")
xs = float(inArr[0,0])*cmic
ys = float(inArr[0,1])*cmic
x1 = float(inArr[0,2])*cmic
y1 = float(inArr[0,3])*cmic
x2 = float(inArr[0,4])*cmic
y2 = float(inArr[0,5])*cmic
x3 = float(inArr[0,6])*cmic
y3 = float(inArr[0,7])*cmic

'''
intersection, cluster, contour_points, and clusterlists functions
adapted from:

https://stackoverflow.com/questions/17416268/how-to-find-
all-the-intersection-points-between-two-contour-set-in-an-efficient
'''

def intersection(points1, points2, eps):
    tree = spatial.KDTree(points1)
    distances, indices = tree.query(points2, k=1, distance_upper_bound=eps)
    intPts = tree.data[indices[np.isfinite(distances)]]
    return intPts


def cluster(points, cluster_size):
    dists = dist.pdist(points, metric='sqeuclidean')
    linkage_matrix = hier.linkage(dists, 'average')
    groups = hier.fcluster(linkage_matrix, cluster_size, criterion='distance')
    return np.array([points[cluster].mean(axis=0)
                     for cluster in clusterlists(groups)])


def contour_points(contour, steps=1):
    return np.row_stack([path.interpolated(steps).vertices
                         for linecol in contour.collections
                         for path in linecol.get_paths()])

def clusterlists(T):
    '''
    http://stackoverflow.com/a/2913071/190597 (denis)
    T = [2, 1, 1, 1, 2, 2, 2, 2, 2, 1]
    Returns [[0, 4, 5, 6, 7, 8], [1, 2, 3, 9]]
    '''
    groups = collections.defaultdict(list)
    for i, elt in enumerate(T):
        groups[elt].append(i)
    return sorted(groups.values(), key=len, reverse=True)   
    
# set up coarse grid for plot:
# create x and y axes of plot (equivalent to meshgrid)
xcoarse = np.arange( xBoundL, xBoundR, xScalCoarse )
ycoarse = np.arange( yBoundL, yBoundR, yScalCoarse )

# tile each axis (because x-axis needs to be same for each row
# of y-axis and vice versa)
Xcoarse = np.tile( xcoarse, [ycoarse.size,1])
Ycoarse = np.tile( ycoarse, [xcoarse.size,1])
Ycoarse = np.rot90(Ycoarse)	# rotate y-axis

intPtsAll = np.array([[0,0]])

# find intersection points for scatter
i = 0
while i < nData:
    ctime = 0.000001    # convert times from microseconds to seconds
    vs = 343            # convert times (seconds) to distance (meters)
    di12 = (inArr[i,8]-inArr[i,9])*ctime*vs
    di13 = (inArr[i,8]-inArr[i,10])*ctime*vs
    di23 = (inArr[i,9]-inArr[i,10])*ctime*vs
    ## coordinates of source and microphones marker='o', markersize=8, color="black"
    plt.plot(xs, ys, marker='.', markersize=8, color="black")   #sound source
    plt.plot(x1, y1, marker='s', markersize=8, color="red")     #mic 1
    plt.plot(x2, y2, marker='s', markersize=8, color="yellow")  #mic 2
    plt.plot(x3, y3, marker='s', markersize=8, color="blue")    #mic 3
    
    contour12 = plt.contour(
        Xcoarse, Ycoarse,
        curve(Xcoarse, Ycoarse, x1, y1, x2, y2),
        [di12],
        colors="orange"
    )
    contour13 = plt.contour(
        Xcoarse, Ycoarse,
        curve(Xcoarse, Ycoarse, x1, y1, x3, y3),
        [di13],  
        colors="purple"
    )
    contour23 = plt.contour(
        Xcoarse, Ycoarse,
        curve(Xcoarse, Ycoarse, x2, y2, x3, y3),
        [di23],
        colors="green"
    )
    
    # every intersection point must be within eps of a point on the other
    # contour path
    eps = .01
    
    # cluster together intersection points so that the original points in each flat
    # cluster have a cophenetic_distance < cluster_size
    cluster_size = 1
    
    pts12 = contour_points(contour12)
    pts13 = contour_points(contour13)
    pts23 = contour_points(contour23)
    
    intPts1 = intersection(pts12, pts13, eps)
    intPts2 = intersection(pts12, pts23, eps)
    intPts3 = intersection(pts13, pts23, eps)
    
    addPts = False
    
    if len(intPts1) > 0:
        if len(intPts2) > 0:
            if len(intPts3) > 0:           
                intPts = np.concatenate((intPts1, intPts2, intPts3),axis=0)
            else:
                intPts = np.concatenate((intPts1, intPts2),axis=0)
            intPts = cluster(intPts, cluster_size)
            addPts = True
        elif len(intPts3) > 0:
            intPts = np.concatenate((intPts1, intPts3),axis=0)
            intPts = cluster(intPts, cluster_size)
            addPts = True
    
    if addPts == True:
        intPtsAll = np.concatenate((intPtsAll, intPts),axis=0)
        # plot intersection points
        plt.scatter(intPts[:,0], intPts[:,1], s=20)
    
    plt.show()
        
    i += 1

intPtsAll = np.delete(intPtsAll, 0, 0)
# set up coarse grid for plot:
# create x and y axes of plot (equivalent to meshgrid)
xfine = np.arange( xBoundL, xBoundR, xScalFine )
yfine = np.arange( yBoundL, yBoundR, yScalFine )

# tile each axis (because x-axis needs to be same for each row
# of y-axis and vice versa)
Xfine = np.tile( xfine, [yfine.size,1])
Yfine = np.tile( yfine, [xfine.size,1])
Yfine = np.rot90(Yfine)	# rotate y-axis

# set uncertainty (standard deviation) of mic delays, in seconds:
sigmaSec = 0.00049204231804344704
# convert to meters:
sigma = sigmaSec * 343

d12 = getDelay(xs, ys, x1, y1, x2, y2)
d13 = getDelay(xs, ys, x1, y1, x3, y3)
d23 = getDelay(xs, ys, x2, y2, x3, y3)

#d12 = getDelay(source[0],source[1],xMic[0],yMic[0],xMic[1],yMic[1])
#d13 = getDelay(source[0],source[1],xMic[0],yMic[0],xMic[2],yMic[2])
#d23 = getDelay(source[0],source[1],xMic[1],yMic[1],xMic[2],yMic[2])

# make array of delays (expected delay at each pixel)
D12 = getDelay(Xfine, Yfine, x1, y1, x2, y2)
D13 = getDelay(Xfine, Yfine, x1, y1, x3, y3)
D23 = getDelay(Xfine, Yfine, x2, y2, x3, y3)

# plot microphones, sound source, and scatter points
def doPlot(title=False,titleName='',save=False,nameOut='file'):
    plt.plot(xs, ys, marker='.', markersize=sourceSize, color="black",label='known location of source',linestyle='None')   #sound source
    plt.plot(x1, y1, marker='s', markersize=micSize, color="red",label='mic. 1',linestyle='None')     #mic 1
    plt.plot(x2, y2, marker='s', markersize=micSize, color="yellow",label='mic. 2',linestyle='None')  #mic 2
    plt.plot(x3, y3, marker='s', markersize=micSize, color="blue",label='mic. 3',linestyle='None')    #mic 3
    plt.scatter(intPtsAll[:,0], intPtsAll[:,1], s=pointSize, color="white",label='machine outputs')
    plt.legend(prop={"size":5})
    
    # label axes
    plt.xlabel('x-position (m)')
    plt.ylabel('y-position (m)')
    
    # save and show figure
    if save == True:
        plt.savefig(nameOut,dpi=1200)
    plt.show()

fileNameOut = fileNameIn

plotIndividual = True

if plotIndividual == True:
    plt.imshow(gauss(D12, sigma, d12),extent=[xBoundL,xBoundR,yBoundL,yBoundR])
    c12 = plt.contour(Xfine,Yfine,D12,[d12],colors="orange",linestyles='solid')   
    doPlot(nameOut=fileNameOut+' Pair 1-2',save=plotIndividual)
    
    plt.imshow(gauss(D13, sigma, d13),extent=[xBoundL,xBoundR,yBoundL,yBoundR])
    c13 = plt.contour(Xfine,Yfine,D13,[d13],colors="magenta",linestyles='solid') 
    doPlot(nameOut=fileNameOut+' Pair 1-3',save=plotIndividual)
    
    plt.imshow(gauss(D23, sigma, d23),extent=[xBoundL,xBoundR,yBoundL,yBoundR])
    c23 = plt.contour(Xfine,Yfine,D23,[d23],colors="green",linestyles='solid')
    doPlot(nameOut=fileNameOut+' Pair 2-3',save=plotIndividual)

# final plot
heatmap = gauss(D12, sigma, d12)*gauss(D13, sigma, d13)*gauss(D23, sigma, d23)
plt.imshow(heatmap,extent=[xBoundL,xBoundR,yBoundL,yBoundR])
# make contours
c12 = plt.contour(Xfine,Yfine,D12,[d12],colors="orange",linestyles='solid')                        # mics 1, 2
c13 = plt.contour(Xfine,Yfine,D13,[d13],colors="magenta",linestyles='solid')                        # mics 1, 3
c23 = plt.contour(Xfine,Yfine,D23,[d23],colors="green",linestyles='solid')                        # mics 2, 3

if labelContours == True:
    plt.clabel(c12, fontsize=10, inline=1,fmt = 'solution 1-2')
    plt.clabel(c13, fontsize=10, inline=1,fmt = 'solution 1-3')
    plt.clabel(c23, fontsize=10, inline=1,fmt = 'solution 2-3')

doPlot(save=True,nameOut=fileNameOut+' Product')