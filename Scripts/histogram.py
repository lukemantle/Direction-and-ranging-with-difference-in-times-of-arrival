'''
Author: Luke Mantle
2020

Created for T2 project

Creates histograms of distributions of TDOAs between each pair of microhphones
for each sound source location. Fits a normal distribution to these and saves
the standard devations to a file.

Takes mic. positions from first data point to be mic.
positions for whole data set
'''

import numpy as np
import matplotlib.pyplot as plt
import statistics as stat

# read data
fileName = "source 5"
lines = open(fileName, "r").readlines()
linesNoBreak = []
for i in lines:
    if i[-1:] == "\n":
        linesNoBreak.append(i[:-1])     # remove \n from each line
    else:
        linesNoBreak.append(i)

#x = []  # create list to hold all x-pos. from data
#y = []  # create list to hold all y-pos. from data

# get mic. positions
firstLineSplit = linesNoBreak[0].split(",")
xmic1 = firstLineSplit[2]
ymic1 = firstLineSplit[3]
xmic2 = firstLineSplit[4]
ymic2 = firstLineSplit[5]
xmic3 = firstLineSplit[6]
ymic3 = firstLineSplit[7]

# get delays
d12 = []
d13 = []
d23 = []
for i in linesNoBreak:
    line = i.split(",")
    t1 = float(line[8])
    t2 = float(line[9])
    t3 = float(line[10])
    d12.append(t1-t2)
    d13.append(t1-t3)
    d23.append(t2-t3)

sdev12 = stat.stdev(d12)
sdev13 = stat.stdev(d13)
sdev23 = stat.stdev(d23)

def gauss(x, sigma, mu):
    y = (1 / (sigma * np.sqrt(2*np.pi))) * np.exp(-1/2 * ((x - mu) / sigma) ** 2)
    return y

def makeaxis(d, n):
    return np.linspace(min(d), max(d), n)

n = 100 # number of samples along x-axis for gaussian distrbn. graphs
nBins = 20  # number of bins for histograms

xAx12 = makeaxis(d12, n)
xAx13 = makeaxis(d13, n)
xAx23 = makeaxis(d23, n)

def doPlot(name=''):
    plt.xlabel('delay (microseconds)')
    plt.ylabel('probability density')
    plt.title(label=name+' for '+fileName)
    plt.savefig(fileName+' '+name,dpi=1200)

plt.hist(d12, bins=nBins, density = True)
plt.plot(xAx12, gauss(xAx12, stat.stdev(d12), np.mean(d12)))
doPlot('delays 1-2')
plt.show()

plt.hist(d13, bins=nBins, density = True)
plt.plot(xAx13, gauss(xAx13, stat.stdev(d13), np.mean(d13)))
doPlot('delays 1-3')
plt.show()

plt.hist(d23, bins=nBins, density = True)
plt.plot(xAx23, gauss(xAx23, stat.stdev(d23), np.mean(d23)))
doPlot('delays 2-3')
plt.show()

# save standard devations to file
if False:
    f = open("distr 1", "a")
    f.write("\n" + fileName + "," + str(stat.stdev(d12)) + "," + str(stat.stdev(d13)) + "," + str(stat.stdev(d23)))
    f.close()
