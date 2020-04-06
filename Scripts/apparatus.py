'''
Authors: Mauricio Soroco, Luke Mantle
2019

Created for T2 project

Reads input from Arduino, saves raw data to file, and produces plot of
microphone positions, manually entered location of sound source (for reference)
and paramentric functions for each pair of microphones to visualize that data
point.
'''

import serial #pySerial is the name of the tool, it might need to be installed
import matplotlib.pyplot as plt
from numpy import arange, meshgrid, sqrt

arduino = serial.Serial('com3', baudrate = 9600, timeout=1)
#go to tools and find the port (choose the highest number) might be com5
#(serial communication port, baudrate=9600, timeout=1)
#timeout is the seconds we should wait before accepting the serial information

i = 0
inpStr = []

while(i < 7): #is this right?

	arduinoData = arduino.readline().decode('ascii') #looks at info coming in and reads untill the end of line
		#the ascii is supposed to remove the prefix "b'" and the suffixes \r\n' that would appear in the python data when printed (may be necessary)
		#found the next part online
	if arduinoData:
		print(arduinoData)
		print(type(arduinoData))
		inpStr.append(arduinoData[:-2]) #append each line of output from Arduino and remove the line break

	i += 1
	
# convert the 3 time outputs to integer and send them to a new list
inpInt = [int(inpStr[1]),int(inpStr[2]),int(inpStr[3])]
print("Time of arrival (microseconds) at mic. 1: " + str(inpInt[0]))
print("Time of arrival (microseconds) at mic. 2: " + str(inpInt[1]))
print("Time of arrival (microseconds) at mic. 3: " + str(inpInt[2]))
print("done") #so we know it's finished

#coordinates for sound source (change for each trial):
xs = 0
ys = 0
#coordinates for mic 1 (left; yellow, green, blue)
x1 = 0
y1 = -415
#coordinates for mic 2 (middle; purple, white, grey) in mm (see desmoms):
x2=455
y2=420
#coordinates for mic 3 (right; black/red ,brown, orange) in mm (see desdads):
x3=-455
y3=420


inpInt0 = inpInt[0]
inpInt1 = inpInt[1]
inpInt2 = inpInt[2]

f = open("dataFeb18", "a")
f.write("\n"+str(xs)+","+str(ys)+","+str(x1)+","+str(y1)+","+str(x2)+","+str(y2)+","+str(x3)+","+str(y3)+","+str(inpInt0)+","+str(inpInt1)+","+str(inpInt2))
f.close()

#differences in distance in mm:
d12=(inpInt[0] - inpInt[1])/1000*343

d13=(inpInt[0] - inpInt[2])/1000*343

d23=(inpInt[1] - inpInt[2])/1000*343

#Plot window setup:
delta = 1
x, y = meshgrid(
    arange(-1300, 1200, delta), #x-range
    arange(-1000, 1200, delta)   #y-range
)

# coordinates of source and microphones
plt.plot(xs, ys, marker='o', markersize=8, color="black")   #sound source
plt.plot(x1, y1, marker='s', markersize=8, color="red")     #mic 1
plt.plot(x2, y2, marker='s', markersize=8, color="yellow")  #mic 2
plt.plot(x3, y3, marker='s', markersize=8, color="blue")    #mic 3
plt.contour(
    x, y,
    sqrt((x - x2) ** 2 + (y - y2) ** 2) - sqrt( (x - x1) ** 2 + (y - y1) ** 2 ) + d12,
    [0],
    colors="orange"
)
plt.contour(
    x, y,
    sqrt((x - x3) ** 2 + (y - y3) ** 2) - sqrt( (x - x1) ** 2 + (y - y1) ** 2 ) + d13,
    [0],  
    colors="purple"
)
plt.contour(
    x, y,
    sqrt((x - x3) ** 2 + (y - y3) ** 2) - sqrt( ( x- x2) ** 2 + (y - y2) ** 2 ) + d23,
    [0],
    colors="green"
)

# print distances between mics in mm:
print("Distance (mm) between mic. 1 and mic. 2 :" + str(d12))
print("Distance (mm) between mic. 1 and mic. 3: " + str(d13))
print("Distance (mm) between mic. 2 and mic. 3: " + str(d23))

plt.grid()
plt.show()
plt.figure(1) #initialize plot