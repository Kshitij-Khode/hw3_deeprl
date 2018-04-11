import matplotlib.pyplot as plt
import sys, time

epTrain = []
trainRew = []
epTest = []
avgRew = []
stdDev = []

for line in open (sys.argv[1], 'r'):
    for word in line.split():
        if 'test' in word:
            epTest.append(int(word.split('-')[1].split(':')[0]))
        if 'meanRews' in word:
            avgRew.append(float(word.split(':')[1].split(',')[0]))
        if 'stdRews' in word:
            stdDev.append(float(word.split(':')[1].split(',')[0]))
        if 'ep' in word:
            epTrain.append(int(word.split(':')[1].split(',')[0]))
        if 'cRew' in word:
            trainRew.append(float(word.split(':')[1]))
epTest  = epTest[:len(stdDev)]
avgRew  = avgRew[:len(stdDev)]
epTrain = epTrain[:len(trainRew)]

epTest.pop()
avgRew.pop()
stdDev.pop()

plt.plot(epTrain, trainRew, color='Red')
plt.errorbar(epTest, avgRew, stdDev, color='Yellow')
plt.gca().axhline(max(avgRew), label='%s'%max(avgRew))
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles, labels)
plt.show()
