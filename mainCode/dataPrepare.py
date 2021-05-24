from loadData import loadData
import numpy as np
import scipy.io as sio

trnDataAll = []
trnLabelAll = []
tstDataAll = []
tstLabelAll = []
for i in range(5):
    # if i < 5:
    path = 'D:/BCI/Data/BCI2019/supervised/mi-t.tar/0' + str(i+1) + '/B0' + str(i+1) + '01T.mat'
    # else:
    #     path = 'D:/BCI/Data/BCI2019/supervised/mi-t.tar/0' + str(i+1) + '/T0' + str(i+1) + '01T.mat'
    dataset = loadData(path)
    data, label = dataset.signalCut()
    data = dataset.signalProcess(data[:, 0:59], dsFlag=True)
    data = dataset.signalFilter(data)

    num = int(len(data)*5/6)

    trnData = data[:num]
    tstData = data[num:]
    trnLabel = label[:num]
    tstLabel = label[num:]

    if i == 0:
        trnDataAll = trnData
        tstDataAll = tstData
    else:
        trnDataAll = np.concatenate((trnDataAll, trnData), axis=0)
        tstDataAll = np.concatenate((tstDataAll, tstData), axis=0)
    trnLabelAll.append(trnLabel)
    tstLabelAll.append(tstLabel)

trnLabelAll = np.hstack(trnLabelAll)
tstLabelAll = np.hstack(tstLabelAll)

sio.savemat('data/allData.mat', {'trnData': trnDataAll,'trnLabel':trnLabelAll,'tstData':tstDataAll,'tstLabel':tstLabelAll})


