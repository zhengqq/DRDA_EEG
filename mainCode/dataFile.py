import tensorflow as tf
import scipy.io as sio
import tensorflow.contrib.slim as slim
import numpy as np
from tensorflow import keras
from sklearn import preprocessing
from loadData import loadData
import random
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit

def build_data_spectrogram(path):

    data = sio.loadmat(path)
    trnData = data['dataTrain']
    trnLabel = data['labelTrain']
    tstData = data['dataTest']
    tstLabel = data['labelTest']

    trnData = np.expand_dims(trnData, axis=3)
    trnData = np.transpose(trnData, [2,0,1,3])
    trnLabel = trnLabel.T[0]

    tstData = np.expand_dims(tstData, axis=3)
    tstData = np.transpose(tstData, [2,0,1,3])
    tstLabel = tstLabel.T[0]

    return trnData, trnLabel, tstData, tstLabel












    # random.seed(2019)

    rand_flag = 0

    if rand_flag:
        ## random pick train and test sample
        trailList = list(range(int(len(trnLabel)/5)))
        random.shuffle(trailList)
        tstList = trailList[:20]
        trnList = trailList[20:]
        trnIdx, tstIdx = [], []

        for i in tstList:
            tstIdx.extend(list(range(i*5, i*5+5)))
        for i in trnList:
            trnIdx.extend(list(range(i*5, i*5+5)))

        trnData1 = trnData[trnIdx,:,:,:]
        trnLabel1 = trnLabel[trnIdx]
        tstData1 = trnData[tstIdx,:,:,:]
        tstLabel1 = trnLabel[tstIdx]
    else:
        samList = list(range(len(trnLabel)))
        num = int(len(trnLabel) * 5 / 6)
        trnList = list(range(num))
        # random.shuffle(trnList)
        samList[:len(trnList)] = trnList

        trnData = trnData[samList]
        trnLabel = trnLabel[samList]

        trnData1 = trnData[0:num,:,:,:]
        trnLabel1 = trnLabel[0:num]
        tstData1 = trnData[num:,:,:,:]
        tstLabel1 = trnLabel[num:]

        # return trnData[:,:,:,[28,25,29]], trnLabel, tstData[:,:,:,[28,25,29]], tstLabel
    return trnData1[:,:,:,:], trnLabel1, tstData1[:,:,:,:], tstLabel1




def build_data_tdp(path):

    data = sio.loadmat(path)
    trnData = data['X']
    trnLabel = data['y']
    trnLabel[trnLabel == -1] = 0

    tstData = data['X_test']
    tstLabel = data['y_test']
    tstLabel[tstLabel == -1] = 0

    trnData = np.transpose(trnData, [2, 0, 1])
    trnData = trnData[:,:,:,np.newaxis]

    tstData = np.transpose(tstData, [2, 0, 1])
    tstData = tstData[:,:,:,np.newaxis]


    return trnData, trnLabel, tstData, tstLabel




def data_7band_ds_2s(i):
    if i <6:
        path = 'D:/BCI/Data/BCI2019/supervised/mi-t.tar/0' + str(i) + '/B0' + str(i) + '01T.mat'
    else:
        path = 'D:/BCI/Data/BCI2019/supervised/mi-t.tar/0' + str(i) + '/T0' + str(i) + '01T.mat'



    dataset = loadData(path)
    data, label = dataset.signalCut()
    data = dataset.signalProcess(data, dsFlag=True)
    data = dataset.signalFilter(data)
    # data = dataset.signalNorm(data)
    # data = (data - data.min(axis=2, keepdims=True)) / (data.max(axis=2, keepdims=True) - data.min(axis=2, keepdims=True))

    label = label - 1

    import random
    # random.seed(2019)

    rand_flag = 0

    if rand_flag:
        ## random pick train and test sample
        num = int(len(data))
        trailList = list(range(int(len(data))))
        random.shuffle(trailList)
        trnIdx = trailList[0:num]
        tstIdx = trailList[num:]

        trnData1 = data[trnIdx, :, :, :]
        trnLabel1 = label[trnIdx]
        tstData1 = data[tstIdx, :, :, :]
        tstLabel1 = label[tstIdx]
    else:
        samList = list(range(data.shape[0]))
        num = int(len(data) * 5 / 6)
        trnList = list(range(num))
        random.shuffle(trnList)
        samList[:len(trnList)] = trnList

        trnData = data[samList]
        trnLabel = label[samList]

        trnData1 = trnData[0:num, :, :, :]
        trnLabel1 = trnLabel[0:num]
        tstData1 = trnData[num:, :, :, :]
        tstLabel1 = trnLabel[num:]

    # return trnData1[:,:59,:,:], trnLabel1, tstData1[:,:59,:,:], tstLabel1
    return trnData1[:,:,:,:], trnLabel1, tstData1[:,:,:,:], tstLabel1
    # return trnData1[:,[28,25,29],:,:], trnLabel1, tstData1[:,[28,25,29],:,:], tstLabel1










def dataB(path):
    # if i < 8:
    #     path = './data/dataB/0' + str(i) + '/B0' + str(i) + '01T.mat'
    # else:
    #     path = './data/dataB/0' + str(i) + '/T0' + str(i) + '01T.mat'
    reproduce = True


    dataset = loadData(path)
    data, label = dataset.signalCut([])
    data_f = dataset.signalFilter(data)
    data = dataset.signalProcess(data_f, dsFlag=True)

    # data = dataset.signalNorm(data)
    # data = (data - data.min(axis=2, keepdims=True)) / (data.max(axis=2, keepdims=True) - data.min(axis=2, keepdims=True))

    if len(data.shape) != 4:
        data = np.expand_dims(data, axis=-1)

    label = label - 1

    first_list = list(np.where(label==0)[0])
    second_list = list(np.where(label==1)[0])
    third_list = list(np.where(label==2)[0])

    if reproduce:
        random.seed(2019)
    random.shuffle(first_list)
    random.shuffle(second_list)
    random.shuffle(third_list)

    num = int(len(data)*5/6 / len(np.unique(label))) #if len(data) == 120 else 150

    trnList = first_list[0:num] + second_list[0:num] + third_list[0:num]
    tstList = first_list[num:] + second_list[num:] + third_list[num:]

    if reproduce:
        random.seed(2019)
    random.shuffle(trnList)
    random.shuffle(tstList)

    trnData = data[trnList]
    trnLabel = label[trnList]
    tstData = data[tstList]
    tstLabel = label[tstList]



    # if reproduce:
    #     seed = 2019
    # else:
    #     seed = None
    #
    # spt = StratifiedShuffleSplit(n_splits=1, test_size=1 / 6, random_state=seed)
    # spt.get_n_splits(data, label)
    #
    # for trnList, tstList in spt.split(data, label):
    #     trnData, trnLabel = data[trnList], data[trnList]
    #     tstData, tstLabel = label[tstList], label[tstList]


    return trnData, trnLabel, tstData, tstLabel



def dataB_valSet(path, aug=False, reproduce=True, loadN='1'):
    # reproduce = True
    # aug = True

    dataset = loadData(path, s=1.5, e=5.5, w=1, sl=7)
    data, label = dataset.signalCut(idx=[])


    # select training idx
    seed = 2019 if reproduce else None
    # spt = StratifiedShuffleSplit(n_splits=5, test_size=24 / 120, random_state=seed)
    spt = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
    # spt.get_n_splits(data, label)

    aa =0
    for ltrn, ltst in spt.split(data, label):
        aa +=1
        if aa == int(loadN):
            trnList, tstList = ltrn, ltst
            tstData, tstLabel = data[ltst], label[ltst]
            # trnData, trnLabel = data[ltrn], label[ltrn]
            # sio.savemat('fold' + str(aa) +'.mat', {'trnList':trnList,'tstList':tstList,'tstData':tstData, 'tstLabel':tstLabel,
            #                                        'trnData':trnData,'trnLabel':trnLabel})
        else:
            continue


    if aug is True:
        datasetTrain = loadData(path, s=1.5, e=5.5, w=1, sl=7)
        trnData, trnLabel = datasetTrain.signalCut(idx=trnList, random_pick=True)
    elif aug is False:
        datasetTrain = loadData(path, s=1.5, e=5.5, w=1, sl=7)
        trnData, trnLabel = datasetTrain.signalCut(idx=trnList)

    trnData = dataset.signalFilter(trnData)
    trnData = dataset.signalProcess(trnData, dsFlag=True)

    tstData = dataset.signalFilter(tstData)
    tstData = dataset.signalProcess(tstData, dsFlag=True)


    if len(trnData.shape) != 4:
        trnData = np.expand_dims(trnData, axis=-1)

    if len(tstData.shape) != 4:
        tstData = np.expand_dims(tstData, axis=-1)

    trnLabel = trnLabel - 1
    tstLabel = tstLabel - 1

    if reproduce:
        random.seed(2019)

    tem = list(range(len(trnData)))
    random.shuffle(tem)

    trnData = trnData[tem]
    trnLabel = trnLabel[tem]


    return trnData, trnLabel, tstData, tstLabel


def data_multitask(aug=False, reproduce=True, dataset='B', loadN='1'):

    trnData, trnLabel, tstData, tstLabel = [], [], [], []
    trnPerLabel, tstPerLabel = [], []

    # trnList = [1,2,3,5,6]
    # for dataPath in trnList:
    for dataPath in range(7):

        dataPath = dataPath + 1
        if dataset == 'A':
            if dataPath < 6:
                path = './data/dataA/0' + str(dataPath) + '/B0' + str(dataPath) + '01T.mat'
            else:
                path = './data/dataA/0' + str(dataPath) + '/T0' + str(dataPath) + '01T.mat'
        elif dataset == 'B':
            if dataPath < 8:
                path = './data/dataB/0' + str(dataPath) + '/B0' + str(dataPath) + '01T.mat'
            else:
                path = './data/dataB/0' + str(dataPath) + '/T0' + str(dataPath) + '01T.mat'

        trnData1, trnLabel1, tstData1, tstLabel1 = dataB_valSet(path, aug=aug, reproduce=reproduce, loadN=loadN)

        trnData.append(trnData1)
        trnLabel.append(trnLabel1)
        tstData.append(tstData1)
        tstLabel.append(tstLabel1)
        trnPerLabel.append(np.ones(len(trnData1))*(dataPath-1))
        tstPerLabel.append(np.ones(len(tstData1))*(dataPath-1))

    trnData = np.concatenate(trnData, axis=0)
    trnLabel = np.hstack(trnLabel)

    tstData = np.concatenate(tstData, axis=0)
    tstLabel = np.hstack(tstLabel)

    trnPerLabel = np.hstack(trnPerLabel)
    tstPerLabel = np.hstack(tstPerLabel)

    return trnData, trnLabel, tstData, tstLabel, trnPerLabel, tstPerLabel



def data_multitask_da(aug=False, reproduce=True, dataset='B', loadN='1', subject=1):

    trnData, trnLabel, tstData, tstLabel = [], [], [], []
    trnPerLabel, tstPerLabel = [], []

    allSubNum = 2

    sub = subject - 1
    SourceList = list(range(allSubNum))
    SourceList = [7,8]
    SourceList.pop(sub-7)

    for dataPath in SourceList:
    # for dataPath in range(7):

        dataPath = dataPath + 1
        if dataset == 'A':
            if dataPath < 6:
                path = './data/dataA/0' + str(dataPath) + '/B0' + str(dataPath) + '01T.mat'
            else:
                path = './data/dataA/0' + str(dataPath) + '/T0' + str(dataPath) + '01T.mat'
        elif dataset == 'B':
            if dataPath < 8:
                path = './data/dataB/0' + str(dataPath) + '/B0' + str(dataPath) + '01T.mat'
            else:
                path = './data/dataB/0' + str(dataPath) + '/T0' + str(dataPath) + '01T.mat'

        trnData1, trnLabel1, tstData1, tstLabel1 = dataB_valSet(path, aug=aug, reproduce=reproduce, loadN=loadN)

        trnData.append(trnData1)
        trnLabel.append(trnLabel1)
        tstData.append(tstData1)
        tstLabel.append(tstLabel1)
        trnPerLabel.append(np.ones(len(trnData1))*(dataPath-1))
        tstPerLabel.append(np.ones(len(tstData1))*(dataPath-1))

    trnData = np.concatenate(trnData, axis=0)
    trnLabel = np.hstack(trnLabel)

    tstData = np.concatenate(tstData, axis=0)
    tstLabel = np.hstack(tstLabel)

    trnPerLabel = np.hstack(trnPerLabel)
    tstPerLabel = np.hstack(tstPerLabel)

    sourceData = np.concatenate((trnData, tstData), axis=0)
    sourceLabel = np.concatenate((trnLabel, tstLabel), axis=0)
    sourcePerLabel = np.concatenate((trnPerLabel, tstPerLabel), axis=0)

    targetList = list(set(range(allSubNum)).difference(set(SourceList)))
    targetList = list(set(range(7,9)).difference(set(SourceList)))
    for dataPath in targetList:
        dataPath = dataPath + 1
        if dataset == 'A':
            if dataPath < 6:
                path = './data/dataA/0' + str(dataPath) + '/B0' + str(dataPath) + '01T.mat'
            else:
                path = './data/dataA/0' + str(dataPath) + '/T0' + str(dataPath) + '01T.mat'
        elif dataset == 'B':
            if dataPath < 8:
                path = './data/dataB/0' + str(dataPath) + '/B0' + str(dataPath) + '01T.mat'
            else:
                path = './data/dataB/0' + str(dataPath) + '/T0' + str(dataPath) + '01T.mat'

        targetData, targetLabel, targetTstData, targetTstLabel = dataB_valSet(path, aug=aug, reproduce=reproduce, loadN=loadN)




    return sourceData, sourceLabel, targetData, targetLabel, targetTstData, targetTstLabel



def trn_tst_split(data, label, loadN='1', reproduce=True):

    seed = 2019 if reproduce else None
    # spt = StratifiedShuffleSplit(n_splits=5, test_size=24 / 120, random_state=seed)
    spt = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
    # spt.get_n_splits(data, label)

    aa = 0
    for ltrn, ltst in spt.split(data, label):
        aa +=1
        if aa == int(loadN):
            # trnList, tstList = ltrn, ltst
            tstData, tstLabel = data[ltst], label[ltst]
            trnData, trnLabel = data[ltrn], label[ltrn]
            # sio.savemat('fold' + str(aa) +'.mat', {'trnList':trnList,'tstList':tstList,'tstData':tstData, 'tstLabel':tstLabel,
            #                                        'trnData':trnData,'trnLabel':trnLabel})
        else:
            continue

    return trnData, trnLabel, tstData, tstLabel


def bciiv2a_multitask_da(aug=False, reproduce=True, dataset='B', loadN='1', subject=1, data_len='data_0-4'):
    # data_len = 'data_1-4'

    trnData, trnLabel, tstData, tstLabel = [], [], [], []
    trnPerLabel, tstPerLabel = [], []

    allSubNum = 9

    sub = subject - 1
    SourceList = list(range(allSubNum))
    SourceList.pop(sub)

    for dataPath in SourceList:
        dataPath = dataPath + 1
        raw = sio.loadmat('./data/BCI_IV_2a/' + data_len + '/' + str(dataPath) + '.mat')
        trnData1, trnLabel1, tstData1, tstLabel1 = raw['trnData'], raw['trnLabel'][0], raw['tstData'], raw['tstLabel'][
            0]

        trnData.append(trnData1)
        trnLabel.append(trnLabel1)
        tstData.append(tstData1)
        tstLabel.append(tstLabel1)
        trnPerLabel.append(np.ones(len(trnData1)) * (dataPath - 1))
        tstPerLabel.append(np.ones(len(tstData1)) * (dataPath - 1))

    trnData = np.concatenate(trnData, axis=0)
    trnLabel = np.hstack(trnLabel)

    tstData = np.concatenate(tstData, axis=0)
    tstLabel = np.hstack(tstLabel)

    trnPerLabel = np.hstack(trnPerLabel)
    tstPerLabel = np.hstack(tstPerLabel)

    sourceData = np.concatenate((trnData, tstData), axis=0)
    sourceLabel = np.concatenate((trnLabel, tstLabel), axis=0)
    sourcePerLabel = np.concatenate((trnPerLabel, tstPerLabel), axis=0)

    targetList = list(set(range(allSubNum)).difference(set(SourceList)))
    # targetList = list(set(range(7,9)).difference(set(SourceList)))
    for dataPath in targetList:
        dataPath = dataPath + 1
        raw = sio.loadmat('./data/BCI_IV_2a/' + data_len + '/' + str(dataPath) + '.mat')
        targetData, targetLabel, targetTstData, targetTstLabel = raw['trnData'], raw['trnLabel'][0], raw['tstData'], \
                                                                 raw['tstLabel'][0]

    # targetTrnData, targetTrnLabel, targetValData, TargetValLabel = trn_tst_split(targetData, targetLabel, loadN)

    return sourceData, sourceLabel, targetData, targetLabel, targetTstData, targetTstLabel



def bciiv2a_multitask(aug=False, reproduce=True, dataset='B', loadN='1', subject=1, data_len='data_0-4'):
    # data_len = 'data_1-4'

    trnData, trnLabel, tstData, tstLabel = [], [], [], []
    trnPerLabel, tstPerLabel = [], []

    allSubNum = 9

    sub = subject - 1
    SourceList = list(range(allSubNum))
    SourceList.pop(sub)

    for dataPath in SourceList:
        dataPath = dataPath + 1
        raw = sio.loadmat('./data/BCI_IV_2a/' + data_len + '/' + str(dataPath) + '.mat')
        trnData1, trnLabel1, tstData1, tstLabel1 = raw['trnData'], raw['trnLabel'][0], raw['tstData'], raw['tstLabel'][
            0]

        trnData.append(trnData1)
        trnLabel.append(trnLabel1)
        tstData.append(tstData1)
        tstLabel.append(tstLabel1)
        trnPerLabel.append(np.ones(len(trnData1)) * (dataPath - 1))
        tstPerLabel.append(np.ones(len(tstData1)) * (dataPath - 1))

    trnData = np.concatenate(trnData, axis=0)
    trnLabel = np.hstack(trnLabel)

    tstData = np.concatenate(tstData, axis=0)
    tstLabel = np.hstack(tstLabel)

    trnPerLabel = np.hstack(trnPerLabel)
    tstPerLabel = np.hstack(tstPerLabel)

    sourceData = np.concatenate((trnData, tstData), axis=0)
    sourceLabel = np.concatenate((trnLabel, tstLabel), axis=0)
    sourcePerLabel = np.concatenate((trnPerLabel, tstPerLabel), axis=0)

    targetList = list(set(range(allSubNum)).difference(set(SourceList)))
    # targetList = list(set(range(7,9)).difference(set(SourceList)))
    for dataPath in targetList:
        dataPath = dataPath + 1
        raw = sio.loadmat('./data/BCI_IV_2a/' + data_len + '/' + str(dataPath) + '.mat')
        targetData, targetLabel, targetTstData, targetTstLabel = raw['trnData'], raw['trnLabel'][0], raw['tstData'], \
                                                                 raw['tstLabel'][0]
        targetPerLabel =  np.ones(len(targetData)) * (dataPath - 1)
        targetTstPerLabel = np.ones(len(targetTstData)) * (dataPath - 1)

    sourceData = np.concatenate((sourceData, targetData), axis=0)
    sourceLabel = np.concatenate((sourceLabel, targetLabel), axis=0)
    sourcePerLabel = np.concatenate((sourcePerLabel, targetPerLabel), axis=0)


    # targetTrnData, targetTrnLabel, targetValData, TargetValLabel = trn_tst_split(targetData, targetLabel, loadN)

    return sourceData, sourceLabel, sourcePerLabel, targetTstData, targetTstLabel, targetTstPerLabel



def bciiv2a_all(aug=False, reproduce=True, dataset='B', loadN='1', subject=1, data_len='data_0-4'):
    # data_len = 'data_1-4'

    trnData, trnLabel, tstData, tstLabel = [], [], [], []
    trnPerLabel, tstPerLabel = [], []

    allSubNum = 9

    sub = subject - 1
    SourceList = list(range(allSubNum))
    SourceList.pop(sub)

    for dataPath in SourceList:
        dataPath = dataPath + 1
        raw = sio.loadmat('./data/BCI_IV_2a/' + data_len + '/' + str(dataPath) + '.mat')
        trnData1, trnLabel1, tstData1, tstLabel1 = raw['trnData'], raw['trnLabel'][0], raw['tstData'], raw['tstLabel'][
            0]

        trnData.append(trnData1)
        trnLabel.append(trnLabel1)
        tstData.append(tstData1)
        tstLabel.append(tstLabel1)
        trnPerLabel.append(np.ones(len(trnData1)) * (dataPath - 1))
        tstPerLabel.append(np.ones(len(tstData1)) * (dataPath - 1))

    trnData = np.concatenate(trnData, axis=0)
    trnLabel = np.hstack(trnLabel)

    tstData = np.concatenate(tstData, axis=0)
    tstLabel = np.hstack(tstLabel)

    trnPerLabel = np.hstack(trnPerLabel)
    tstPerLabel = np.hstack(tstPerLabel)

    sourceData = np.concatenate((trnData, tstData), axis=0)
    sourceLabel = np.concatenate((trnLabel, tstLabel), axis=0)
    sourcePerLabel = np.concatenate((trnPerLabel, tstPerLabel), axis=0)

    targetList = list(set(range(allSubNum)).difference(set(SourceList)))
    # targetList = list(set(range(7,9)).difference(set(SourceList)))
    for dataPath in targetList:
        dataPath = dataPath + 1
        raw = sio.loadmat('./data/BCI_IV_2a/' + data_len + '/' + str(dataPath) + '.mat')
        targetData, targetLabel, targetTstData, targetTstLabel = raw['trnData'], raw['trnLabel'][0], raw['tstData'], \
                                                                 raw['tstLabel'][0]

    # targetTrnData, targetTrnLabel, targetValData, TargetValLabel = trn_tst_split(targetData, targetLabel, loadN)

    return sourceData, sourceLabel, targetData, targetLabel, targetTstData, targetTstLabel


def bciiv2a_single(aug=False, reproduce=True, dataset='B', loadN='1', subject=1):


    raw = sio.loadmat('./data/BCI_IV_2a/data_0-4/' + str(subject) + '.mat')
    targetData, targetLabel, targetTstData, targetTstLabel = raw['trnData'], raw['trnLabel'][0], raw['tstData'], raw['tstLabel'][0]

    return targetData, targetLabel, targetTstData, targetTstLabel


def dataB_aug(path):
    # if i < 8:
    #     path = './data/dataB/0' + str(i) + '/B0' + str(i) + '01T.mat'
    # else:
    #     path = './data/dataB/0' + str(i) + '/T0' + str(i) + '01T.mat'
    reproduce = True


    dataset = loadData(path)
    data, label = dataset.signalCut()

    label = label - 1


    first_list = list(np.where(label==0)[0])
    second_list = list(np.where(label==1)[0])
    third_list = list(np.where(label==2)[0])

    if reproduce:
        random.seed(2019)
    random.shuffle(first_list)
    random.shuffle(second_list)
    random.shuffle(third_list)

    num = 50 #if len(data) == 120 else 150

    trnList = first_list[0:num] + second_list[0:num] + third_list[0:num]
    tstList = first_list[num:] + second_list[num:] + third_list[num:]

    if reproduce:
        random.seed(2019)
    random.shuffle(trnList)
    random.shuffle(tstList)

    trnData = data[trnList]
    trnLabel = label[trnList]
    tstData = data[tstList]
    tstLabel = label[tstList]




    data_aug = []
    label_aug = []
    for i in range(5):
        data_a = dataset.augment(trnData, sigma=0.5)
        data_f = dataset.signalFilter(data_a)

        if i == 0:
            data_aug = dataset.signalProcess(data_f, dsFlag=True)
            label_aug = trnLabel
        else:
            data_aug = np.concatenate((data_aug,dataset.signalProcess(data_f, dsFlag=True)), axis=0)
            label_aug = np.concatenate((label_aug, trnLabel), axis=0)


    trnData = dataset.signalFilter(trnData)
    tstData = dataset.signalFilter(tstData)

    data = np.concatenate((trnData[:,:,:,0], data_aug), axis=0)
    label = np.concatenate((trnLabel, label_aug),axis=0)


    # data = dataset.signalNorm(data)
    # data = (data - data.min(axis=2, keepdims=True)) / (data.max(axis=2, keepdims=True) - data.min(axis=2, keepdims=True))

    if len(data.shape) != 4:
        data = np.expand_dims(data, axis=-1)

    if len(tstData.shape) != 4:
        tstData = np.expand_dims(tstData, axis=-1)

    ###### all random
    # samList = list(range(len(data)))
    # random.shuffle(samList)
    #
    # num = 100 if len(data) == 120 else 150
    #
    # trnData = data[samList[:num]]
    # trnLabel = label[samList[:num]]
    # tstData = data[samList[num:]]
    # tstLabel = label[samList[num:]]

    ## label random

    first_list = list(np.where(label==0)[0])
    second_list = list(np.where(label==1)[0])
    third_list = list(np.where(label==2)[0])

    if reproduce:
        random.seed(2019)
    random.shuffle(first_list)
    random.shuffle(second_list)
    random.shuffle(third_list)

    trnList = first_list + second_list + third_list
    # tstList = first_list[num:] + second_list[num:] + third_list[num:]

    if reproduce:
        random.seed(2019)
    random.shuffle(trnList)

    trnData = data[trnList]
    trnLabel = label[trnList]
    # tstData = data[tstList]
    # tstLabel = label[tstList]





    return trnData, trnLabel, tstData, tstLabel




def bciii3a_multitask_da(aug=False, reproduce=True, dataset='B', loadN='1', subject=1, data_len='data_3-7'):
    data_len = 'data_0-4'

    trnData, trnLabel, tstData, tstLabel = [], [], [], []
    trnPerLabel, tstPerLabel = [], []

    allSubNum = 3

    sub = subject - 1
    SourceList = list(range(allSubNum))
    SourceList.pop(sub)



    for dataPath in SourceList:
        dataPath = dataPath + 1
        raw = sio.loadmat('./data/BCI_III_3a/' + data_len + '/s' + str(dataPath) + '.mat')
        trnData1, trnLabel1, tstData1, tstLabel1 = raw['trnData'], raw['trnLabel'][0], raw['tstData'], raw['tstLabel'][0]

        #### save mat
        # trnData1 = np.transpose(trnData1, [2,1,0])
        # tstData1 = np.transpose(tstData1, [2,1,0])
        # temTrn = []
        # temTst = []
        # for jab in range(len(trnData1)):
        #     trnData1_tem = exponential_running_standardize(trnData1[jab].T, factor_new=0.001,
        #                                     init_block_size=None, eps=1e-4)
        #     tstData1_tem = exponential_running_standardize(tstData1[jab].T, factor_new=0.001,
        #                                     init_block_size=None, eps=1e-4)
        #     temTrn.append(trnData1_tem.T)
        #     temTst.append(tstData1_tem.T)
        #
        # temTrn = np.stack(temTrn, axis=0)
        # temTst = np.stack(temTst, axis=0)
        # sio.savemat(str(dataPath)+'.mat', {'X_train': temTrn, 'Y_train':trnLabel1, 'X_test':temTst, 'Y_test':tstLabel1})

        trnData.append(trnData1)
        trnLabel.append(trnLabel1)
        tstData.append(tstData1)
        tstLabel.append(tstLabel1)
        trnPerLabel.append(np.ones(len(trnData1)) * (dataPath - 1))
        tstPerLabel.append(np.ones(len(tstData1)) * (dataPath - 1))

    trnData = np.concatenate(trnData, axis=0)
    trnLabel = np.hstack(trnLabel)

    tstData = np.concatenate(tstData, axis=0)
    tstLabel = np.hstack(tstLabel)

    trnPerLabel = np.hstack(trnPerLabel)
    tstPerLabel = np.hstack(tstPerLabel)

    sourceData = np.concatenate((trnData, tstData), axis=0)
    sourceLabel = np.concatenate((trnLabel, tstLabel), axis=0)
    sourcePerLabel = np.concatenate((trnPerLabel, tstPerLabel), axis=0)

    targetList = list(set(range(allSubNum)).difference(set(SourceList)))
    # targetList = list(set(range(7,9)).difference(set(SourceList)))
    for dataPath in targetList:
        dataPath = dataPath + 1
        raw = sio.loadmat('./data/BCI_III_3a/' + data_len + '/s' + str(dataPath) + '.mat')
        targetData, targetLabel, targetTstData, targetTstLabel = raw['trnData'], raw['trnLabel'][0], raw['tstData'], \
                                                                 raw['tstLabel'][0]
        # targetData = np.transpose(targetData, [2,1,0])
        # targetTstData = np.transpose(targetTstData, [2,1,0])

    # targetTrnData, targetTrnLabel, targetValData, TargetValLabel = trn_tst_split(targetData, targetLabel, loadN)
    channel = (np.array(list(range(17, 24)) + list(range(26, 33)) + list(range(35, 41))) - 1).tolist()
    return sourceData[:,channel,:], sourceLabel-1, targetData[:,channel,:], targetLabel-1, targetTstData[:,channel,:], targetTstLabel-1
     
    #return sourceData, sourceLabel-1, targetData, targetLabel-1, targetTstData, targetTstLabel-1



def bciii3a_single(aug=False, reproduce=True, dataset='B', loadN='1', subject=1):


    raw = sio.loadmat('./data/BCI_III_3a/data_0-4/s' + str(subject) + '.mat')
    targetData, targetLabel, targetTstData, targetTstLabel = raw['trnData'], raw['trnLabel'][0], raw['tstData'], raw['tstLabel'][0]
    channel = (np.array(list(range(17, 24)) + list(range(26, 33)) + list(range(35, 41))) - 1).tolist()

    return targetData[:, channel, :], targetLabel-1, targetTstData[:, channel, :], targetTstLabel-1





def bciiv2b_multitask_da(aug=False, reproduce=True, dataset='B', loadN='1', subject=1, data_len='data_3-7'):
    data_len = 'data_0-4'

    trnData, trnLabel, tstData, tstLabel = [], [], [], []
    trnPerLabel, tstPerLabel = [], []

    allSubNum = 9

    sub = subject - 1
    SourceList = list(range(allSubNum))
    SourceList.pop(sub)



    for dataPath in SourceList:
        dataPath = dataPath + 1
        raw = sio.loadmat('./data/BCI_IV_2b/' + data_len + '/' + str(dataPath) + '.mat')
        trnData1, trnLabel1, tstData1, tstLabel1 = raw['trnData'], raw['trnLabel'][0], raw['tstData'], raw['tstLabel'][0]

        #### save mat
        # trnData1 = np.transpose(trnData1, [2,1,0])
        # tstData1 = np.transpose(tstData1, [2,1,0])
        # temTrn = []
        # temTst = []
        # for jab in range(len(trnData1)):
        #     trnData1_tem = exponential_running_standardize(trnData1[jab].T, factor_new=0.001,
        #                                     init_block_size=None, eps=1e-4)
        #     tstData1_tem = exponential_running_standardize(tstData1[jab].T, factor_new=0.001,
        #                                     init_block_size=None, eps=1e-4)
        #     temTrn.append(trnData1_tem.T)
        #     temTst.append(tstData1_tem.T)
        #
        # temTrn = np.stack(temTrn, axis=0)
        # temTst = np.stack(temTst, axis=0)
        # sio.savemat(str(dataPath)+'.mat', {'X_train': temTrn, 'Y_train':trnLabel1, 'X_test':temTst, 'Y_test':tstLabel1})

        trnData.append(trnData1)
        trnLabel.append(trnLabel1)
        tstData.append(tstData1)
        tstLabel.append(tstLabel1)
        trnPerLabel.append(np.ones(len(trnData1)) * (dataPath - 1))
        tstPerLabel.append(np.ones(len(tstData1)) * (dataPath - 1))

    trnData = np.concatenate(trnData, axis=0)
    trnLabel = np.hstack(trnLabel)

    tstData = np.concatenate(tstData, axis=0)
    tstLabel = np.hstack(tstLabel)

    trnPerLabel = np.hstack(trnPerLabel)
    tstPerLabel = np.hstack(tstPerLabel)

    sourceData = np.concatenate((trnData, tstData), axis=0)
    sourceLabel = np.concatenate((trnLabel, tstLabel), axis=0)
    sourcePerLabel = np.concatenate((trnPerLabel, tstPerLabel), axis=0)

    targetList = list(set(range(allSubNum)).difference(set(SourceList)))
    # targetList = list(set(range(7,9)).difference(set(SourceList)))
    for dataPath in targetList:
        dataPath = dataPath + 1
        raw = sio.loadmat('./data/BCI_IV_2b/' + data_len + '/' + str(dataPath) + '.mat')
        targetData, targetLabel, targetTstData, targetTstLabel = raw['trnData'], raw['trnLabel'][0], raw['tstData'], \
                                                                 raw['tstLabel'][0]
        # targetData = np.transpose(targetData, [2,1,0])
        # targetTstData = np.transpose(targetTstData, [2,1,0])

    # targetTrnData, targetTrnLabel, targetValData, TargetValLabel = trn_tst_split(targetData, targetLabel, loadN)

    # channel = (np.array(list(range(17, 24)) + list(range(26, 33)) + list(range(35, 41))) - 1).tolist()


    return sourceData, sourceLabel, targetData, targetLabel, targetTstData, targetTstLabel



def bciiv2b_single(aug=False, reproduce=True, dataset='B', loadN='1', subject=1):


    raw = sio.loadmat('./data/BCI_IV_2b/data_0-4/' + str(subject) + '.mat')
    targetData, targetLabel, targetTstData, targetTstLabel = raw['trnData'], raw['trnLabel'][0], raw['tstData'], raw['tstLabel'][0]

    return targetData, targetLabel, targetTstData, targetTstLabel






def bciiv2b_all(aug=False, reproduce=True, dataset='B', loadN='1', subject=1, data_len='data_3-7'):
    data_len = 'data_0-4'

    trnData, trnLabel, tstData, tstLabel = [], [], [], []
    trnPerLabel, tstPerLabel = [], []

    allSubNum = 9

    sub = subject - 1
    SourceList = list(range(allSubNum))
    SourceList.pop(sub)



    for dataPath in SourceList:
        dataPath = dataPath + 1
        raw = sio.loadmat('./data/BCI_IV_2b/' + data_len + '/' + str(dataPath) + '.mat')
        trnData1, trnLabel1, tstData1, tstLabel1 = raw['trnData'], raw['trnLabel'][0], raw['tstData'], raw['tstLabel'][0]

        #### save mat
        # trnData1 = np.transpose(trnData1, [2,1,0])
        # tstData1 = np.transpose(tstData1, [2,1,0])
        # temTrn = []
        # temTst = []
        # for jab in range(len(trnData1)):
        #     trnData1_tem = exponential_running_standardize(trnData1[jab].T, factor_new=0.001,
        #                                     init_block_size=None, eps=1e-4)
        #     tstData1_tem = exponential_running_standardize(tstData1[jab].T, factor_new=0.001,
        #                                     init_block_size=None, eps=1e-4)
        #     temTrn.append(trnData1_tem.T)
        #     temTst.append(tstData1_tem.T)
        #
        # temTrn = np.stack(temTrn, axis=0)
        # temTst = np.stack(temTst, axis=0)
        # sio.savemat(str(dataPath)+'.mat', {'X_train': temTrn, 'Y_train':trnLabel1, 'X_test':temTst, 'Y_test':tstLabel1})

        trnData.append(trnData1)
        trnLabel.append(trnLabel1)
        tstData.append(tstData1)
        tstLabel.append(tstLabel1)
        trnPerLabel.append(np.ones(len(trnData1)) * (dataPath - 1))
        tstPerLabel.append(np.ones(len(tstData1)) * (dataPath - 1))

    trnData = np.concatenate(trnData, axis=0)
    trnLabel = np.hstack(trnLabel)

    tstData = np.concatenate(tstData, axis=0)
    tstLabel = np.hstack(tstLabel)

    trnPerLabel = np.hstack(trnPerLabel)
    tstPerLabel = np.hstack(tstPerLabel)

    sourceData = np.concatenate((trnData, tstData), axis=0)
    sourceLabel = np.concatenate((trnLabel, tstLabel), axis=0)
    sourcePerLabel = np.concatenate((trnPerLabel, tstPerLabel), axis=0)

    targetList = list(set(range(allSubNum)).difference(set(SourceList)))
    # targetList = list(set(range(7,9)).difference(set(SourceList)))
    for dataPath in targetList:
        dataPath = dataPath + 1
        raw = sio.loadmat('./data/BCI_IV_2b/' + data_len + '/' + str(dataPath) + '.mat')
        targetData, targetLabel, targetTstData, targetTstLabel = raw['trnData'], raw['trnLabel'][0], raw['tstData'], \
                                                                 raw['tstLabel'][0]

        targetPerLabel = np.ones(len(targetData)) * (dataPath - 1)
        targetTstPerLabel = np.ones(len(targetTstData)) * (dataPath - 1)

    sourceData = np.concatenate((sourceData, targetData), axis=0)
    sourceLabel = np.concatenate((sourceLabel, targetLabel), axis=0)
    sourcePerLabel = np.concatenate((sourcePerLabel, targetPerLabel), axis=0)

    
        # targetData = np.transpose(targetData, [2,1,0])
        # targetTstData = np.transpose(targetTstData, [2,1,0])

    # targetTrnData, targetTrnLabel, targetValData, TargetValLabel = trn_tst_split(targetData, targetLabel, loadN)

    # channel = (np.array(list(range(17, 24)) + list(range(26, 33)) + list(range(35, 41))) - 1).tolist()


    return sourceData, sourceLabel, targetTstData, targetTstLabel
