from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import os

class MyDataset(Dataset):
    def __init__(self, data, label, transform=None):
        self.data = data
        self.label = label
        self.transform = transform

    # end

    def __getitem__(self, index):
        x = self.data[index]
        y = self.label[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    # end

    def __len__(self):
        return len(self.data)

    # end


def is_float(str_number):
    if (str_number.split(".")[0]).isdigit() or str_number.isdigit() or  (str_number.split('-')[-1]).split(".")[-1].isdigit():
        return (True)
    return (False)


def myDraw(data, dataLabel, pathName, imgName):
    dataLabel = [str(dl) for dl in dataLabel]
    colorAll = ["green", "red", "blue", "brown","yellow", "purple", "black", "skyblue","coral", "sienna"]
    datingLabel_unique = np.unique(dataLabel)
    datingLabel_unique = np.sort(datingLabel_unique)

    classInfo = list()
    for j in range(len(datingLabel_unique)):
        classInfo.append([])


    for j in range(len(datingLabel_unique)):
        label = datingLabel_unique[j]
        color = ''
        xs = list()
        ys = list()
        if label == '':
            color = "grey"
        else:
            if len(datingLabel_unique) > 10:
                color = "red"
            else:
                if datingLabel_unique[j].isdigit() or is_float(datingLabel_unique[j]):
                    color = colorAll[int(float(datingLabel_unique[j]))]
                else:
                    n = -(j+1)
                    color = colorAll[n]
        for i in range(len(dataLabel)):
            if str(datingLabel_unique[j]) == str(dataLabel[i]):
                xs.append(data[i][0])
                ys.append(data[i][1])
        classInfo[j].append([label, color, xs ,ys])

    #[print(c[0]) for c in classInfo]
    #print(len(classInfo))
    fig = plt.figure(figsize = (10, 6))
    ax = fig.add_subplot(111)


    N = len(datingLabel_unique) 
    cmap = plt.cm.get_cmap("hsv", N+1)

    typeC = list()
    for j in range(len(datingLabel_unique)):
        if datingLabel_unique[j] == '' or datingLabel_unique[j] == "nan":
            t = ax.scatter(classInfo[j][0][2], classInfo[j][0][3], s = 30, c = "gray" )
        else:
            if N < 10:
                t = ax.scatter(classInfo[j][0][2], classInfo[j][0][3], s = 30, c = classInfo[j][0][1] )
            else:
                t = ax.scatter(classInfo[j][0][2], classInfo[j][0][3], s = 30, c = cmap(j) )
        typeC.append(t)

    for i in range(len(datingLabel_unique)):
        if datingLabel_unique[i] == '' or datingLabel_unique[i] == "nan":
            datingLabel_unique[i] = "NaN"
    
    ax.legend(typeC, datingLabel_unique, loc=[1,0])
    #plt.tight_layout()
    
 
    if  not os.path.exists(pathName):
        os.makedirs(pathName)
    fileName = pathName + imgName
    plt.savefig(fileName)
    #plt.show()
    return()

