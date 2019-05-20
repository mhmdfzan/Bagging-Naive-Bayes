import csv
import random
import pandas as pd 
import numpy as np
import statistics
import math

dataTest = pd.read_csv("Testset.csv").values
dataTrain = pd.read_csv('Trainset.csv').values
dataUji = pd.read_csv('Testset.csv') #untuk copy semua isi dataFrame ke file tebakan

bootstrap_num = 5 #jumlah model bootstrap yang dibuat
len_bootstrap = 100 #panjang array list setiap model bootstrap yang dibuat
list_bootstrap = {}
meanstd = []; output = []

def getMeanandStd(data):
    kol11 = []; kol12 = []; kol21 = []; kol22 = []; total1 = 0; total2 = 0; mnstdev = []
    for i in data:
        if (i[2] == 1):
            kol11.append(i[0])
            kol21.append(i[1])
            total1 += 1
        if (i[2] == 2):
            kol12.append(i[0])
            kol22.append(i[1])
            total2 += 2
    hasil11 = [statistics.mean(kol11), statistics.stdev(kol11), total1]
    hasil12 = [statistics.mean(kol12), statistics.stdev(kol12), total2]
    hasil21 = [statistics.mean(kol21), statistics.stdev(kol21), total1]
    hasil22 = [statistics.mean(kol22), statistics.stdev(kol22), total2]
    mnstdev.append(hasil11); mnstdev.append(hasil12); mnstdev.append(hasil21); mnstdev.append(hasil22)
    return mnstdev

def rumus(x, mean, std):
    var = float(std)**2
    divider = (2*math.pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/divider

def naivebayes(data, meanstd):
    nilai1 = rumus(data[0], meanstd[0][0], meanstd[0][1]) * rumus(data[1], meanstd[2][0], meanstd[2][1]) * (meanstd[0][2]/len_bootstrap) 
    nilai2 = rumus(data[0], meanstd[1][0], meanstd[1][1]) * rumus(data[1], meanstd[3][0], meanstd[3][1]) * (meanstd[1][2]/len_bootstrap) 
    if (nilai1 > nilai2):
        return 0
    else:
        return 1

def main():
    for i in range(bootstrap_num): #bikin list model atau bootstrap, yaitu 5 list bootstrap 
        bootstrap = []
        for j in range(len_bootstrap):
            dataRandom = dataTrain[np.random.randint(0, len(dataTrain))]
            bootstrap.append(dataRandom)
        list_bootstrap[i] = np.copy(bootstrap)

    for i in range(bootstrap_num): #bikin list untuk nilai mean dan stardart deviasi per model
        hasil = getMeanandStd(list_bootstrap[i])
        meanstd.append(hasil)

    for test in dataTest:
        nilai = 0
        for mnstd in meanstd:
            nilai += naivebayes(test,mnstd)
        if (nilai > 0):
            output.append(2)
        else:
            output.append(1)
    print(output)
    df = pd.DataFrame({'Class' : output})
    df.to_csv("TebakanTugas4ML.csv", header=False, index=False)
    print("Silahkan cek hasil pada file TebakanTugas4ML.csv")
main()
