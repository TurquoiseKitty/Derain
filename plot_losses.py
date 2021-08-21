import os
import matplotlib.pyplot as plt
import csv

all_loss_recoreds=[]
path_to_search="model_loss_record"
count = 0
for filename in os.listdir(path_to_search):
    # get loss list
    data = []
    with open(path_to_search+"/"+filename,'r') as datafile:
        reader = csv.reader(datafile,delimiter=",")
        for row in reader:
            data.append(float(row[0]))
    plt.figure(count)
    plt.plot(data)
    plt.savefig("loss_graph/"+filename+"_loss_graph.png")
    count += 1