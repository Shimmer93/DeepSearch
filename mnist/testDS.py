from mnistWrapper import *
from LazierGreedy import *
from pickle import dump,load
import sys
from sys import argv
from datetime import datetime
import time
from os import mkdir
from os.path import exists
from tqdm import tqdm

if not exists("DSBatched"):
    mkdir("DSBatched")
path="DSBatched/"+str(datetime.now())+"/"
mkdir(path)
sys.stdout=open(path+"log.txt","w")
loss=False
grs=4
if len(argv)>1:
    loss=argv[1]=="xent"
if len(argv)>2:
    grs=int(argv[2])
Data={}
succ=0
tot=0

#target_set=load(open("/data/dnntest/zpengac/DeepSearch/CIFAR/indices.pkl","rb"))
with open('mnist_indices.txt', 'r') as f:
    target_set = [int(i) for i in f.readlines()]

t0 = time.time()
for j in tqdm(target_set):
    tot+=1
    print("Starting attack on image", tot, "with index",j)
    ret=DeepSearchBatched(x_test[j:j+1],mymodel,y_test[j],8/255,max_calls=20000, batch_size=64,x_ent=loss,gr_init=grs)
    dump(ret[1].reshape(1,32,32,1),open(path+"image_"+str(j)+".pkl","wb"))
    Data[j]=(ret[0],ret[2])
    if ret[0]:
        succ+=1
        print("Attack Succeeded with",ret[2],"queries, success rate is",100*succ/tot)
    else:
        print("Attack Failed using",ret[2],"queries, success rate is",100*succ/tot)
    dump(Data,open(path+"data.pkl","wb"))
print(f"Total time: {(time.time()-t0)/3600}hrs")
