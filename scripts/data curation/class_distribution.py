'''
This script calculates the class distribution in a dataset. 
'''
#%%
import numpy as np
import argparse
import os
from collections import Counter
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

# Aux functions

def percentage(x):
    """Compute softmax values for each sets of scores in x."""
    x= np.array(x,dtype=np.int64)
    print(x)
    dist = np.round(x*100 / x.sum(),3)
    print(dist)
    return dist

#Load configuration file
# region 


parser = argparse.ArgumentParser("./class_distribution.py")

parser.add_argument('--dataset', '-d',
    type=str,
    default="00",
    required=True,
    help='Dataset folder',
)

parser.add_argument('--sequence','-s',
    type=str,
    default="all",
    required=False,
    help='Sequence to analyze.',
)

parser.add_argument('--config','-c',
    type=str,
    default="True",
    required=True,
    help='configuration file.',
)

parser.add_argument('--debug','-D',
    type=bool,
    default=False,
    required=False,
    help='Print debug information.',
)

parser.add_argument('--output_name','-o',
    type=str,
    default="dataset",
    required=False,
    help='Print debug information.',
)

parser.add_argument('--start_scan','-ss',
    type=int,
    default=-1,
    required=False,
    help='Slice index start',
)
parser.add_argument('--end_scan','-es',
    type=int,
    default=-1,
    required=False,
    help='Slice index end',
)

FLAGS,_ = parser.parse_known_args()
# endregion

with open(FLAGS.config) as file:
    classes = yaml.full_load(file)

if FLAGS.debug == True:
    print(classes['labels'])

#Analyze a single sequence or analyze all the dataset.
# region 

if FLAGS.sequence != "all":
    # does sequence folder exist?
    label_paths = os.path.join(FLAGS.dataset, "sequences",
                                    FLAGS.sequence, "labels")

    print(label_paths)
    if os.path.isdir(label_paths):
        print("Labels folder exists! Using labels from %s" % label_paths)
    else:
        print("Labels folder doesn't exist! Exiting...")
        quit()
    label_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
            os.path.expanduser(label_paths)) for f in fn]
    label_names.sort()

    total_counts = {}
    
    if FLAGS.start_scan != -1:
        label_names = label_names[FLAGS.start_scan:FLAGS.end_scan]

    label_names = tqdm(label_names)
    label_names.set_description("Sequence: {}".format(FLAGS.sequence))

    
    for scan in label_names:
        labels = np.fromfile(scan,dtype=np.uint16)
        labels = labels.reshape((len(labels))//2,2)[:,0]
        unique, counts = np.unique(labels, return_counts=True)
        counts = dict(zip(unique,counts))
        total_counts = Counter(total_counts) + Counter(counts)
else:
    #training_sequences = ["00","01","02","03","04","05","06","07","09","10"]
    #training_sequences = ["71","72","93","94","95","96","97"]
    #training_sequences = ["71","72","73","74","75","76","77"]
    #training_sequences = ["101","102","103","104","105","106","107","108","109","111","112","113","114","115","116","117"]
    training_sequences = ["60","61","62","63","64","65","66","67"]
    total_counts = {}

    for sequence in training_sequences:
         # does sequence folder exist?
        label_paths = os.path.join(FLAGS.dataset, "sequences",
                                        sequence, "labels")
        if os.path.isdir(label_paths):
            print("Labels folder exists! Using labels from %s" % label_paths)
        else:
            print("Labels folder doesn't exist! Exiting...")
            quit()
        label_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
                os.path.expanduser(label_paths)) for f in fn]
        label_names.sort()
        label_names = label_names[:4000]
        label_names = tqdm(label_names)
        label_names.set_description("Sequence: {}".format(sequence))

        for scan in label_names:
            labels = np.fromfile(scan,dtype=np.uint16)
            labels = labels.reshape((len(labels))//2,2)[:,0]
            unique, counts = np.unique(labels, return_counts=True)
            counts = dict(zip(unique,counts))
            total_counts = Counter(total_counts) + Counter(counts)

# endregion

#Process results
keys = list(total_counts.keys())
conversion = np.vectorize(classes['labels'].get)(keys)

#Counts per class
total_counts = dict(zip(conversion,[int(a) for a in list(total_counts.values())]))
total_distribution = dict(zip(conversion,percentage([int(a) for a in list(total_counts.values())])))

data = {'counts':total_counts,'percentage':total_distribution}

if FLAGS.sequence != "all":
    with open('class_distribution_{}_seq_{}.json'.format(FLAGS.output_name,FLAGS.sequence),'w') as f:
        json.dump(data,f,indent=4)
else:
    print('class_distribution_{}.json'.format(FLAGS.output_name))
    with open('class_distribution_{}.json'.format(FLAGS.output_name),'w') as f:
        json.dump(data,f,indent=4)

plt.bar(range(len(total_counts)),list(total_counts.values()),align='center')
plt.xticks(range(len(total_counts)), list(total_counts.keys()),rotation=90)
plt.yscale("log")
plt.show()

print(total_counts)
print(total_distribution)

