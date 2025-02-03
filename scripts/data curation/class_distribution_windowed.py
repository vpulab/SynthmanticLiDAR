#Class distribution calculator
#%%
'''
This script can be used to find sections of sequences where classes may be oversampled. Particularly useful when there are samples where the capture vehicle stopped in a traffic light or a stop sign, oversampling their surroundings.
'''


import argparse
import os
import json

import yaml
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm


# Aux functions

def percentage(x):
    """Compute softmax values for each sets of scores in x."""
    x= np.array(x)
    dist = np.round(x*100 / x.sum(),3)
    return dist


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

parser.add_argument('--window_size','-ws',
    type=int,
    default=100,
    required=False,
    help='Window size',
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
    if os.path.isdir(label_paths):
        print("Labels folder exists! Using labels from %s" % label_paths)
    else:
        print("Labels folder doesn't exist! Exiting...")
        quit()
    label_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
            os.path.expanduser(label_paths)) for f in fn]
    label_names.sort()

    total_counts = {}
    
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
    #training_sequences = ["71","72","73","74","75","76","77"]
    training_sequences = ["71","72","93","94","95","96","97"]
    training_sequences = ["101","102","103","104","105","106","107"]#,"108","109"]
    training_sequences = ["111","113","114","115","116","117"]
    training_sequences = ["60","61","62","63","64","65","66","67"]
    dataset_counts = {}

    for sequence in training_sequences:
        total_counts = {}
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
        label_names = tqdm(label_names)
        label_names.set_description("Sequence: {}".format(sequence))

        counter = 0
        window_counts = {}
        windows_processed = 0
        for scan in label_names:
            if counter == FLAGS.window_size:
                #Get and "add" labels
                labels = np.fromfile(scan,dtype=np.uint16)
                labels = labels.reshape((len(labels))//2,2)[:,0]
                unique, counts = np.unique(labels, return_counts=True)
                counts = dict(zip(unique,counts))
                window_counts = Counter(window_counts) + Counter(counts)

                
                for key in window_counts.keys():
                    if key not in list(total_counts.keys()):
                        total_counts[key] = [0 for a in np.arange(windows_processed)]
                        total_counts[key].append(window_counts[key])
                    else:
                        total_counts[key].append(window_counts[key])
                
                
                for key in total_counts.keys():
                    if key not in list(window_counts.keys()):
                        total_counts[key].append(0)
                window_counts = {}
                counter = 0
                windows_processed += 1
            else:
                #Get and "add" labels
                labels = np.fromfile(scan,dtype=np.uint16)
                labels = labels.reshape((len(labels))//2,2)[:,0]
                unique, counts = np.unique(labels, return_counts=True)
                counts = dict(zip(unique,counts))
                window_counts = Counter(window_counts) + Counter(counts)
                counter +=1
    
        dataset_counts[sequence] = total_counts



# endregion

#Process results


print(dataset_counts)
#Counts per class
dataset_counts_fixed_int = {}
for sequence_no,sequence_dict in dataset_counts.items():
    keys = list(sequence_dict.keys())
    conversion = np.vectorize(classes['labels'].get)(keys)
    converted_lists = []
    for class_list in sequence_dict.values():
        converted_list = []
        for item in class_list:
            converted_list.append(int(item))
        converted_lists.append(converted_list)

    sequence_counts = dict(zip(conversion,converted_lists))
    dataset_counts_fixed_int[int(sequence_no)] = sequence_counts
#total_distribution = dict(zip(conversion,percentage(list(total_counts.values()))))

print(dataset_counts_fixed_int)
data = {'counts':dataset_counts_fixed_int}#,'percentage':total_distribution}

if FLAGS.sequence != "all":
    with open('class_distribution_windowed_{}_seq_{}.json'.format(FLAGS.output_name,FLAGS.sequence),'w') as f:
        json.dump(data,f,indent=4)
else:
    with open('class_distribution_windowed_{}.json'.format(FLAGS.output_name),'w') as f:
        json.dump(data,f,indent=4)

