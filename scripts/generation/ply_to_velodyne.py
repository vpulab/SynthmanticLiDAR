from email import header
import os
import numpy as np
import argparse
import pandas as pd
from tqdm import tqdm

def match_kitti_labels(np_labels):
    '''Convert CARLA label index into kitty '''

    Carla_labels = {'Unlabeled':0,
                    'Building':1,
                    'Fence':2,
                    'Other':3,
                    'Pedestrian':4,
                    'Pole':5,
                    'RoadLine':6,
                    'Road':7,
                    'SideWalk':8,
                    'Vegetation':9,
                    'Vehicles':10,
                    'Wall':11,
                    'TrafficSign':12,
                    'Sky':13,
                    'Ground':14,
                    'Bridge':15,
                    'RailTrack':16,
                    'GuardRail':17,
                    'TrafficLight':18,
                    'Static':19,
                    'Dynamic':20,
                    'Water':21,
                    'Terrain':22,
                    'Truck':23,
                    'Bicycle':24,
                    'Bus':25,
                    'Motorcycle':26,
                    'Motocyclist':27,
                    'Motorbikedriver':28,
                    'Cyclist':29,
                    'Bikedriver':30
                    }
    
    equivalences = {0:0,
                    1:50,
                    2:51,
                    3:1,
                    4:30,
                    5:80,
                    6:60,
                    7:40,
                    8:48,
                    9:70,
                    10:10,
                    11:50,
                    12:81,
                    13:0,
                    14:72,
                    15:52,
                    16:1,
                    17:51,
                    18:99,
                    19:99,
                    20:99,
                    21:1,
                    22:72,
                    23:18,
                    24:11,
                    25:13,
                    26:15,
                    27:32,
                    28:32,
                    29:31,
                    30:31}

    conversion = np.vectorize(equivalences.get)(np_labels)

    return conversion


def ply_to_velodyne(ply_df,name):
    ##Start parsing until end_header
    ply_position = ply_df.iloc[:,:-3]
    ply_label = ply_df.iloc[:,-1]

    positions_np = ply_position.to_numpy(dtype = float)


    #fake remissions
    positions_np = np.hstack([positions_np,np.zeros((positions_np.shape[0],1))]).flatten().astype('float32')
    #store as a binary
    positions_np.tofile("velodyne/{}.bin".format(name))
   
    #label_conversion



    #store_labels
    labels_np =  np.vstack([match_kitti_labels(ply_label).astype('uint16'),np.zeros(ply_label.shape)]).T.flatten().astype('uint16')
    labels_np.tofile("labels/{}.label".format(name))



if __name__ == '__main__':
    parser = argparse.ArgumentParser("./ply_to_velodyne.py")

    parser.add_argument('--folder', '-f',
        type=str,
        default="00",
        required=True,
        help='Folder with files to change',
    )
    FLAGS,_ = parser.parse_known_args()

    if not os.path.exists("velodyne/"):
        os.mkdir("velodyne")
    if not os.path.exists("labels"):
        os.mkdir("labels")
    

    ply_folder = os.path.join(FLAGS.folder)
    ply_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(ply_folder)) for f in fn]
    ply_files.sort()
    ply_files = tqdm(ply_files)
    
    for ply in ply_files:

        ply_df = pd.read_csv(ply,skiprows=10,sep=' ',header=None)

        ply_to_velodyne(ply_df,ply[-10:-4])









