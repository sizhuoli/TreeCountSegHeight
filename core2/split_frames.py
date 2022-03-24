#    Author: Ankit Kariryaa, University of Bremen

import os
import json
from sklearn.model_selection import train_test_split, KFold

#Divide the frames into n-splits
def cross_validation_split(frames, frames_json, patch_dir, n=10):
    """ n-times divide the frames into training, validation and test.

    Args:
        frames: list(FrameInfo)
            list of the all the frames.
        frames_json: str
            Filename of the json where data is written.
        patch_dir: str
            Path to the directory where frame_json is stored.
    """
    if os.path.isfile(frames_json):
        print("Reading n-splits from file")
        with open(frames_json, 'r') as file:
            fjson = json.load(file)
            splits = fjson['splits']
    else:
        print("Creating and writing n-splits to file")
        frames_list = list(range(len(frames)))
        # Divide into n-split, each containing training and test set
        kf = KFold(n_splits=n, shuffle=True, random_state=1117)
        print("Number of spliting iterations:", kf.get_n_splits(frames_list))
        splits = []
        for train_index, test_index in kf.split(frames_list):
            splits.append([train_index.tolist(), test_index.tolist()])
        frame_split = {
            'splits': splits
        }
        if not os.path.exists(patch_dir):
            os.makedirs(patch_dir)
        with open(frames_json, 'w') as f:
            json.dump(frame_split, f)

    return splits

def split_dataset(frames, frames_json, patch_dir, val_size = 0.2):
    """Divide the frames into training, validation and test.

    Args:
        frames: list(FrameInfo)
            list of the all the frames.
        frames_json: str
            Filename of the json where data is written.
        patch_dir: str
            Path to the directory where frame_json is stored.
        test_size: float, optional
            Percentage of the test set.
        val_size: float, optional
            Percentage of the val set.
    """
    if os.path.isfile(frames_json):
        print("Reading train-test split from file")
        with open(frames_json, 'r') as file:
            fjson = json.load(file)
            training_frames = fjson['training_frames']
            # testing_frames = fjson['testing_frames']
            validation_frames = fjson['validation_frames']
    else:
        print("Creating and writing train-test split from file")
        frames_list = list(range(len(frames)))
        # Divide into training and test set
        # training_frames, testing_frames = train_test_split(frames_list, test_size=test_size)

        # Further divide into training set into training and validataion set
        training_frames, validation_frames = train_test_split(frames_list, test_size=val_size)
        frame_split = {
            'training_frames': training_frames,
            # 'testing_frames': testing_frames,
            'validation_frames': validation_frames
        }
        if not os.path.exists(patch_dir):
            os.makedirs(patch_dir)
        with open(frames_json, 'w') as f:
            json.dump(frame_split, f)

    print('training_frames', training_frames)
    print('validation_frames', validation_frames)
    # print('testing_frames', testing_frames)
    return (training_frames,validation_frames)  # (training_frames,validation_frames, testing_frames )
