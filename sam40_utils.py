import glob
import numpy as np
import os
import pandas as pd
import scipy.io

NUM_SUBJECTS = 40
NUM_SITUATIONS = 9
EEG_CHANNELS = 32

def load_sam40_data(sam40_path: str) -> tuple:
    sub_data = {}
    sub_labels = {}

    labels = pd.read_csv(os.path.join(sam40_path, 'labels.csv'))
    situations = ["math_1", "math_2", "math_3", "mirror_1", "mirror_2", "mirror_3",\
                  "stroop_1", "stroop_2", "stroop_3"]

    for sub in range(1, NUM_SUBJECTS + 1):
        subject_data = {}
        subject_labels = []
        print(f"Subject {sub:02d}")
        combined_data = get_subject_sit_data_sam40(sam40_path, sub)
        combined_labels = labels.iloc[sub - 1]
        for sit in range(0, NUM_SITUATIONS):
            subject_data[sit + 1] = np.array(combined_data[sit]).astype(np.float32)
            label1 = combined_labels[situations[sit]]
            label2 = combined_labels[situations[sit]]
            label3 = combined_labels[situations[sit]]
            subject_labels.append([label1, label2, label3])
            print(f"\t> Situation {sit:02d}")
        sub_data[sub] = subject_data
        sub_labels[sub] = subject_labels
        print(f"Done...\n---")
    
    return (sub_data, sub_labels)

def get_subject_sit_data_sam40(sam40_path, sub_id):
    files = sorted(glob.glob(f"{sam40_path}/*sub_{sub_id}_*.mat"))
    combined = []
    for file in files:
        data = scipy.io.loadmat(file)
        data = data['Clean_data']
        combined.append(data)
    return combined
