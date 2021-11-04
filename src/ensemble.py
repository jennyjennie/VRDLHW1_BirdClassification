"""
Ensemble - Bagging (Majority Voting)
"""

# Imports
import pandas as pd
import numpy as np
import os
import zipfile

from config import *
from inference import write_pred_to_txt, write_ans_to_zip


# Apply Majority Voting in the existing zip file
def MajorityVoting(file_dir, fileNames):
    for i, fn in enumerate(fileNames):
        filepath = file_dir + '/' + fn
        zip_file = zipfile.ZipFile(filepath)
        df = pd.read_csv(zip_file.open('answer.txt'),
                         sep=' ', names=['id', 'labels'])
        labels_int = [int(label[0:3])-1 for label in df['labels']]
        numOflabels = len(labels_int)
        # Apply one hot encoding to deal with majority voting
        # Convert array of indices to 1-hot encoded numpy array
        # initalize np 2D-array with all zeros
        if i == 0:
            one_hot_Mat = np.zeros((numOflabels, NUM_CLASSES))  # 2D array -> (# of data in testing set, # of classes)
            img_id = df["id"]
        one_hot_Mat[np.arange(numOflabels), labels_int] += 1  # Accumulate voting from each model to corresponding label in 1-hot encoding

    # pick up the majority label in 1-hot encoding
    ensemble_result = np.argmax(one_hot_Mat, axis=1)
    # tranlate label(int) to label(str)
    label_result = [str(label_int + 1).zfill(3) + '.' + LABELS[label_int]
                    for label_int in ensemble_result]

    # write final result to text file
    d = {'id': img_id, 'label': label_result}
    df = pd.DataFrame(data=d)
    df.to_csv(file_dir+'/'+"ensemble.txt",
              index=False, header=None, sep=' ')
    print("Successfully complete ensemble prediction.")


def main():
    # Do Majority Voting in all zip file in the folder
    fileNames = []
    folder_name = "test_results"
    file_dir = PARENT_PATH + folder_name
    for file in os.listdir(file_dir):
        if file.endswith('zip'):
            fileNames.append(file)
    MajorityVoting(file_dir, fileNames)

    # Make the zip file for submission
    ansfile_name = 'ensemble.txt'
    save_zip_name = 'ensemble.zip'
    write_ans_to_zip(file_dir, ansfile_name, save_zip_name)


if __name__ == '__main__':
    main()
