"""
Generate ans file and its zip file for submission.
"""

# Imports
import os
from zipfile import ZipFile

from config import *


# Write the prediction to the required .txt format.
def write_pred_to_txt(file_ids, all_pred, ans_file_dir, ansfile_name):
    completeName = os.path.join(ans_file_dir, ansfile_name)

    file =  open(completeName, "w+")
    for file_id, pred in zip(file_ids, all_pred):
        pred_class = str(int(pred)+1).zfill(3) + '.' + LABELS[int(pred)]
        file.write(file_id + ' ' + pred_class + "\n")
    file.close()

    print("Done generating ans.txt.")


# Write the answer file generated previously and make it into a zip file.
def write_ans_to_zip(ans_file_dir, ansfile_name, save_zip_name, filetag=''):
    completeName = os.path.join(ans_file_dir, save_zip_name)
    zipObj = ZipFile(completeName, 'w')
    zipObj.write(os.path.join(ans_file_dir, ansfile_name), 'answer.txt')
    zipObj.close()
    if(zipObj):
        print("Done.")