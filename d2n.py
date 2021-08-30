import numpy as np
import os
import matplotlib.pyplot as plt
import nibabel
import pydicom
import dicom2nifti

def sub_dicom2nifti(sub_path):
    modality = os.listdir(sub_path)
    for folder in modality:
        dicom_folder = sub_path + folder
        nii_folder = sub_path + 'nii_' + folder
        os.mkdir(nii_folder)
        dicom2nifti.convert_directory(dicom_folder, nii_folder, compression=False)
        print(folder + "has been converted.")
    print("Conversion finished.")

sub_path = "..."
sub_dicom2nifti(sub_path)