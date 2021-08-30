import vtk
import nrrd
import numpy as np
import nibabel as nib
import os
src_path = "..."

def readnrrd(path):
    reader = vtk.vtkNrrdReader()
    reader.SetFileName(path)
    reader.Update()
    return reader.GetOutput(), reader.GetInformation()

def writenifti(img, path, info):
    writer = vtk.vtkNIFTIImageWriter()
    writer.SetInputData(img)
    writer.SetFileName(path)
    writer.SetInformation(info)
    writer.Write()

def scan_file(directory , prefix=None, postfix=None):
    files_list = []
    for root, sub_dirs, files in os.walk(directory):
        for file in files:
            if postfix:
                if file.endswith(postfix):
                    files_list.append(os.path.join(root, file))
            elif prefix:
                if file.startswith(prefix):
                    files_list.append(os.path.join(root, file))
            else:
                files_list.append(os.path.join(root,file))
    return files_list

def nrrd2nii(src, dst):
    _nrrd = nrrd.read(src)
    data = _nrrd[0]
    header = _nrrd[1]
    img = nib.Nifti1Image(data, np.eye(4))
    nib.save(img, dst)

if __name__ == '__main__':
    sub_list = os.listdir(src_path)
    for i, sub in enumerate(sub_list):
        sub_source_path = os.path.join(src_path, sub)
        src_obj = scan_file(sub_source_path)[0]
        dst_obj = os.path.join(sub_source_path, "fined_label_DR1.nii")
        nrrd2nii(src_obj, dst_obj)