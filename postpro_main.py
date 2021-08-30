import os
import warnings
import subprocess
import shutil
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import nipype.interfaces.ants as ants
from medpy.filter.smoothing import anisotropic_diffusion

src_path = "/home/lukedude/gliomas/PreprocessPipeLine/post_poc/fined_data"
cache_path = "/home/lukedude/gliomas/PreprocessPipeLine/post_poc/cache"
out_path = "/home/lukedude/gliomas/PreprocessPipeLine/post_poc/post_fined_data"

MODALITY_ = ["ADC", "DWI", "T1", "T1C", "T2"]
ANAT_MODALITY = ["T2"]

def bet(in_file, out_file):
    command = ["bet", in_file, out_file]
    subprocess.call(command)
    return

def bias_correction(in_file, out_file, image_type=sitk.sitkFloat64):
    correct = ants.N4BiasFieldCorrection()
    correct.inputs.input_image = in_file
    correct.inputs.output_image = out_file
    try:
        done = correct.run()
        return done.outputs.output_image
    except IOError:
        warnings.warn(RuntimeWarning("ANTs N4BIasFieldCorrection could not be found."
                                     "Will try using SimpleITK for bias field correction"
                                     " which will take much longer. To fix this problem, add N4BiasFieldCorrection"
                                     " to your PATH system variable. (example: EXPORT PATH=${PATH}:/path/to/ants/bin)"))
        input_image = sitk.ReadImage(in_file, image_type)
        output_image = sitk.N4BiasFieldCorrection(input_image, input_image > 0)
        sitk.WriteImage(output_image, out_file)
        return os.path.abspath(out_file)

def noise_filtering(in_file, out_file):
    img = nib.load(in_file)
    img_arr = img.dataobj
    fined_img_arr = anisotropic_diffusion(img_arr)
    fined_img = nib.Nifti1Image(dataobj=fined_img_arr, affine=img.affine, header=img.header)
    nib.save(fined_img, out_file)

def do_post_pro(in_file,
                cache_path,
                out_file=None,
                do_BrainExtraction=False,
                do_BiasFieldCorrection=False,
                do_NoiseFilte=False):

    # TODO: 所有结果存入cache

    cache_file = os.path.join(cache_path, os.path.basename(in_file))

    ######################
    #  Brain Extraction
    ######################

    if do_BrainExtraction:
        try:
            bet(in_file=in_file,
                out_file=cache_file)
        except RuntimeError:
            print("\t Failed on : ", in_file, " in brain extraction! ")
            return

    ######################
    #  Bias Filed Correction
    ######################

    if do_BiasFieldCorrection:
        try:
            bias_correction(in_file=cache_file,
                            out_file=cache_file)
        except RuntimeError:
            print("\t Failed on : ", in_file, "in bias correction! ")
            return

    ######################
    #  Noise Filter
    ######################

    if do_NoiseFilte:
        try:
            noise_filtering(in_file=cache_file,
                            out_file=cache_file)
        except RuntimeError:
            print("\t Failed on : ", in_file, "in noise-filtering")
            return

    return


def scan_files(directory, prefix=None, postfix=None):
    files_list = []
    for root, sub_dirs, files in os.walk(directory):
        for file in files:
            if postfix:
                if file.endswith(postfix):
                    os.rename(os.path.join(root, file), os.path.join(root, file.replace(" ", "_")))
                    files_list.append(os.path.join(root, file.replace(" ", "_")))
            elif prefix:
                if file.startswith(prefix):
                    os.rename(os.path.join(root, file), os.path.join(root, file.replace(" ", "_")))
                    files_list.append(os.path.join(root, file.replace(" ", "_")))
            else:
                os.rename(os.path.join(root, file), os.path.join(root, file.replace(" ", "_")))
                files_list.append(os.path.join(root, file.replace(" ", "_")))
    return files_list

if __name__ == '__main__':
    sub_list = os.listdir(src_path)

    n_samples = len(sub_list)
    n_modality = 5
    means_modality = np.zeros(shape=(n_modality, n_samples))
    stds_modality = np.zeros(shape=(n_modality, n_samples))

    for i, sub in enumerate(sub_list):
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n", i+1, sub)
        sub_source_path = os.path.join(src_path, sub)
        sub_cache_path = os.path.join(cache_path, sub)
        sub_output_path = os.path.join(out_path, sub)

        if not os.path.exists(sub_cache_path):
            os.mkdir(sub_cache_path)
        if not os.path.exists(sub_output_path):
            os.mkdir(sub_output_path)

        # label
        label_ = scan_files(sub_source_path, None, None)[0]
        print(label_)
        label_out = os.path.join(sub_output_path, os.path.basename(label_))
        shutil.copyfile(label_, label_out)

        for j, modality in enumerate(MODALITY_):
            modality_source_path = os.path.join(sub_source_path, modality)
            modality_cache_path = os.path.join(sub_cache_path, modality)
            modality_output_path = os.path.join(sub_output_path, modality)

            if not os.path.exists(modality_output_path):
                os.mkdir(modality_output_path)
            if not os.path.exists(modality_cache_path):
                os.mkdir(modality_cache_path)

            obj_file = scan_files(modality_source_path, None, None)[0]
            print(obj_file)
            do_post_pro(in_file=obj_file,
                        cache_path=modality_cache_path,
                        do_BrainExtraction=True,
                        do_BiasFieldCorrection=True,
                        do_NoiseFilte=True)

            cache_file = os.path.join(modality_cache_path, os.path.basename(obj_file))

            img = nib.load(cache_file)
            img_arr = np.array(img.dataobj)
            means_modality[j, i] = img_arr.mean()
            stds_modality[j, i] = img_arr.std()

    # intensity normalization

    print(means_modality, "\n\n", stds_modality)
    mean = means_modality.mean(axis=1)
    std = stds_modality.mean(axis=1)
    print(mean, std)

    print("\n\n finish post-process, and calculating mean & std for every modality across samples\n")
    sub_list_cache = os.listdir(cache_path)
    for i, sub in enumerate(sub_list_cache):
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n", i+1, sub)
        sub_source_path = os.path.join(cache_path, sub)
        sub_output_path = os.path.join(out_path, sub)
        for j, modality in enumerate(MODALITY_):
            modality_source_path = os.path.join(sub_source_path, modality)
            modality_output_path = os.path.join(sub_output_path, modality)

            obj_file = scan_files(modality_source_path, None, None)[0]
            img = nib.load(obj_file)
            img_arr = np.array(img.dataobj).astype(np.float64)
            img_arr -= mean[j]
            img_arr /= std[j]
            fined_img = nib.Nifti1Image(dataobj=img_arr, affine=img.affine, header=img.header)
            nib.save(fined_img, os.path.join(modality_output_path, os.path.basename(obj_file)))

            img = nib.load(os.path.join(modality_output_path, os.path.basename(obj_file)))
            img_arr = np.array(img.dataobj)
            means_modality[j, i] = img_arr.mean()
            stds_modality[j, i] = img_arr.std()

    print(means_modality, "\n\n", stds_modality)
    mean = means_modality.mean(axis=1)
    std = stds_modality.mean(axis=1)
    print(mean, std)
