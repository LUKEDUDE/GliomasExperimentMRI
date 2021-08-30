import os
import subprocess

# src_path = "/home/lukedude/gliomas/PreprocessPipeLine/fsl_test/original_data"
src_path = "/home/lukedude/gliomas/PreprocessPipeLine/fsl_test/pypreprocess_output"
cache_path = "/home/lukedude/gliomas/PreprocessPipeLine/fsl_test/cache"
out_path = "/home/lukedude/gliomas/PreprocessPipeLine/fsl_test/fined_data"

FUNC_MODALITY_PATH_ = ['ADC', 'reports', 'DWI/DWI_1000', 'T1', 'T1C']
FUNC_MODALITY_NAME_ = ['ADC', 'null', 'DWI', 'T1', 'T1C']
ANAT_PATH = ["T2"]

FSL_MNI_TEMPLATE = "/home/lukedude/gliomas/PreprocessPipeLine/fsl_test/MNI152_T1_1mm.nii.gz"

def registeration(src_path, dst_path, ref_path, matrix_path_saved):
    command = ["flirt", "-in", src_path, "-ref", ref_path, "-out", dst_path,
               "-bins", "256", "-cost", "corratio", "-searchrx", "0", "0",
               "-searchry", "0", "0", "-searchrz", "0", "0", "-dof", "12",
               "-interp", "spline", "-omat", matrix_path_saved]
    subprocess.call(command, stdout=open(os.devnull, "r"),
                    stderr=subprocess.STDOUT)
    return

def applytrans4Norm(src_path, dst_path, ref_path, matrix_path):
    command = ["flirt", "-in", src_path, "-ref", ref_path, "-out", dst_path,
               "-init", matrix_path, "-applyxfm"]
    subprocess.call(command, stdout=open(os.devnull, "r"),
                    stderr=subprocess.STDOUT)
    return

def orient2std(src_path, dst_path):
    command = ["fslreorient2std", src_path, dst_path]
    subprocess.call(command)
    return

def coregister(src_path, ref_path, dst_path, matrix_path):
    command = ["flirt", "-in", src_path, "-ref", ref_path, "-out", dst_path,
              "-omat", matrix_path]
    subprocess.call(command)
    return

def do_preproc(in_file,
               cache_path,
               out_file,
               is_anat=False,
               do_InitialProcess=True,
               do_Coregistration=False,
               do_Normalization=False,
               do_Label_Transformation=False,
               label_src=None,
               label_out=None,
               ref_anat_=None,
               trans_matrix_=None):

    ##################################
    #  Initial Process ( anat / func )
    ##################################

    # TODO: 真滴服了，要不就直接在原始文件上做修改吧,要不功能像陪准太麻烦了
    # TODO: 放到cache里吧...

    if do_InitialProcess:
        try:
            if is_anat:
                # anat
                cache_file = os.path.join(cache_path, "reo_"+os.path.basename(in_file))
                orient2std(in_file, cache_file)
                orient2std(label_src, label_out)
            else:
                # func
                orient2std(in_file, out_file)
        except RuntimeError:
            print("\t Failed on : ", in_file, " during initial processing! ")
            return

    ##################################
    #  Co-Registration ( func )
    ##################################

    if do_Coregistration:
        TransMatrix_coregis = os.path.join(cache_path, "trans_matrix_coregis.mat")
        try:
            coregister(out_file, ref_anat_, out_file, TransMatrix_coregis)
        except RuntimeError:
            print("\t Failed on : ", in_file, " during co-registering! ")
            return

    ##################################
    #  Spatial normalization ( anat / func )
    ##################################

    if do_Normalization:
        if is_anat:
            TransMatrix_norm = os.path.join(cache_path, "trans_matrix_norm.mat")
            try:
                registeration(cache_file, out_file, FSL_MNI_TEMPLATE, TransMatrix_norm)
            except RuntimeError:
                print("\t Failed on : ", in_file, " during normalization! ")
                return
        else:
            # for func
            if trans_matrix_:
                try:
                    applytrans4Norm(out_file, out_file, FSL_MNI_TEMPLATE, trans_matrix_)
                except RuntimeError:
                    print("\t Failed on : ", in_file, "during Normalization! ")
            else:
                print("There's no matrix")
            return

    ##################################
    #  Transformation on label
    ##################################

    if do_Label_Transformation:
        try:
            applytrans4Norm(label_out, label_out, FSL_MNI_TEMPLATE, TransMatrix_norm)
        except RuntimeError:
            print("\t Failed on : ", label_src, "during translating label_ to template! ")
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
    for i, sub in enumerate(sub_list):
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n", i+1, sub)
        sub_source_path = os.path.join(src_path, sub)
        sub_cache_path = os.path.join(cache_path, sub)
        sub_output_path = os.path.join(out_path, sub)

        if not os.path.exists(sub_cache_path):
            os.mkdir(sub_cache_path)
        if not os.path.exists(sub_output_path):
            os.mkdir(sub_output_path)

        # label path
        label_ = scan_files(sub_source_path, prefix="ra")[0]
        print(label_)
        label_out = os.path.join(sub_output_path, "fined_" + os.path.basename(label_))

        # TODO: 得调整下在除T2模态下，其他模态的预处理通道，尤其是向结构像和MNI模板的陪准过程 (done)

        ################################
        #  preprocess anatomical image
        ################################

        modality_cache_path = os.path.join(sub_cache_path, ANAT_PATH[0])
        modality_output_path = os.path.join(sub_output_path, ANAT_PATH[0])
        if not os.path.exists(modality_output_path):
            os.mkdir(modality_output_path)
        if not os.path.exists(modality_cache_path):
            os.mkdir(modality_cache_path)

        # anatomical image path
        str_ = os.path.join(sub_source_path, ANAT_PATH[0])
        anat_src = scan_files(str_, None, None)[0]
        anat_out = os.path.join(modality_output_path, "fined_"+os.path.basename(anat_src))
        print(anat_src)

        do_preproc(in_file=anat_src,
                   cache_path=modality_cache_path,
                   out_file=anat_out,
                   is_anat=True,
                   do_Normalization=True,
                   do_Label_Transformation=True,
                   label_src=label_,
                   label_out=label_out)

        # the anatomical image after being reoriented (cache) : as template for co-register
        anat_cache = os.path.join(modality_cache_path, "reo_"+os.path.basename(anat_src))
        # the trans matrix | T2 -> MNI152 normalization (cache) : ...
        trans_matrix = os.path.join(modality_cache_path, "trans_matrix_norm.mat")

        ################################
        #  preprocess functional image
        ################################

        for j, modality in enumerate(FUNC_MODALITY_PATH_):
            if modality == "reports":
                continue

            modality_source_path = os.path.join(sub_source_path, modality)
            modality_output_path = os.path.join(sub_output_path, FUNC_MODALITY_NAME_[j])
            modality_cache_path = os.path.join(sub_cache_path, FUNC_MODALITY_NAME_[j])

            if not os.path.exists(modality_output_path):
                os.mkdir(modality_output_path)
            if not os.path.exists(modality_cache_path):
                os.mkdir(modality_cache_path)

            # if j == 0:
            #     # ADC
            #     # TODO : ADC模态的图像数据质量较差，使用该方法经陪准后无法使用
            #     #        暂时仅施加归一化陪准根据变换矩阵
            #     obj_path = scan_files(modality_source_path, prefix="ra")[0]
            #     print(obj_path)
            #     do_preproc(in_file=obj_path,
            #                cache_path=modality_cache_path,
            #                out_file=os.path.join(modality_output_path, "fined_" + os.path.basename(obj_path)),
            #                do_Coregistration=False,
            #                do_Normalization=True,
            #                ref_anat_=anat_cache,
            #                trans_matrix_=trans_matrix
            #                )
            # else:
            #     # T1 & T1C & DWI
            #     obj_path = scan_files(modality_source_path, prefix='rs')[0]
            #     print(obj_path)
            #     do_preproc(in_file=obj_path,
            #                cache_path=modality_cache_path,
            #                out_file=os.path.join(modality_output_path, "fined_" + os.path.basename(obj_path)),
            #                do_Coregistration=False,
            #                do_Normalization=True,
            #                ref_anat_=anat_cache,
            #                trans_matrix_=trans_matrix)

            # ADC & DWI & T1 & T1C
            obj_path = scan_files(modality_source_path, prefix="ra")[0]
            print(obj_path)
            do_preproc(in_file=obj_path,
                       cache_path=modality_cache_path,
                       out_file=os.path.join(modality_output_path, "fined_" + os.path.basename(obj_path)),
                       do_Coregistration=False,
                       do_Normalization=True,
                       trans_matrix_=trans_matrix)