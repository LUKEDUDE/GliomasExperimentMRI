import nibabel as nib
import numpy as np

def manipulate_adc(b0_path, b1000_path, output_path):
    print(b0_path)
    print(b1000_path)

    img_b0 = nib.load(b0_path).get_data()
    print(img_b0.shape)
    img_b1000 = nib.load(b1000_path).get_data()
    print(img_b1000.shape)
    img_ADC = np.zeros(shape=(int(img_b0.shape[0]), int(img_b0.shape[1]), int(img_b0.shape[2])))
    print(img_ADC.shape)
    ConstPixelDims = (int(img_b0.shape[0]), int(img_b0.shape[1]), int(img_b0.shape[2]))
    for z in range(ConstPixelDims[2]):
        for x in range(ConstPixelDims[0]):
            for y in range(ConstPixelDims[1]):
                img_ADC[x, y, z] = np.log((img_b1000[x, y, z]/img_b0[x, y, z])+1e-5)/-1000

    img = nib.load(b0_path)
    affine = img.affine
    new_image = nib.Nifti1Image(img_ADC, affine)
    nib.save(new_image, output_path)

    print(output_path)