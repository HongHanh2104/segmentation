import SimpleITK as sitk
import argparse
import os
import glob 
import time
from pathlib import Path
import numpy as np

def load_nii_image(root_path=None,input_folder=None):
    assert root_path is not None, "Missing the input path!"
    root_path = Path(root_path)
    path = root_path / input_folder

    def get_sort_key(x):
        return int(x.split("-")[-1].split(".")[0])

    nii_list = sorted(glob.glob(str(path / "volume-*")),
                      key=get_sort_key)
    return nii_list

def get_nii_image(nii_serie, i, type):
    img_path = ""
    if (type == "PATIENT"):
        img_path = nii_serie[i]
    elif (type == "LABEL"):
        img_path = nii_serie[i].replace("volume", "segmentation")
    
    image = sitk.ReadImage(img_path)
    if image.GetPixelID() > 4:
        arr = sitk.GetArrayFromImage(image).astype(np.int16)
        return sitk.GetImageFromArray(arr)
    else:
        return image

def read_information(filename):
    reader = sitk.ImageFileReader()
    reader.SetFileName(filename)
    reader.ReadImageInformation()
    for k in reader.GetMetaDataKeys():
        v = reader.GetMetaData(k)
        print("({0}) = = \"{1}\"".format(k, v))

def set_series_tag_values(nii_image):
    direction = nii_image.GetDirection()
    modification_time = time.strftime("%H%M%S")
    modification_date = time.strftime("%Y%m%d")
    series_tag_values = [("0008|0031",modification_time), # Series Time
                         ("0008|0021",modification_date), # Series Date
                         ("0008|0008","DERIVED\\SECONDARY"), # Image Type
                         ("0020|000e", "1.2.826.0.1.3680043.2.1125."+modification_date+".1"+modification_time), # Series Instance UID
                         ("0020|0037", '\\'.join(map(str, (direction[0], direction[3], direction[6],# Image Orientation (Patient)
                                                    direction[1],direction[4],direction[7])))),
                         ("0008|103e", "Created-SimpleITK")] # Series Description
    return series_tag_values

def writeSlices(img_serie, series_tag_values, i, final_folder):
    image_slice = img_serie[:,:,i]
    
    # Tags shared by the series.
    list(map(lambda tag_value: image_slice.SetMetaData(tag_value[0], tag_value[1]), series_tag_values))

    # Slice specific tags.
    image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d")) # Instance Creation Date
    image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S")) # Instance Creation Time

    # Setting the type to CT preserves the slice location.
    image_slice.SetMetaData("0008|0060", "CT")  # set the type to CT so the thickness is carried over

    # (0020, 0032) image position patient determines the 3D spacing between slices.
    image_slice.SetMetaData("0020|0032", '\\'.join(map(str, img_serie.TransformIndexToPhysicalPoint((0,0,i))))) # Image Position (Patient)
    image_slice.SetMetaData("0020,0013", str(i)) # Instance Number

    # Write to the output directory and add the extension dcm, to force writing in DICOM format.
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    writer.SetFileName(os.path.join(final_folder,str(i)+'.dcm'))
    writer.Execute(image_slice)


def convert():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root')
    parser.add_argument('--input')
    parser.add_argument('--output')
    parser.add_argument('--type')
    args = parser.parse_args()

    nii_list = load_nii_image(args.root, args.input)
    type_file = args.input.split('_')[-1]
    output_path = os.path.join(args.root, args.output)
    type_file_folder = os.path.join(output_path, type_file)
    if os.path.exists(output_path) == False:
        os.mkdir(output_path)
    
    if os.path.exists(os.path.join(type_file_folder)) == False:
        os.mkdir(os.path.join(type_file_folder))

    for i in range(len(nii_list)):
        patient = int(nii_list[i].split('-')[-1].split('.')[0])
        patient_folder = os.path.join(type_file_folder, "LiTS" + str(patient), args.type)
        if os.path.exists(patient_folder) == False:
            os.makedirs(patient_folder)

        nii_image = get_nii_image(nii_list, i, args.type) 
        
        series_tag_values = set_series_tag_values(nii_image)
        
               
        list(map(lambda k: writeSlices(img_serie=nii_image, series_tag_values=series_tag_values, 
                i=k, final_folder=patient_folder), range(nii_image.GetDepth())))
        print("done {0}".format(str(patient)))

def test():
    path = '../Data/LiTS/LiTS_train/segmentation-38.nii'
    img = sitk.ReadImage(path)
    if img.GetPixelID() > 4:
        print("ahihi")
       
if __name__ == "__main__":
    convert()
