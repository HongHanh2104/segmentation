import pydicom as dicom 
import cv2 
from utils import load_file_from_folder
import argparse

def convert_dicom_to_img(input, output, type):
    dicom_arr = load_file_from_folder(input)
    print(dicom_arr)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dinput')
    parser.add_argument('--output')
    parser.add_argument('--type', default='png')
    args = parser.parse_args()

    convert_dicom_to_img(input=args.input, output=args.output,
                        type=args.type)

if __name__ == "__main__":
    main()
