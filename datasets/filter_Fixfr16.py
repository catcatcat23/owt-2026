
import os
import random
import h5py
import numpy as np
import pandas as pd
import re
import sys

def sorted_nicely( l ): 
    """ Sort the given iterable in the way that humans expect.""" 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

fix_frame = int(sys.argv[3]) ## 4 / 16
base_dir = sys.argv[1] #'/raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224/Training_H100_2.csv'
output_dir = sys.argv[2]

df = pd.read_csv(base_dir)

sample_list = [sample_pth for sample_pth in df["image_pth"]]
masks_list = [sample_pth for sample_pth in df["mask_pth"]]

filtered_sample_list = {"image_pth":[], "mask_pth":[]}

for data_path in sample_list:
	# print("data_path", data_path)
	cur_frame = int(data_path.split(".jpg")[0].split("_")[-1])
	# print("cur_frame", cur_frame)
	base_data_path = "_".join(data_path.split(".jpg")[0].split("_")[:-1])
	# print("base_data_path", base_data_path)
	num = 0
	for i in range(cur_frame, cur_frame+fix_frame):
		final_data_path = base_data_path+"_"+str(i)+".jpg"
		# print("final_data_path", final_data_path)
		if os.path.isfile(final_data_path):
			num+=1
	if num == fix_frame:
		filtered_sample_list["image_pth"].append(data_path)
	# break

for data_path in masks_list:
	# print("data_path", data_path)
	cur_frame = int(data_path.split(".png")[0].split("_")[-1])
	# print("cur_frame", cur_frame)
	base_data_path = "_".join(data_path.split(".jpg")[0].split("_")[:-1])
	# print("base_data_path", base_data_path)
	num = 0
	for i in range(cur_frame, cur_frame+fix_frame):
		final_data_path = base_data_path+"_"+str(i)+".png"
		# print("final_data_path", final_data_path)
		if os.path.isfile(final_data_path):
			num+=1
	if num == fix_frame:
		filtered_sample_list["mask_pth"].append(data_path)
	# break

# print("filtered_sample_list", filtered_sample_list)
submit_df = pd.DataFrame.from_dict(filtered_sample_list)
submit_df.to_csv(output_dir, index=False)


## python filter_Fixfr16.py /raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224/Training_H100_2.csv /raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224/Training_Fixfr16_H100_2.csv
## python filter_Fixfr16.py /raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224/Test_H100_2.csv /raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224/Test_Fixfr16_H100_2.csv
## python filter_Fixfr16.py /raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAtlas_1000_224/Training_H100_2.csv /raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAtlas_1000_224/Training_Fixfr16_H100_2.csv
## python filter_Fixfr16.py /raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAtlas_1000_224/Test_H100_2.csv /raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAtlas_1000_224/Test_Fixfr16_H100_2.csv

# python filter_Fixfr16.py /raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Final/Training_H100_2_Final112.csv /raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Final/Training_Fixfr16_H100_2_Final112.csv
# python filter_Fixfr16.py /raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Final/Test_H100_2_Final112.csv /raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Final/Test_Fixfr16_H100_2_Final112.csv

# python filter_Fixfr16.py /raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Final/Training_H100_2_Final112.csv /raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Final/Training_Fixfr4_H100_2_Final112.csv
# python filter_Fixfr16.py /raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Final/Test_H100_2_Final112.csv /raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Final/Test_Fixfr4_H100_2_Final112.csv

# python filter_Fixfr16.py /raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAtlas/AbdAtlas_1000_224_Final/Training_H100_2_Final112.csv /raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAtlas/AbdAtlas_1000_224_Final/Training_Fixfr4_H100_2_Final112.csv 4 
# python filter_Fixfr16.py /raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAtlas/AbdAtlas_1000_224_Final/Test_H100_2_Final112.csv /raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAtlas/AbdAtlas_1000_224_Final/Test_Fixfr4_H100_2_Final112.csv 4

# python filter_Fixfr16.py /raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAtlas/AbdAtlas_1000_224_Final/Training_H100_2_Final112.csv /raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAtlas/AbdAtlas_1000_224_Final/Training_Fixfr16_H100_2_Final112.csv 16
# python filter_Fixfr16.py /raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAtlas/AbdAtlas_1000_224_Final/Test_H100_2_Final112.csv /raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAtlas/AbdAtlas_1000_224_Final/Test_Fixfr16_H100_2_Final112.csv 16

# python filter_Fixfr16.py /mnt/weka/wekafs/rad-megtron/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Training/Training_Final112.csv /mnt/weka/wekafs/rad-megtron/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Training/Training_Fixfr16_Final112.csv 16
# python filter_Fixfr16.py /raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Training/Training_H100_2_Final112.csv /raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Training/Training_Fixfr16_H100_2_Final112.csv 16
# python filter_Fixfr16.py /raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Test/Test_H100_2_Final112.csv /raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Test/Test_Fixfr16_H100_2_Final112.csv 16

# python filter_Fixfr16.py /mnt/weka/wekafs/rad-megtron/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Training/Training_Final112.csv /mnt/weka/wekafs/rad-megtron/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Training/Training_Fixfr4_Final112.csv 4
# python filter_Fixfr16.py /raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Training/Training_H100_2_Final112.csv /raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Training/Training_Fixfr4_H100_2_Final112.csv 4
# python filter_Fixfr16.py /raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Test/Test_H100_2_Final112.csv /raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAutoPet/Test/Test_Fixfr4_H100_2_Final112.csv 4

# python filter_Fixfr16.py /raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAtlas/AbdAtlas_1000_224_Final/Training_H100_2_Final112_4c.csv /raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAtlas/AbdAtlas_1000_224_Final/Training_Fixfr16_H100_2_Final112_4c.csv 16
# python filter_Fixfr16.py /raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAtlas/AbdAtlas_1000_224_Final/Test_H100_2_Final112_4c.csv /raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAtlas/AbdAtlas_1000_224_Final/Test_Fixfr16_H100_2_Final112_4c.csv 16

# python filter_Fixfr16.py /raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAtlas/AbdAtlas_1000_224_Final/Training_H100_2_Final112_4c.csv /raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAtlas/AbdAtlas_1000_224_Final/Training_Fixfr4_H100_2_Final112_4c.csv 4
# python filter_Fixfr16.py /raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAtlas/AbdAtlas_1000_224_Final/Test_H100_2_Final112_4c.csv /raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/AbdAtlas/AbdAtlas_1000_224_Final/Test_Fixfr4_H100_2_Final112_4c.csv 4

# python filter_Fixfr16.py /raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Tumor_Final/Training_H100_2_Final112.csv /raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Tumor_Final/Training_Fixfr16_H100_2_Final112.csv 16
# python filter_Fixfr16.py /raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Tumor_Final/Test_H100_2_Final112.csv /raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Tumor_Final/Test_Fixfr16_H100_2_Final112.csv 16

# python filter_Fixfr16.py /raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Tumor_Final/Training_H100_2_Final112.csv /raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Tumor_Final/Training_Fixfr4_H100_2_Final112.csv 4
# python filter_Fixfr16.py /raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Tumor_Final/Test_H100_2_Final112.csv /raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/Abdomen1k/Abdomen1k_224_Tumor_Final/Test_Fixfr4_H100_2_Final112.csv 4

# python filter_Fixfr16.py /raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/Delay/Training/Training_H100_2_Final112.csv /raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/Delay/Training/Training_Fixfr4_H100_2_Final112.csv 4
# python filter_Fixfr16.py /raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/Delay/Test/Test_H100_2_Final112.csv /raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/Delay/Test/Test_Fixfr4_H100_2_Final112.csv 4

# python filter_Fixfr16.py /raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/PreArtery/Training/Training_H100_2_Final112.csv /raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/PreArtery/Training/Training_Fixfr4_H100_2_Final112.csv 4
# python filter_Fixfr16.py /raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/PreArtery/Test/Test_H100_2_Final112.csv /raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/PreArtery/Test/Test_Fixfr4_H100_2_Final112.csv 4

# python filter_Fixfr16.py /raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/PV/Training/Training_H100_2_Final112.csv /raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/PV/Training/Training_Fixfr4_H100_2_Final112.csv 4
# python filter_Fixfr16.py /raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/PV/Test/Test_H100_2_Final112.csv /raid/home/CAMCA/ss3112/Datasets/Med3d/Med3d_Others/RAOS/SyntheticMRI_pre/PV/Test/Test_Fixfr4_H100_2_Final112.csv 4







