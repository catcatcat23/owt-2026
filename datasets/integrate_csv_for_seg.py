## cd /raid/home/CAMCA/ss3112/Github/mae/datasets
## Usage: python integrate_csv_for_seg.py /raid/home/CAMCA/ss3112/Github/mae/Results/path/ "file1.txt; file2.txt; file3.txt"

import sys
import os

def extract_metrics(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
        
    # Extract metrics
    metrics = {}
    for line in content.split('\n'):
        if "dice_list" in line:
            metrics['dice'] = float(line.split(' ')[1])
        elif "nsd_list" in line:
            metrics['nsd'] = float(line.split(' ')[1])
        # else:
        #     metrics['nsd'] = 0
        # elif "ASD_list" in line:
        #     metrics['asd'] = float(line.split(' ')[1])
        # elif "hausdorff100_list" in line:
        #     metrics['hausdorff100'] = float(line.split(' ')[1])
        # elif "hausdorff95_list" in line:
        #     metrics['hausdorff95'] = float(line.split(' ')[1])
        # elif "SOverlap_list" in line:
        #     metrics['soverlap'] = float(line.split(' ')[1])
            
    return metrics

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_path> <file_list>")
        sys.exit(1)
        
    input_path = sys.argv[1]
    file_list = sys.argv[2].split('; ')

    # Check if all files exist
    for filename in file_list:
        full_path = os.path.join(input_path, filename)
        if not os.path.exists(full_path):
            print(f"Error: File {full_path} does not exist")
            sys.exit(1)
    
    dice_list = []
    nsd_list = []
    # asd_list = []
    # hausdorff100_list = []
    # hausdorff95_list = []
    # soverlap_list = []
    
    for filename in file_list:
        full_path = os.path.join(input_path, filename)
        metrics = extract_metrics(full_path)
        
        dice_list.append(metrics['dice'])
        nsd_list.append(metrics['nsd'])
        # asd_list.append(metrics['asd'])
        # hausdorff100_list.append(metrics['hausdorff100'])
        # hausdorff95_list.append(metrics['hausdorff95'])
        # soverlap_list.append(metrics['soverlap'])
    
    num_files = len(file_list)
    
    print("\nMetrics Lists:")
    print(f"Dice List: {dice_list}")
    print(f"NSD List: {nsd_list}")
    # print(f"ASD List: {asd_list}")
    # print(f"Hausdorff100 List: {hausdorff100_list}")
    # print(f"Hausdorff95 List: {hausdorff95_list}")
    # print(f"Surface Overlap List: {soverlap_list}")
    
    print("\nAverage Metrics:")
    print(f"Average Dice: {sum(dice_list)/num_files:.8f}")
    print(f"Average NSD: {sum(nsd_list)/num_files:.8f}")
    # print(f"Average ASD: {sum(asd_list)/num_files:.8f}")
    # print(f"Average Hausdorff100: {sum(hausdorff100_list)/num_files:.8f}")
    # print(f"Average Hausdorff95: {sum(hausdorff95_list)/num_files:.8f}")
    # print(f"Average Surface Overlap: {sum(soverlap_list)/num_files:.8f}")

if __name__ == "__main__":
    main()
