## cd /raid/home/CAMCA/ss3112/Github/mae/datasets
## Usage: python integrate_csv_for_gen.py /raid/home/CAMCA/ss3112/Github/mae/Results/Abdomen1k_2D/Token_mae_vit_base_patch16-LA_1e-4_4_224_20_v11_v01_L2-LPIPS_GPU6_96_1200/eval_gen/ "cross_eval_gen_test_epoch1199_cls0,2,3,4_0.15.txt; cross_eval_gen_test_epoch1199_cls0,1,3,4_0.15.txt; cross_eval_gen_test_epoch1199_cls0,1,2,4_0.15.txt; cross_eval_gen_test_epoch1199_cls0,1,2,3_0.15.txt"

import sys
import os

def extract_metrics(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
        
    # Split into original and thresholded sections
    sections = content.split("Original Predictions Metrics:")[1].split("Thresholded Predictions Metrics:")
    original = sections[0]
    thresholded = sections[1]
    
    # Extract metrics from original section
    orig_metrics = {}
    for line in original.split('\n'):
        if "Average L2 Loss:" in line:
            orig_metrics['l2'] = float(line.split(': ')[1])
        elif "Average LPIPS Loss:" in line:
            orig_metrics['lpips'] = float(line.split(': ')[1])
        elif "Average 3D SSIM:" in line:
            orig_metrics['ssim3d'] = float(line.split(': ')[1])
            
    # Extract metrics from thresholded section
    thresh_metrics = {}
    for line in thresholded.split('\n'):
        if "Average L2 Loss:" in line:
            thresh_metrics['l2'] = float(line.split(': ')[1])
        elif "Average LPIPS Loss:" in line:
            thresh_metrics['lpips'] = float(line.split(': ')[1])
        elif "Average 3D SSIM:" in line:
            thresh_metrics['ssim3d'] = float(line.split(': ')[1])
            
    return orig_metrics, thresh_metrics

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
    
    orig_l2_list = []
    orig_lpips_list = []
    orig_ssim3d_list = []
    thresh_l2_list = []
    thresh_lpips_list = []
    thresh_ssim3d_list = []
    
    for filename in file_list:
        full_path = os.path.join(input_path, filename)
        orig_metrics, thresh_metrics = extract_metrics(full_path)
        
        orig_l2_list.append(orig_metrics['l2'])
        orig_lpips_list.append(orig_metrics['lpips'])
        orig_ssim3d_list.append(orig_metrics['ssim3d'])
        
        thresh_l2_list.append(thresh_metrics['l2'])
        thresh_lpips_list.append(thresh_metrics['lpips'])
        thresh_ssim3d_list.append(thresh_metrics['ssim3d'])
    
    print("\nOriginal Predictions Lists:")
    print(f"L2 Loss List: {orig_l2_list}")
    print(f"LPIPS Loss List: {orig_lpips_list}")
    print(f"3D SSIM List: {orig_ssim3d_list}")
    
    print("\nThresholded Predictions Lists:")
    print(f"L2 Loss List: {thresh_l2_list}")
    print(f"LPIPS Loss List: {thresh_lpips_list}")
    print(f"3D SSIM List: {thresh_ssim3d_list}")
    
    num_files = len(file_list)
    
    print("\nOriginal Predictions Average Metrics:")
    print(f"Average L2 Loss: {sum(orig_l2_list)/num_files:.8f}")
    print(f"Average LPIPS Loss: {sum(orig_lpips_list)/num_files:.8f}")
    print(f"Average 3D SSIM: {sum(orig_ssim3d_list)/num_files:.8f}")
    
    print("\nThresholded Predictions Average Metrics:")
    print(f"Average L2 Loss: {sum(thresh_l2_list)/num_files:.8f}")
    print(f"Average LPIPS Loss: {sum(thresh_lpips_list)/num_files:.8f}")
    print(f"Average 3D SSIM: {sum(thresh_ssim3d_list)/num_files:.8f}")

if __name__ == "__main__":
    main()
