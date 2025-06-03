## cd /raid/home/CAMCA/ss3112/Github/mae/datasets
## Usage: python copy_txt.py /raid/home/CAMCA/ss3112/Github/mae/

import os
import shutil
import sys

def copy_txt_files(src_dir, dst_dir):
    # Create destination directory if it doesn't exist
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
        
    # Walk through source directory
    for root, dirs, files in os.walk(src_dir):
        # Filter for .txt files that don't start with .
        txt_files = [f for f in files if f.endswith('.txt') and not f.startswith('.')]
        
        if txt_files:
            # Create corresponding subdirectory structure in destination
            rel_path = os.path.relpath(root, src_dir)
            dst_subdir = os.path.join(dst_dir, rel_path)
            if not os.path.exists(dst_subdir):
                os.makedirs(dst_subdir)
                
            # Copy each txt file
            for txt_file in txt_files:
                src_file = os.path.join(root, txt_file)
                dst_file = os.path.join(dst_subdir, txt_file)
                shutil.copy2(src_file, dst_file)

if __name__ == '__main__':
    # Get current directory from command line argument
    current_dir = sys.argv[1]
    
    # Define source and destination paths
    result_dir = os.path.join(current_dir, 'Results')
    result_txt_dir = os.path.join(current_dir, 'Results_txt')
    
    # Copy txt files preserving directory structure
    copy_txt_files(result_dir, result_txt_dir)
