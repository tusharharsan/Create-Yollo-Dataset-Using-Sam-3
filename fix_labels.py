import os
from glob import glob
from tqdm import tqdm

def fix_labels(dataset_dir):
    label_files = glob(os.path.join(dataset_dir, "**", "*.txt"), recursive=True)
    print(f"Found {len(label_files)} label files.")
    
    modified_count = 0
    for file_path in tqdm(label_files):
        with open(file_path, "r") as f:
            lines = f.readlines()
        
        new_lines = []
        modified = False
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            
            class_id = parts[0]
            # Remap class 2 to class 0
            if class_id == "2":
                parts[0] = "0"
                modified = True
            
            new_lines.append(" ".join(parts) + "\n")
        
        if modified:
            with open(file_path, "w") as f:
                f.writelines(new_lines)
            modified_count += 1
            
    print(f"Fixed {modified_count} files.")

if __name__ == "__main__":
    fix_labels("datasets/sack-counting-2")
