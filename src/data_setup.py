
import os
import shutil
import zipfile
import gdown
from sklearn.model_selection import train_test_split

def download_and_prepare_data():
    """
    Download, extract, and organize the Face Mask Dataset.
    Includes removing __MACOSX and .DS_Store files, and automatically splitting into 80% training set and 20% test set.
    """
    dataset_zip = "Face_Mask_Dataset.zip"
    dataset_dir = "Face_Mask_Dataset"

    if not os.path.exists(dataset_zip):
        print("Downloading dataset from Google Drive...")
        url = "https://drive.google.com/uc?export=download&id=1-M7ki8zY2c-e0_lt1T80ALvbhHvefp2-"
        gdown.download(url, dataset_zip, quiet=False)
    else:
        print("Dataset zip file already exists, skipping download.")

    if not os.path.exists(dataset_dir):
        print("Extracting dataset...")
        with zipfile.ZipFile(dataset_zip) as zip_file:
            for file in zip_file.namelist():
                if not file.startswith("__MACOSX"):
                    zip_file.extract(file, dataset_dir)
    
    inner_dir_path = os.path.join(dataset_dir, dataset_dir)
    if os.path.exists(inner_dir_path):
        print("Restructuring folders...")
        for mask_class in os.listdir(inner_dir_path):
            src_path = os.path.join(inner_dir_path, mask_class)                
            dst_path = os.path.join(dataset_dir, mask_class)
            shutil.move(src_path, dst_path)
        shutil.rmtree(inner_dir_path)

    if not os.path.exists(os.path.join(dataset_dir, "train")):
        print("Splitting into training and test sets (80% / 20%)...")

        classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
        
        for mask_class in classes:
            original_mask_class_path = os.path.join(dataset_dir, mask_class) 
            train_path = os.path.join(dataset_dir, "train", mask_class)
            test_path = os.path.join(dataset_dir, "test", mask_class)

            os.makedirs(train_path, exist_ok=True)
            os.makedirs(test_path, exist_ok=True)

            files = [f for f in os.listdir(original_mask_class_path) if f != '.DS_Store']
            
            if files:
                train_files, test_files = train_test_split(files, test_size=0.2, random_state=1)

                for file in train_files:
                    src_path = os.path.join(original_mask_class_path, file)
                    dst_path = os.path.join(train_path, file)                    
                    shutil.move(src_path, dst_path)

                for file in test_files:
                    src_path = os.path.join(original_mask_class_path, file)
                    dst_path = os.path.join(test_path, file)
                    shutil.move(src_path, dst_path)
            
            shutil.rmtree(original_mask_class_path)

    print("Cleaning up .DS_Store system files...")
    for root, dirs, files in os.walk(dataset_dir): 
            if file == '.DS_Store':
                file_path = os.path.join(root, file)
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
    
    print("Dataset preparation complete!")

if __name__ == "__main__":
    download_and_prepare_data()