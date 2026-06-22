
import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ImageDataset(Dataset):
    """
    Custom mask classification dataset, inheriting from torch.utils.data.Dataset.
    Responsible for reading image paths, corresponding class labels, and performing data transforms at runtime.
    """
    def __init__(self, dataset_root, train, transform=None): 

        if train:
            train_or_test_path = Path(dataset_root) / "train"
        else:
            train_or_test_path = Path(dataset_root) / "test"

        self.classes = []
        if train_or_test_path.exists():
            self.classes = sorted([d for d in os.listdir(train_or_test_path) if (train_or_test_path / d).is_dir()]) 
            
        self.paths = [f for f in train_or_test_path.rglob("*") if f.is_file()] 
        self.transform = transform 

    def __getitem__(self, index):
        """
        Read a single image based on index and return the transformed tensor and class index.
        """
        img = Image.open(self.paths[index]).convert("RGB")
        class_name = self.paths[index].parent.name
        class_idx = self.classes.index(class_name)

        if self.transform:
            return self.transform(img), class_idx
        else:
            return img, class_idx

    def __len__(self):
        """
        Return the total length of the dataset (total number of images).
        """
        return len(self.paths)

train_transform = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.TrivialAugmentWide(),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.ToTensor()
])