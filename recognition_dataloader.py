import os
from os.path import join
import torch
import torch.utils.data as data
from torchvision.transforms import transforms
from PIL import Image

# Create a train folder in the root directory to hold the training data
# Create subfolders for each label under the train folder according to the label name
# Put the corresponding image data into the corresponding word folder.
################################################
# Set the directory structure
# -NUS Hand Posture Dataset
# ---train
# -----1
# -------g1.jpg
# -------....
# -----2
# -------g2.jpg
# -------....
# -----3
# -------g3.jpg
# -------....
# ...

#Define image pre-processing (data enhancement) for training, test and validation data
#We can modify the following parameters

# test_transform = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.Resize(45),
#     transforms.CenterCrop(32),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])

#define suffix for finding and reading images
#Had an error ValueError: num_samples should be a positive integer value, but got num_samples=0,
#because the suffix is wrong, can't read the image
#need to use () here, can't use [], can't be list data
FileNameEnd = ('.jpeg', '.JPEG', '.tif', '.jpg', '.png', '.bmp')

class RecognitionImageFolder(data.Dataset):
    def __init__(self, root, subdir='RecognitionData', transform=None):
        super(RecognitionImageFolder,self).__init__()

        self.transform = transform  #Define the type of image conversion
        self.image = []     #define the list to store images
        self.img_dir = join(root, subdir)
        print(len(os.listdir(self.img_dir)))
        self.label_dict = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4, 'f':5, 'g':6, 'h':7, 'i':8, 'j':9}
        
    def __len__(self):
        return len(os.listdir(self.img_dir))
    
    def __getitem__(self,idx):
        name = os.listdir(self.img_dir)[idx]

        label = self.label_dict[name[0]]

        path = os.path.join(self.img_dir, name)
        image = Image.open(path)
        if self.transform:
            image = self.transform(image)
        return image,label


