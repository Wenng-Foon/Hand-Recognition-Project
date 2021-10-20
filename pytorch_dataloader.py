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
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(32),
    transforms.RandomRotation(20),
    transforms.Resize((180, 160)),
    transforms.RandomAffine(20, translate=(10, 20))
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

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

class ImageFolder(data.Dataset):
    def __init__(self, root, subdir='train', transform=None):
        super(ImageFolder,self).__init__()

        self.transform = transform  #Define the type of image conversion
        self.image = []     #define the list to store images


        train_dir = join(root, 'train')

        self.class_names = sorted(os.listdir(train_dir))
        self.names2index = {v: k for k, v in enumerate(self.class_names)}
        for label in self.class_names:
            d = join(root, subdir, label)

            for directory, _, names in os.walk(d):
                for name in names:
                    filename = join(directory, name)
                    if filename.endswith(FileNameEnd):
                        self.image.append((filename, self.names2index[label]))


    def __getitem__(self, item):
        path, label = self.image[item]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.image)


if __name__ == '__main__':
    data_dir = '/Users/xxx/Downloads/NUS Hand Posture Dataset'
    train_data = ImageFolder(data_dir, subdir='train', transform=transform)
    print(train_data.class_names)

