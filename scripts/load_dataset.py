import json
from pathlib import Path
import os
from PIL import Image, ImageOps
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


# function to parse the json annotations file
## outputs: 
##   1. dictionary of label ids and their names
##      -> {category_id: 'name'}
##   2. dictionary of image data: unique image id, file name, height & width of image
##      -> {image_id: {'file_name': _, 'height': _, 'width': _}}
##   3. dictionary of ALL category ids of the labels for each image
##      -> {image_id: {'category_id': {_,_,_}}}
##      Note: for now, I made the category ids form a set, so that they're unordered and duplicates are omitted
def parse_annotations(label_path: str) -> tuple[dict, dict, dict]:
    with open(label_path, 'r') as file:
        label_file = json.load(file)

    label_dict = {}
    for item in label_file['categories']:
        label_dict[item['id']] = item['name']
    
    image_data = {}
    for image in label_file['images']:
        image_data[image['id']] = dict(list(image.items())[1:4])
    
    labels = {}
    for item in label_file['annotations']:
        if item['image_id'] in labels:
            labels[item['image_id']]['category_id'].add(item['category_id'])
        else:
            labels[item['image_id']] = {'category_id': {item['category_id']}}

    return label_dict, image_data, labels


# to start, I'm resizing all the images to match the largest image, 
#   so this function finds the largest dimension and returns a dictionary
# Note: I kept getting confused on which dimension should be which when passed to other fns., 
#       so I wrote a dictionary like this so I can easily change it later when calling those fns.
def get_max_dim(image_data: dict) -> dict:
    dim = {'width': 0, 'height': 0}
    for item in image_data.values():
        dim['width'] = max(dim['width'],item['width'])
        dim['height'] = max(dim['height'],item['height'])
    return dim


# image transformation for dataset: 
#   1. convert each image to grayscale, 
#   2. pad the image with black pixels to match the largest image's size,
#   3. convert the image to a tensor
# Ouput: tensor of shape [1, max_width, max_height]
class image_transf(object):
    def __init__(self, dim: tuple):
        self.dim = dim
        self.to_tensor = transforms.ToTensor()
    def __call__(self, img):
        img = img.convert('L')
        img = ImageOps.pad(img, self.dim, color=0)
        img = self.to_tensor(img).permute(0,2,1)
        return img


# label transformation for dataset
## Input: n_cats = no. of categories, id0 = first index of category ids
## Output: vector with length = no. of categories, 
##            ones at indices of each image's categories, zeros elsewhere
class OneHotEncoder(object):
    def __init__(self, n_cats: int, id0: int = 0):
        self.n_cats = n_cats
        self.id0 = id0
    def __call__(self, ids: set):
        vec = torch.zeros(self.n_cats, dtype=torch.float16)
        for id in ids:
            vec[id-self.id0] = 1.0
        return vec


# custom dataset class
class ntdDataset(Dataset):
    def __init__(self, img_dir, img_data, annotations, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.img_data = img_data
        self.img_labels = annotations
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_data[idx]['file_name'])
        image = Image.open(img_path)
        label = self.img_labels[idx]['category_id']
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


# function wrapper to set everything up, returns dataset object
def get_dataset(dir_path: str = None) -> Dataset:
    if dir_path is None:
        dir_path = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "train"
    
    # get all label and image data
    label_dict, image_data, labels = parse_annotations(dir_path / "labels.json")
    
    # parameters for transforms in dataset class
    dim = get_max_dim(image_data)
    dim = (dim['width'],dim['height'])   # hardcoding this by hand, as mentioned above
    n_cats = len(label_dict)             # number of categories for labels
    id0 = min(label_dict.keys())         # lowest index used in labels.json for the categories (just to be safe)

    # initialize transforms and dataset
    transf = image_transf(dim)
    t_transf = OneHotEncoder(n_cats=n_cats, id0=id0)
    dataset = ntdDataset(img_dir = dir_path / "data", img_data=image_data, annotations=labels, 
                         transform=transf, target_transform=t_transf)
    return dataset