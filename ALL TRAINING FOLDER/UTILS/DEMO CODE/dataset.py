import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from random import shuffle
import time
# import debugpy
# try:
#     debugpy.listen(("localhost", 5678))
#     debugpy.wait_for_client()
# except:
#     pass
RSCD_WITH_AUGUMENTATION = False

def augmenter(image):
    return transforms.RandomHorizontalFlip(p=0.5)(
        transforms.ColorJitter(contrast=0.25)(
            transforms.RandomAffine(
                0, translate=(0.03, 0.03))(image)))


def process_image(image):
    means = [0.485, 0.456, 0.406]
    inv_stds = [1/0.229, 1/0.224, 1/0.225]

    image = Image.fromarray(image)
    image = transforms.ToTensor()(image)
    for channel, mean, inv_std in zip(image, means, inv_stds):
        channel.sub_(mean).mul_(inv_std)
    return image

COCO_categories = ['airplane', 'apple', 'backpack', 'banana', 'baseball bat',
              'baseball glove', 'bear', 'bed', 'bench', 'bicycle', 'bird',
              'boat', 'book', 'bottle', 'bowl', 'broccoli', 'bus', 'cake',
              'car', 'carrot', 'cat', 'cell phone', 'chair', 'clock', 'couch',
              'cow', 'cup', 'dining table', 'dog', 'donut', 'elephant',
              'fire hydrant', 'fork', 'frisbee', 'giraffe', 'hair drier',
              'handbag', 'horse', 'hot dog', 'keyboard', 'kite', 'knife',
              'laptop', 'microwave', 'motorcycle', 'mouse', 'orange', 'oven',
              'parking meter', 'person', 'pizza', 'potted plant', 'refrigerator',
              'remote', 'sandwich', 'scissors', 'sheep', 'sink', 'skateboard',
              'skis', 'snowboard', 'spoon', 'sports ball', 'stop sign', 'suitcase',
              'surfboard', 'teddy bear', 'tennis racket', 'tie', 'toaster', 'toilet',
              'toothbrush', 'traffic light', 'train', 'truck', 'tv', 'umbrella', 'vase',
              'wine glass', 'zebra']
COCO_categories_sorted_by_freq = ['person', 'chair', 'car', 'dining table', 'cup',
                             'bottle', 'bowl', 'handbag', 'truck', 'backpack',
                             'bench', 'book', 'cell phone', 'sink', 'tv', 'couch',
                             'clock', 'knife', 'potted plant', 'dog', 'sports ball',
                             'traffic light', 'cat', 'bus', 'umbrella', 'tie', 'bed',
                             'fork', 'vase', 'skateboard', 'spoon', 'laptop',
                             'train', 'motorcycle', 'tennis racket', 'surfboard',
                             'toilet', 'bicycle', 'airplane', 'bird', 'skis', 'pizza',
                             'remote', 'boat', 'cake', 'horse', 'oven', 'baseball glove',
                             'baseball bat', 'giraffe', 'wine glass', 'refrigerator',
                             'sandwich', 'suitcase', 'kite', 'banana', 'elephant',
                             'frisbee', 'teddy bear', 'keyboard', 'cow', 'broccoli', 'zebra',
                             'mouse', 'orange', 'stop sign', 'fire hydrant', 'carrot',
                             'apple', 'snowboard', 'sheep', 'microwave', 'donut', 'hot dog',
                             'toothbrush', 'scissors', 'bear', 'parking meter', 'toaster',
                             'hair drier']
RSCD_categories = ['smooth', 'slight', 'severe', 'dry', 'wet',
              'water', 'fresh_snow', 'melted_snow', 'ice', 'asphalt', 'concrete',
              'gravel', 'mud']
RSCD_categories_sorted_by_freq = ['smooth', 'dry', 'asphalt', 'wet', 'concrete',
                                  'water', 'mud', 'gravel', 'slight', 'fresh_snow',
                                  'severe', 'melted_snow', 'ice']

categories_sorted_by_freq = dict((x, len(RSCD_categories) - count)
                                 for count, x in enumerate(RSCD_categories_sorted_by_freq))
category_dict_classification = dict((category, count) for count, category in enumerate(RSCD_categories))
category_dict_sequential = dict((category, count) for count, category in enumerate(RSCD_categories))
category_dict_sequential['<end>'] = len(RSCD_categories)
category_dict_sequential['<start>'] = len(RSCD_categories) + 1
category_dict_sequential['<pad>'] = len(RSCD_categories) + 2
category_dict_sequential_inv = dict((value, key)
                                    for key, value in category_dict_sequential.items())

class COCOMultiLabel(Dataset):
    def __init__(self, train, classification, image_path, sort_by_freq=False):
        super(COCOMultiLabel, self).__init__()
        self.train = train
        if self.train == True:
            self.coco_json = json.load(open('coco_train.json', 'r'))
            self.max_length = 18 + 2 # highest number of labels for one image in training
            self.image_path = image_path + '/train2014/'
        elif self.train == False:
            self.coco_json = json.load(open('coco_val.json', 'r'))
            self.max_length = 15 + 2
            self.image_path = image_path + '/val2014/'

        else:
            assert 0 == 1
        assert classification in [True, False]
        self.classification = classification
        self.fns = self.coco_json.keys()
        self.sort_by_freq = sort_by_freq
        if self.sort_by_freq:
            print('Sorting by frequency')

    def __len__(self):
        return len(self.coco_json)

    def __getitem__(self, idx):
        json_key = self.fns[idx]
        categories_batch = self.coco_json[json_key]['categories']
        image_fn = self.image_path + json_key

        image = Image.open(image_fn)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.train:
            try:
                image = augmenter(image)
            except IOError:
                print("augmentation error")
        transform=transforms.Compose([
                           transforms.Resize((288, 288)),
                           transforms.ToTensor(),
                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
                           ])
        try:
            image = transform(image)        
        except IOError:
            return None

        # labels
        labels_freq_indexes = [categories_sorted_by_freq[x] for x in categories_batch]
        labels = []
        labels_classification = np.zeros(len(COCO_categories), dtype=np.float32)
        labels.append(category_dict_sequential['<start>'])
        for category in categories_batch:
            labels.append(category_dict_sequential[category])
            labels_classification[category_dict_classification[category]] = 1

        if self.sort_by_freq:
            labels_new = [category_dict_sequential['<start>']]
            labels_new.extend([label
                               for _, label in sorted(zip(labels_freq_indexes, labels[1:]), reverse=True)])
            labels = labels_new[:]

        labels.append(category_dict_sequential['<end>'])
        for _ in range(self.max_length - len(categories_batch) - 1):
            labels.append(category_dict_sequential['<pad>'])

        labels = torch.LongTensor(labels)
        labels_classification = torch.from_numpy(labels_classification)
        label_number = len(categories_batch) + 2 # including the <start> and <end>

        if self.classification:
            return_tuple = (image, labels_classification)
        else:
            return_tuple = (image, labels, label_number, labels_classification)
        return return_tuple

class RSCDMultiLabel(Dataset):
    def __init__(self, train, classification, image_path, sort_by_freq=False):
        super(RSCDMultiLabel, self).__init__()        
        self.train = train
        if self.train == True:
            self.coco_json = json.load(open('/user/mmarseglia/second_split_60_30_10/recursive_model_paper_folder_for_second_split/orderless-rnn-classification-master/RSCD_train.json', 'r'))
            # # Estrai le prime 128*5 chiavi
            # keys_to_extract = list(self.coco_json.keys())[:128*10]

            # # Crea un nuovo dizionario con le chiavi estratte e i loro valori corrispondenti
            # self.coco_json = {key: self.coco_json[key] for key in keys_to_extract}
            self.max_length = 2 + 2 # highest number of labels for one image in training
            self.image_path = image_path + '/train/'
        elif self.train == False:
            self.coco_json = json.load(open('/user/mmarseglia/second_split_60_30_10/recursive_model_paper_folder_for_second_split/orderless-rnn-classification-master/RSCD_validation.json', 'r'))
            # # Estrai le prime 128*5 chiavi
            # keys_to_extract = list(self.coco_json.keys())[:128*10]

            # # Crea un nuovo dizionario con le chiavi estratte e i loro valori corrispondenti
            # self.coco_json = {key: self.coco_json[key] for key in keys_to_extract}
            self.max_length = 2 + 2
            self.image_path = image_path + '/validation/'

        else:
            assert 0 == 1
        assert classification in [True, False]
        self.classification = classification
        self.fns = list(self.coco_json.keys())
        self.sort_by_freq = sort_by_freq
        if self.sort_by_freq:
            print('Sorting by frequency')

    def __len__(self):
        return len(self.coco_json)

    def __getitem__(self, idx):
        json_key = self.fns[idx]
        categories_batch = self.coco_json[json_key]['categories']
        image_fn = self.image_path + json_key

        image = Image.open(image_fn)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.train and RSCD_WITH_AUGUMENTATION:
            try:
                image = augmenter(image)
            except IOError:
                print("augmentation error")
        transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize([0.498, 0.498, 0.498], [0.500, 0.500, 0.500]),
                           ])
        try:
            image = transform(image)        
        except IOError:
            return None

        # labels
        labels_freq_indexes = [categories_sorted_by_freq[x] for x in categories_batch]

        labels = []
        labels_classification = np.zeros(len(RSCD_categories), dtype=np.float32)
        labels.append(category_dict_sequential['<start>'])
        for category in categories_batch:
            labels.append(category_dict_sequential[category])
            labels_classification[category_dict_classification[category]] = 1

        if self.sort_by_freq:
            labels_new = [category_dict_sequential['<start>']]
            labels_new.extend([label
                               for _, label in sorted(zip(labels_freq_indexes, labels[1:]), reverse=True)])
            labels = labels_new[:]

        labels.append(category_dict_sequential['<end>'])
        for _ in range(self.max_length - len(categories_batch) - 1):
            labels.append(category_dict_sequential['<pad>'])

        labels = torch.LongTensor(labels)
        labels_classification = torch.from_numpy(labels_classification)
        label_number = len(categories_batch) + 2 # including the <start> and <end>

        if self.classification:
            return_tuple = (image, labels_classification)
        else:
            return_tuple = (image, labels, label_number, labels_classification)
        return return_tuple
    
class RSCDMultiLabel_test(Dataset):
    def __init__(self, train, classification, image_path, sort_by_freq=False, path_to_json=""):
        super(RSCDMultiLabel_test, self).__init__()        
        self.train = train
        self.coco_json = json.load(open(path_to_json, 'r'))
        self.max_length = 2 + 2
        self.image_path = image_path
        assert classification in [True, False]
        self.classification = classification
        self.fns = list(self.coco_json.keys())
        self.sort_by_freq = sort_by_freq
        if self.sort_by_freq:
            print('Sorting by frequency')

    def __len__(self):
        return len(self.coco_json)

    def __getitem__(self, idx):
        json_key = self.fns[idx]
        categories_batch = self.coco_json[json_key]['categories']
        image_fn = self.image_path + json_key

        image = Image.open(image_fn)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.train and RSCD_WITH_AUGUMENTATION:
            try:
                image = augmenter(image)
            except IOError:
                print("augmentation error")
        transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize([0.498, 0.498, 0.498], [0.500, 0.500, 0.500]),
                           ])
        try:
            image = transform(image)        
        except IOError:
            return None

        # labels
        labels_freq_indexes = [categories_sorted_by_freq[x] for x in categories_batch]

        labels = []
        labels_classification = np.zeros(len(RSCD_categories), dtype=np.float32)
        labels.append(category_dict_sequential['<start>'])
        for category in categories_batch:
            labels.append(category_dict_sequential[category])
            labels_classification[category_dict_classification[category]] = 1

        if self.sort_by_freq:
            labels_new = [category_dict_sequential['<start>']]
            labels_new.extend([label
                               for _, label in sorted(zip(labels_freq_indexes, labels[1:]), reverse=True)])
            labels = labels_new[:]

        labels.append(category_dict_sequential['<end>'])
        for _ in range(self.max_length - len(categories_batch) - 1):
            labels.append(category_dict_sequential['<pad>'])

        labels = torch.LongTensor(labels)
        labels_classification = torch.from_numpy(labels_classification)
        label_number = len(categories_batch) + 2 # including the <start> and <end>

        if self.classification:
            return_tuple = (image, labels_classification)
        else:
            return_tuple = (image, labels, label_number, labels_classification)
        return return_tuple