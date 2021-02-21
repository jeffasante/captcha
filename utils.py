from torchvision import transforms, datasets
import numpy as np

import glob
from PIL import Image, ImageOps
import os
import string


# Read data
def extractZipFiles(zip_file, extract_to):
    ''' Extract from zip '''
    with zipfile.ZipFile(zip_file, 'r')as zipped_ref:
        zipped_ref.extractall(extract_to)
    print('done')
    

data_dir = 'data/captcha_images_v2/*.png'

def findFiles(path): return glob.glob(path)

    # find letter inde from targets_flat
def letterToIndex(letter):
    return all_letters.find(letter)
# print(letterToIndex('l'))

# index to letter
indexToLetter = {letterToIndex(i):i for i in all_letters}


data = [img for img in findFiles(data_dir)]
targets = [os.path.basename(x)[:-4] for x in glob.glob(data_dir)]

# abcde -> [a, b, c, d, e]
pre_targets_flat = [[c for c in x] for x in targets]

encoded_targets = np.array([[letterToIndex(c) for c in x] for x in pre_targets_flat])

targets_flat = [char for word in pre_targets_flat for char in word]
unique_letters = set(char for word in targets for char in word)

    


class CaptchaDataset(Dataset):
    """
        Args:
            data (string): Path to the csv file with all the images.
            target (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
    
    def __init__(self, data, target=None, transform=None):
        self.data = data
        self.target = target
        self.transform = transform   

    def __getitem__(self, index):
        # read image
        x = Image.open(self.data[index]).convert('RGB')
        y = self.target[index]

        # resize, turn to 0,1
        if self.transform:
            x = self.transform(x)
        
#         x = np.array(x).astype(np.float32)
        return x, torch.tensor(y, dtype=torch.long)
#         x = torch.tensor(x, dtype=torch.float32).clone().detach() 
#         y = torch.tensor(y, dtype=torch.long)
       
        return x, y    
    
    def __len__(self):
        return len(self.data) 
    
