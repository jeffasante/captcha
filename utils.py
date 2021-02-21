from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import numpy as np

import glob
from PIL import Image, ImageOps
import os
import string


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
    
    
    # find letter inde from targets_flat
def letterToIndex(letter):
    return all_letters.find(letter)
print(letterToIndex('l'))

# index to letter
indexToLetter = {letterToIndex(i):i for i in all_letters}


data_dir = 'data/captcha_images_v2/*.png'

def findFiles(path): return glob.glob(path)


data = [img for img in findFiles(data_dir)]
targets = [os.path.basename(x)[:-4] for x in glob.glob(data_dir)]

# abcde -> [a, b, c, d, e]
pre_targets_flat = [[c for c in x] for x in targets]

encoded_targets = np.array([[letterToIndex(c) for c in x] for x in pre_targets_flat])

targets_flat = [char for word in pre_targets_flat for char in word]
unique_letters = set(char for word in targets for char in word)


# define transforms
transform = transforms.Compose([transforms.Resize((75, 300)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.73199,), (0.28809,)),
                                ])

# build partion -- train test split
n_data = len(data)
train_size = int(0.9 * n_data)
test_size = n_data - train_size 

full_dataset = CaptchaDataset(data, encoded_targets, transform=transform)

train_dataset, val_dataset = torch.utils.data.random_split(full_dataset,
                                                           [train_size, test_size])

batch_size = 16

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')