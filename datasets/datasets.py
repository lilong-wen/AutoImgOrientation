import random
from torch.utils.data import Dataset
import random
import glob
import cv2
from .rotation_utils import rotate_bound
from .rotation_utils import rotate


class CameraData(Dataset):

    def __init__(self, root, transform=None):

        self.root = root
        self.img_list = glob.glob(root + "*.png")
        random.shuffle(self.img_list)
        self.transform = transform

    def __len__(self):

        return len(self.img_list)

    def __getitem__(self, idx):

        angle_random = random.randint(20, 340)

        img = cv2.imread(self.img_list[idx])

        random_num = random.random()
        if random_num > 0.5:

            img_rotated = rotate_bound(img, angle_random)
        else:
            img_rotated = rotate(img, angle_random)

        if self.transform is not None:
            img_rotated = self.transform(img_rotated)

        return img_rotated, angle_random/360


class CameraData_offline(Dataset):

    def __init__(self, root, transform=None):

        self.root = root
        self.img_list = glob.glob(root + "*.png")
        random.shuffle(self.img_list)
        self.transform = transform

    def __len__(self):

        return len(self.img_list)

    def __getitem__(self, idx):

        angle_random = self.img_list[idx].split("_")[-1].split(".")[0]

        img = cv2.imread(self.img_list[idx])

        if self.transform is not None:
            img_rotated = self.transform(img_rotated)

        return img_rotated, angle_random/360
