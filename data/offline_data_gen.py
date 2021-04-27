import cv2
import random
import glob
import sys

sys.path.append("../")
from datasets.rotation_utils import rotate_bound
from datasets.rotation_utils import rotate


data_root = './camera/'

def gen(num):

    img_list = glob.glob(data_root + "*.png")
    random.shuffle(img_list)

    count = 0
    while num > 0:

        for img_item in img_list:

            img = cv2.imread(img_item)
            img = cv2.resize(img, (448, 448))
            angle_random = random.randint(10, 350)
            random_num = random.random()
            if random_num > 0.5:
                img_rotated = rotate_bound(img, angle_random)
            else:
                img_rotated = rotate(img, angle_random)

            count = count + 1
            print(count)

            cv2.imwrite("offline/" + \
                        img_item.split("/")[-1].split(".")[0] + \
                        "_" + str(count) + "_" + str(angle_random) + ".png", img_rotated)
        num = num - 1


if __name__ == "__main__":

    gen(1000)
