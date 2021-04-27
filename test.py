import torchvision.transforms as transforms
import datetime
import os
import time
import yaml
import cv2
import torch
from network.model import anglePrediction
import argparse
from datasets.rotation_utils import rotate_bound_black

checkpoints_folder = "./weights/"
# checkpoints_folder = "./runs/Apr26_21-34-49_t640/checkpoints/"

device = 'cuda' if torch.cuda.is_available() else "cpu"

input_shape = (448, 448)

def transform_pipeline():
        # get a set of data augmentation transformations as described in the SimCLR paper.
        color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        data_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Scale((input_shape[0], input_shape[1])),
            #transforms.RandomResizedCrop(size=self.input_shape[0], scale=(0.8, 1.0)),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomApply([color_jitter], p=0.8),
            #transforms.RandomGrayscale(p=0.2),
            # GaussianBlur(kernel_size=int(0.1 * self.input_shape[0])),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        return data_transforms

def test(img_path):

    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    model = anglePrediction(config["model_name"]).to(device)
    state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
    model.load_state_dict(state_dict)

    with torch.no_grad():
        model.eval()

        img = cv2.imread(img_path)
        # img = cv2.resize(img, (448, 448))
        img = transform_pipeline()(img)
        print(f'input image shape: {img.shape}')

        # img = torch.from_numpy(img).float().to(device)
        img = img.to(device)
        img = img.unsqueeze(0)
        # img = img.permute((0, 3, 1, 2))

        angle = model(img)

    return (angle.cpu().numpy()[0] * 360)

def time_measure(img_path):

    # start_time = datetime.datetime.now()
    start_time = time.time()
    angle = test(img_path)
    # end_time = datetime.datetime.now()
    end_time =time.time()

    print(f'predicted angle {angle}')
    print(f'used time {end_time - start_time}')


def rotation(img_path):

    angle = test(img_path)
    img = cv2.imread(img_path)
    result_img = rotate_bound_black(img, 360 - angle)
    print(f"angle: {angle}")
    return result_img

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--img', default='test.jpg')
    args = parser.parse_args()

    # time_measure(args.img)

    result = rotation(args.img)

    cv2.imwrite("result.png", result)
