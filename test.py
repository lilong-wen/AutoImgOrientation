import datetime
import os
import yaml
import cv2
import torch
from network.model import anglePrediction
import argparse

checkpoints_folder = "./runs/Apr22_23-09-13_t640/checkpoints/"

device = 'cuda' if torch.cuda.is_available() else "cpu"

def test(img_path):

    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    model = anglePrediction(config["model_name"]).to(device)
    state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
    model.load_state_dict(state_dict)

    with torch.no_grad():
        model.eval()

        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        print(f'input image shape: {img.shape}')

        img = torch.from_numpy(img).float().to(device)
        img = img.unsqueeze(0)
        img = img.permute((0, 3, 1, 2))

        angle = model(img)

    return (angle.cpu().numpy()[0] * 360)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--img', default='test.jpg')
    args = parser.parse_args()

    start_time = datetime.datetime.now()
    angle = test(args.img)
    end_time = datetime.datetime.now()

    print(f'predicted angle {angle}')
    print(f'used time {end_time - start_time}')
