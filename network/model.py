import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F


class anglePrediction(nn.Module):

    def __init__(self, res_name):

        super(anglePrediction, self).__init__()
        self.resnet_dict = {
            "resnet18": models.resnet18(pretrained=True),
            "resnet50": models.resnet50(pretrained=True)
        }

        resnet = self._get_res_basemodel(res_name)
        num_features = resnet.fc.in_features
        self.res_features = nn.Sequential(*list(resnet.children())[:-1])
        # print(f"num_features: {num_features}")
        self.res_l1 = nn.Linear(num_features, num_features)
        self.res_l2 = nn.Linear(num_features, 1)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()


    def _get_res_basemodel(self, res_model_name):

        try:
            res_model = self.resnet_dict[res_model_name]
            print(f"base model: {res_model_name}")
            return res_model
        except:
            raise("Invalid model name")

    def forward(self, img):

        x1 = self.res_features(img)
        x1 = x1.squeeze()

        x2 = self.res_l1(x1)
        x2 = F.relu(x2)

        x3 = self.res_l2(x2)

        result = self.sigmoid(x3)
        #print(result)
        return result
