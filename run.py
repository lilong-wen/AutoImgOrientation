import argparse
from train import Prediction
import yaml
from datasets.dataset_wrapper import DataSetWrapper


def main():

    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)

    dataset = DataSetWrapper(config['batch_size'], **config['dataset'])

    predict = Prediction(dataset, config)
    predict.train()


if __name__ == "__main__":

    main()
