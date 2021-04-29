from decimal import Decimal
import shutil
import numpy as np
import torch
import os
from network.model import anglePrediction
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class Prediction(object):

    def __init__(self, dataset, config):

        self.config = config
        self.device = self._get_device()
        self.dataset = dataset
        self.writer = SummaryWriter()

    def _get_device(self):

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)
        return device

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('./runs',
                                              self.config['fine_tune_from'],
                                              'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder,
                                                 'model.pth'))
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def train(self):

        train_loader, valid_loader = self.dataset.get_data_loader()

        # model = anglePrediction(self.config['model_name']).to(self.device)
        model = anglePrediction(self.config['model_name'])
        model = torch.nn.DataParallel(model)
        model = model.to(self.device)

        # print(model)

        # model = self._load_pre_trained_weights(model)

        optimizer = torch.optim.Adam(model.parameters(),
                                     eval(self.config['learning_rate']),
                                     weight_decay=eval(self.config['weight_decay']))

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=len(train_loader),
                                                               eta_min=0,
                                                               last_epoch=-1)
        self.loss_fn = torch.nn.MSELoss()
        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        self. _save_config_file(model_checkpoints_folder)
        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        print('Training...')

        for epoch_counter in range(self.config['epochs']):
            print(f'Epoch {epoch_counter}')
            for x_img, y_angle in tqdm(train_loader):

                optimizer.zero_grad()

                x_img = x_img.to(self.device)
                y_angle = y_angle.to(self.device).float()

                pre_angle = model(x_img).squeeze(1)  # [N,C]

                loss = self.loss_fn(pre_angle, y_angle)


                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)
                else:
                    continue

                loss.backward()

                optimizer.step()

                n_iter += 1

            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                valid_loss = self._validate(model, valid_loader, n_iter)
                if valid_loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = valid_loss
                    torch.save(model.state_dict(),
                               os.path.join(model_checkpoints_folder, 'model.pth'))
                self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                scheduler.step()
            self.writer.add_scalar('cosine_lr_decay', scheduler.get_lr()[0], global_step=n_iter)

    def _validate(self, model, valid_loader, n_iter):

        # validation steps
        with torch.no_grad():
            model.eval()
            # model_bert.eval()
            valid_loss = 0.0
            counter = 0
            acc = 0
            print('Validation step')
            for x_img, y_angle in tqdm(valid_loader):

                x_img = x_img.to(self.device)
                y_angle = y_angle.to(self.device)

                pred_angle = model(x_img).squeeze(1)  # [N,C]
                loss = self.loss_fn(pred_angle, y_angle)
                #Decimal(a).quantize(Decimal("0.00"))
                y_angle_round = [float(Decimal(item.item()).quantize(Decimal("0.000"))) for item in y_angle.cpu().numpy()]
                pred_angle_round = [float(Decimal(item.item()).quantize(Decimal("0.000"))) for item in pred_angle.cpu().numpy()]
                print('True angles:')
                print(y_angle_round)
                print('Predcted angles:')
                print(pred_angle_round)
                val_acc = abs(np.array(y_angle_round) - np.array(pred_angle_round)).sum()
                acc += val_acc

                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter
            print(f"average distance is {(acc /(self.config['batch_size']*counter))}")
        model.train()

        return valid_loss

    def _save_config_file(self, model_checkpoints_folder):
        if not os.path.exists(model_checkpoints_folder):
            os.makedirs(model_checkpoints_folder)
            shutil.copy('./config.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))
