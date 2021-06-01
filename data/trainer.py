import sys
import argparse

import torch
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix, accuracy_score
from tqdm import tqdm

from dataset.patches_dataset import PatchesDataset

TB_DIR = 'tensorboard/'


import inspect

def fun(a,b,c=0,e=0, **other):
    print(a+b+c+e)
    print(other)

args = {'a':1, 'b':2, 'c':3, 'd':5}
# funargs = {k: v for k, v in args.items() if k in inspect.getfullargspec(fun)[0]}
# fun(19,**funargs)
fun(**args)


def parse_args():
    parser = argparse.ArgumentParser(description='Train feature_pooling cnn model')
    parser.add_argument('-tensorboard_path', type=str, default='tensorboard/', required=False)
    parser.add_argument('-source', help="source path with raw .tif images in subdirectories", type=str, required=True)
    parser.add_argument('-destination', help="path or folder name for patches", type=str, required=False)
    parser.add_argument('-patch_size', help="patch width and height", type=int, default=224, required=False)
    return parser.parse_args()


def norm_cm(arr):
    sums = arr.sum(axis=1)
    arr = arr / sums[:, np.newaxis]
    arr = arr.round(2)
    return arr


def trash(self, config, ):
    self.device = config['device'] if config['device'] is not None else torch.device('cuda')
    self.output_dim = config['output_dim']
    self.class_name = config['class_name']
    self.learning_rate = config['learning_rate']
    self.loss = get_loss(config['loss_function'])
    self.tb_writer = SummaryWriter(log_dir=os.path.join(TB_DIR, config['model_name']))
    self.model_name = config['model_name']
    self.model_path = config['model_path']
    self.model = self.prepare_model()
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
    self.scheduler = StepLR(self.optimizer, step_size=config['lr_step'], gamma=config['lr_gamma'])
    self.dataloaders = {}
    self.dataiters = {}
    self.tb_writer.add_scalar(f'Learning_rate', self.learning_rate, 0)
    config_file = sys.argv[1]
    config = read_config(config_file)
    model = FeaturePoolingCNN(config)
    train_dl = DataLoader(
        PatchesDataset(config['patches_filename'], config['labels_filename'], ['train'], config['split']),
        batch_size=config['batch_size'], shuffle=True)
    test_dl = DataLoader(
        PatchesDataset(config['patches_filename'], config['labels_filename'], ['valid'], config['split']),
        batch_size=config['batch_size'], shuffle=True)
    model.train(train_dl, test_dl, 20000, 100)

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X = X.to(self.device)
        y_hat = self.model(X).cpu().detach().numpy()
        del X
        return y_hat

    def init_data(self, train_dataloader, test_dataloader):
        self.dataloaders = {'train': train_dataloader, 'test': test_dataloader}
        self.dataiters = {'train': iter(train_dataloader), 'test': iter(test_dataloader)}

    def get_next_batch(self, phase):
        try:
            return next(self.dataiters[phase])
        except StopIteration:
            self.dataiters[phase] = iter(self.dataloaders[phase])
            return next(self.dataiters[phase])

    for step in tqdm(range(total_steps)):
        for phase in ['train', 'test']:
            X, y, _, _, _ = self.get_next_batch(phase)
            if phase == 'train':
                self.model.train()
            else:
                self.model.eval()

            inputs = X.to(self.device)
            labels = y.to(self.device)

            self.optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.loss(outputs, labels.long().reshape(outputs.shape[0]))

                if phase == 'train':
                    loss.backward()
                    self.optimizer.step()

            y_vals = y.cpu().detach().numpy()
            preds_vals = preds.cpu().detach().numpy()
            cm = norm_cm(confusion_matrix(y_vals, preds_vals))
            acc = accuracy_score(y_vals, preds_vals)

            self.tb_writer.add_scalar(f'Loss/{phase}', loss.item(), step)
            self.tb_writer.add_scalar(f'Acc/{phase}', acc, step)
            for d in range(len(self.class_name)):
                try:
                    self.tb_writer.add_scalar(f'{phase}/{self.class_name[d]}', cm[d, d], step)
                except IndexError:
                    self.tb_writer.add_scalar(f'{phase}/{self.class_name[d]}', 0, step)

            if step % save_step == 0:
                self.save_model(step)

        self.tb_writer.add_scalar(f'Learning_rate', self.optimizer.param_groups[0]['lr'], step)
        self.scheduler.step()



