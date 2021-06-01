"""Loads the test set two times: for testing and for visualization. For every image z's are computed. All z's are
scaled. Each patch is multiplied with it's correcponding z."""

from __future__ import print_function

from pathlib import Path
import numpy as np
import pandas as pd
import yaml

from skimage import io
from skimage.exposure import rescale_intensity
from sklearn.metrics import confusion_matrix as cm

import torch
import torch.utils.data as data_utils
import torchvision.transforms as transforms
from torch.autograd import Variable

from dataset.bacteria_bag import BacteriaBagWithCoord, BacteriaBag

to_pil = transforms.Compose([transforms.ToPILImage()])

kwargs = {'num_workers': 0, 'pin_memory': True}


def run_predictions(config):
    labels = pd.read_csv(config['labels_filename'], index_col=0)
    patches = pd.read_csv(config['patches_filename'], index_col=0)
    features = torch.load(config['features_filename'])
    for current_fold in range(5):
        print('fold:' + str(current_fold))
        for current_set in config['dataset']:
            # Load datasets
            data_set = BacteriaBag(labels,
                                patches,
                                features,
                                f'SPLIT_CV{current_fold}', config['curr_class'], current_set, shuffle=False)

            data_loader = torch.utils.data.DataLoader(data_set, batch_size=1, shuffle=False, **kwargs)

            # Get all the data
            whole_imgs = []
            whole_labels = []
            for batch_idx, (data_whole, labels_whole) in enumerate(data_loader):
                data_whole = data_whole[0].squeeze(0)
                data_whole = data_whole.numpy()
                whole_imgs.append(data_whole)
                whole_labels.append(labels_whole)

            # Load model and set to eval mode
            trained_model_path = config['model_path'].replace('fold_0', f'fold_{current_fold}')
            model = torch.load(trained_model_path)
            model.eval()
            y_true_list = []
            y_pred_list = []
            # Predict
            for batch_idx, inputs in enumerate(data_loader):
                data, target = inputs
                print(batch_idx)
                target = target[0]
                data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)

                y_prob, y_hat, z, A, _, __ = model.forward(data)
                y_true_list.append(target.cpu().data.numpy())
                y_pred_list.append(y_hat.cpu().data.numpy()[0][0])

            curr_cm = cm(y_true_list, y_pred_list)
            pd.DataFrame(curr_cm).to_csv(
                f'{config["result_path"]}{current_set}_{current_fold}.csv')


def draw_whole_image(config):
    labels = pd.read_csv(config['labels_filename'], index_col=0)
    patches = pd.read_csv(config['patches_filename'], index_col=0)
    paths = patches[patches['filter_std'] == 1]['path'].values
    features = torch.load(config['features_filename'])
    for current_fold in range(5):
        print('fold:' + str(current_fold))
        for current_set in config['dataset']:
            # Load datasets
            data_set = BacteriaBagWithCoord(labels,
                                            patches,
                                            features,
                                            f'SPLIT_CV{current_fold}', config['curr_class'], current_set, shuffle=False)

            data_loader = torch.utils.data.DataLoader(data_set, batch_size=1, shuffle=False, **kwargs)

            # Get all the data
            whole_imgs = []
            whole_labels = []
            whole_coordinates = []
            for batch_idx, (data_whole, labels_whole, coordinates_whole) in enumerate(data_loader):
                data_whole = data_whole[0].squeeze(0)
                data_whole = data_whole.numpy()
                whole_imgs.append(data_whole)
                whole_labels.append(labels_whole)
                whole_coordinates.append(coordinates_whole.squeeze(0).numpy())

            # Load model and set to eval mode
            trained_model_path = config['model_path'].replace('fold_0', f'fold_{current_fold}')
            model = torch.load(trained_model_path)
            model.eval()
            y_true_list = []
            y_pred_list = []
            # Predict
            for batch_idx, (data, target, _) in enumerate(data_loader):
                print(batch_idx)
                target = target[0]
                data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)

                y_prob, y_hat, z, A, _, __ = model.forward(data)
                y_true_list.append(target.cpu().data.numpy())
                y_pred_list.append(y_hat.cpu().data.numpy()[0][0])

                target_cpu = target.cpu().data.numpy()
                y_hat_cpu = y_hat.cpu().data.numpy()[0][0]

                z_numpy = z.cpu().data.numpy()
                z_norm = (z_numpy - np.min(z_numpy)) / (np.max(z_numpy) - np.min(z_numpy))

                ### PLOT
                path = data_set.get_paths(batch_idx)[0]
                clone_graft_img = '_'.join(path.split('/')[-4:-1])
                # All cells create and apply mask
                xs = np.array(whole_coordinates[batch_idx][:, 0])
                ys = np.array(whole_coordinates[batch_idx][:, 1])
                image = np.zeros((xs.max(), ys.max(), 3))

                z_norm = z_norm.tolist()
                z_norm = np.array(z_norm)[0]  # use to visualize

                count = 0
                for x in range(0, max(xs), 125):
                    for y in range(0, max(ys), 125):
                        path_tmp = f"{'/'.join(path.split('/')[:-1])}/{x}_{y}.png"
                        try:
                            patch = io.imread(path_tmp)
                            if path_tmp in paths and count < len(whole_coordinates[batch_idx]) - 1:
                                image[x:x+125, y:y+125] = patch[:125, :125] * z_norm[count]
                                count += 1
                            else:
                                image[x:x+125, y:y+125] = patch[:125, :125] * 0.05
                        except FileNotFoundError:
                            image[x:x+125, y:y+125] = 0
                image = rescale_intensity(image, out_range=(0, 255)).astype(np.uint8)
                Path(config['image_dir']).mkdir(parents=True, exist_ok=True)
                io.imsave(f'/{config["image_dir"]}/{target_cpu}_{int(y_hat_cpu)}_{clone_graft_img}.png', image)


if __name__ == '__main__':
    with open('/workspace/eeml/configs/experiment_main/exp_config_2020_05_13_AD_class_A.yml', 'r') as config_file:
        config = yaml.load(config_file)
    if config['action'] == 'visualize':
        draw_whole_image(config)
    elif config['action'] == 'predict':
        run_predictions(config)