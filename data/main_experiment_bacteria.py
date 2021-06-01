from __future__ import print_function

import datetime
import os
import sys

import numpy as np
import pandas as pd
import yaml


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler

from experiments import experiment
from tensorboardX import SummaryWriter

from munch import munchify
from dataset.bacteria_bag import BacteriaBag


kwargs = {'num_workers': 0, 'pin_memory': True}


def load_bacteria_cv(labels, patches, features, cv, curr_class, shuffle):
    train_dataset = BacteriaBag(labels,
                                patches,
                                features,
                                cv, curr_class, 'train', shuffle)
    valid_dataset = BacteriaBag(labels,
                                patches,
                                features,
                                cv, curr_class, 'valid', shuffle)
    test_dataset = BacteriaBag(labels,
                                patches,
                                features,
                                cv, curr_class, 'test', shuffle)
    return train_dataset, valid_dataset, test_dataset


def run(config, kwargs):
    config['model_signature'] = str(datetime.datetime.now())[0:19]

    model_name = '' + config['model_name']

    if config['loc_gauss'] or config['loc_inv_q'] or config['loc_att']:
        config['loc_info'] = True

    if config['att_gauss_abnormal'] or config['att_inv_q_abnormal'] or config['att_gauss_spatial'] or config['att_inv_q_spatial'] or \
            config['att_module']:
        config['self_att'] = True

    print(config)

    with open('experiment_log_' + config['operator'] + '.txt', 'a') as f:
        print(config, file=f)

    # IMPORT MODEL======================================================================================================
    from models.CNN import CNN as Model

    # START KFOLDS======================================================================================================
    print('\nSTART KFOLDS CROSS VALIDATION\n')

    train_error_folds = []
    test_error_folds = []
    labels = pd.read_csv(config['labels_filename'], index_col=0)
    patches = pd.read_csv(config['patches_filename'], index_col=0)
    features = torch.load(config['features_filename'])
    curr_class = config['curr_class']
    for current_fold in range(5):

        print('#################### Train-Test fold: {}/{} ####################'.format(current_fold + 1, config['kfold']))

        # DIRECTORY FOR SAVING==========================================================================================
        snapshots_path = 'snapshots/'
        dir = snapshots_path + model_name + '_' + config['model_signature'] + '/'
        sw = SummaryWriter(f"tensorboard/{model_name}_{config['model_signature']}_fold_{current_fold}")

        if not os.path.exists(dir):
            os.makedirs(dir)

        # LOAD DATA=====================================================================================================
        # train_fold, val_fold = kfold_indices_warwick(len(train_folds[current_fold - 1]), args.kfold_val, seed=args.seed)
        # train_fold = [train_folds[current_fold - 1][i] for i in train_fold]
        # val_fold = [train_folds[current_fold - 1][i] for i in val_fold]
        # loc = True if args.loc_info or args.out_loc else False
        train_set, val_set, test_set = load_bacteria_cv(labels, patches, features, config['split']+str(current_fold), curr_class, shuffle=True)
        clss, counts = np.unique(train_set.label_list, return_counts=True)
        counts = 1 - counts / np.sum(counts)
        class_counts = {int(clss[c]): counts[c] for c in range(len(clss))}
        train_sampleweights = [class_counts[int(y_bi)] for y_bi in train_set.label_list]
        sampler = WeightedRandomSampler(
            weights=train_sampleweights,
            num_samples=len(train_sampleweights),
        )

        # CREATE MODEL==================================================================================================
        print('\tcreate models')
        args = munchify(config)
        args.activation = nn.ReLU()
        model = Model(args)
        model.cuda(config['device'])

        # INIT OPTIMIZER================================================================================================
        print('\tinit optimizer')
        if config['optimizer'] == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.999), weight_decay=config['reg'])
        elif config['optimizer'] == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=config['lr'], weight_decay=config['reg'], momentum=0.9)
        else:
            raise Exception('Wrong name of the optimizer!')

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)

        # PERFORM EXPERIMENT============================================================================================
        print('\tperform experiment\n')

        train_error, test_error = experiment(
            args,
            kwargs,
            current_fold,
            train_set,
            val_set,
            test_set,
            sampler,
            model,
            optimizer,
            scheduler,
            dir,
            sw,
        )

        # APPEND FOLD RESULTS===========================================================================================
        train_error_folds.append(train_error)
        test_error_folds.append(test_error)

        with open('final_results_' + config['operator'] + '.txt', 'a') as f:
            print('Class: {}\n'
                  'RESULT FOR A SINGLE FOLD\n'
                  'SEED: {}\n'
                  'OPERATOR: {}\n'
                  'FOLD: {}\n'
                  'ERROR (TRAIN): {}\n'
                  'ERROR (TEST): {}\n\n'.format(curr_class, config['seed'], config['operator'], current_fold, train_error, test_error),
                  file=f)
    # ======================================================================================================================
    print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')
    with open('experiment_log_' + config['operator'] + '.txt', 'a') as f:
        print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n', file=f)

    return np.mean(train_error_folds), np.std(train_error_folds), np.mean(test_error_folds), np.std(test_error_folds)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


if __name__ == "__main__":
    with open(sys.argv[1], 'r') as config_file:
        config = yaml.load(config_file)
    # seeds = [71, 79, 53, 32, 98]
    seeds = [71]

    train_mean_list = []
    test_mean_list = []

    for seed in seeds:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        config['seed'] = seed
        train_mean, train_std, test_mean, test_std = run(config, kwargs)

        with open('final_results_' + config['operator'] + '.txt', 'a') as f:
            print('RESULT FOR A SINGLE SEED, 5 FOLDS\n'
                  'SEED: {}\n'
                  'OPERATOR: {}\n'
                  'TRAIN MEAN {} AND STD {}\n'
                  'TEST MEAN {} AND STD {}\n\n'.format(seed, config['operator'], train_mean, train_std, test_mean, test_std),
                  file=f)

        train_mean_list.append(train_mean)
        test_mean_list.append(test_mean)

    with open('final_results_' + config['operator'] + '.txt', 'a') as f:
        print('RESULT FOR 1 SEEDS, 5 FOLDS\n'
              'OPERATOR: {}\n'
              'TRAIN MEAN {} AND STD {}\n'
              'TEST MEAN {} AND STD {}\n\n'.format(config['operator'], np.mean(train_mean_list), np.std(train_mean_list), np.mean(test_mean_list), np.std(test_mean_list)),
              file=f)

    with open('final_results_' + config['operator'] + '.txt', 'a') as f:
        print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n', file=f)

# # # # # # # # # # #
# END EXPERIMENTS # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # #
