import numpy as np
import pandas as pd

from torch.utils.data import DataLoader

import yaml
from feature_pooling.cnn_feature_pool import FeaturePoolingCNN
from dataset.patches_dataset import PatchesDataset

BASEPATH = '/workspace/other/bakterie/results/results_2020_05_24/'


def run(path, datasets, split, config_file):
    print(path, split)
    with open(config_file, 'r') as cf:
        config = yaml.load(cf)
    cnn = FeaturePoolingCNN(config)
    cnn.load_model(path)
    ds = PatchesDataset(config['patches_filename'], config['labels_filename'], datasets, split)
    bs = 64
    dl = DataLoader(ds, batch_size=bs, shuffle=False)
    patches = []
    count = 0
    for X, y, image_id, pos_x, pos_y in dl:
        print(f'{count}/{1430 if "train" in datasets else 356}', end=', ')
        count += 1
        y_hat = cnn.predict(X)
        y_pred = np.argmax(y_hat, 1)
        for i in range(len(y_hat)):
            patches.append([int(y[i]), y_pred[i], y_hat[i][0], y_hat[i][1], image_id[i], pos_x[i], pos_y[i]])
    results = pd.DataFrame(patches, columns=['y_true', 'y_pred', 'y_val0', 'y_val1', 'image_id', 'pos_x', 'pos_y'])

    del cnn
    save_results(results, path, datasets)


def save_results(results, path, datasets):
    filename = path.split('/')[-1].replace('pkl', 'csv')
    sets = '_'.join(datasets)
    results.to_csv(f'{BASEPATH}/{sets}_{filename}')


if __name__ == '__main__':
    config = '/workspace/other/bakterie/configs/feature_pooler/2020_05_21_config_1.yml'
    run('/workspace/other/bakterie/models/2020_05_21_model_1_2000.pkl', ['valid'], 'SPLIT_CV1', config)
    run('/workspace/other/bakterie/models/2020_05_21_model_1_2000.pkl', ['train'], 'SPLIT_CV1', config)
    run('/workspace/other/bakterie/models/2020_05_21_model_2_2000.pkl', ['valid'], 'SPLIT_CV2', config)
    run('/workspace/other/bakterie/models/2020_05_21_model_2_2000.pkl', ['train'], 'SPLIT_CV2', config)