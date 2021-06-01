import sys
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from bacteria.bacteria_utils import read_config
from bacteria.feature_pooling.cnn_feature_pool import FeaturePoolingCNN
from bacteria.feature_pooling.patches_dataset import PatchesDataset


features_hook = torch.Tensor()


def assign_hook(module, input, output):
    global features_hook
    features_hook = output


class FeatureExtraction:

    def __init__(self, config):
        self.device = config['device'] if config['device'] is not None else torch.device('cuda')
        self.save_name = config['save_name']
        self.cnn_model = FeaturePoolingCNN(config)
        self.cnn_model.load_model(config['save_model_path'])
        self.cnn_model.model.avgpool.register_forward_hook(assign_hook)

    def extract_features(self, patches_dataset, batch_size):
        batches = len(patches_dataset) // batch_size
        with torch.no_grad():
            features = {}
            dl = DataLoader(dataset=patches_dataset, batch_size=batch_size, shuffle=False)

            for i, ds_sample in enumerate(dl):
                print(f'{datetime.now().strftime("%H:%M:%S")} Extracting batch: {i}/{batches}')
                images, _, ids, x, y = ds_sample
                image_batch = images.to(self.device)
                _ = self.cnn_model.model.forward(image_batch)
                features_batch = features_hook.reshape(-1, features_batch.shape[1]).cpu()

                for i in range(len(ids)):
                    if ids[i] in features.keys():
                        features[ids[i]][f'{x[i]}_{y[i]}'] = features_batch[i]
                    else:
                        features[ids[i]] = {f'{x[i]}_{y[i]}': features_batch[i]}
            torch.save(features, self.save_name)


if __name__ == '__main__':
    config_file = sys.argv[1]
    config = read_config(config_file)
    extractor = FeatureExtraction(config)
    dataset = PatchesDataset(config['patches_filename'],
                             config['labels_filename'],
                             ['train', 'test', 'val', 'valid'],
                             split=config['split'])
    extractor.extract_features(dataset, 64)