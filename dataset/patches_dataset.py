import json
import pathlib

import pandas as pd
import torch.utils.data
from PIL import Image
from torchvision.transforms import Compose, RandomHorizontalFlip, ToTensor

from dataset.augmentation import RandomRotate


class PatchesDataset(torch.utils.data.Dataset):
    def __init__(self, source_path, augment=True, return_coverage=False, min_coverage=0, max_coverage=1, fold=None,  **kwargs):
        with open(source_path / 'dataset_metadata.json', 'r') as f:
            self.dataset_args = json.load(f)
        assert self.dataset_args['content_type'] == 'patches', 'incompatible dataset type ' + str(source_path)

        assert (source_path / 'dataset_contents.csv').exists(), 'missing dataset_contents.csv'
        self.data = pd.read_csv(source_path / 'dataset_contents.csv', index_col=0)

        if fold is not None:
            parts = self.data['path'].apply(lambda x: pathlib.Path(x).parts)
            try:
                self.data = self.data[parts.apply(lambda x: x[-2] in fold[x[-3]])]
            except KeyError:
                raise NotImplementedError('Subset of classes is not available')

        self.return_coverage = return_coverage
        if self.return_coverage or min_coverage > 0 or max_coverage < 1:
            assert self.dataset_args['coverage'], 'dataset doesnt have coverage info stored ' + str(source_path)
            if min_coverage > 0:
                self.data = self.data[self.data['coverage'] >= min_coverage]
            if max_coverage < 1:
                self.data = self.data[self.data['coverage'] <= max_coverage]
        self.data['path'] = self.data['path'].apply(lambda x: source_path/x)

        if not augment:
            self.transforms = Compose([
                RandomRotate(),
                RandomHorizontalFlip(p=0.5),
                ToTensor(),
            ])
        else:
            self.transforms = ToTensor()

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        patch = self.data.iloc[idx]
        img = Image.open(patch['path'])
        # todo find usages of x y
        metadata = {'xy': pathlib.Path(patch['path']).stem.split('_')}
        if self.return_coverage:
            metadata['coverage'] = patch['coverage']
        return {'image': self.transforms(img), 'label': patch['label'], 'metadata': metadata}

    def __len__(self):
        return len(self.data)
