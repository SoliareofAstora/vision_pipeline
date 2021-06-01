import json
import pathlib

import torch
import torch.utils.data
from dataset.patches_dataset import PatchesDataset


class FeatureVectorsDataset(torch.utils.data.Dataset):
    def __init__(self, source_path, return_coverage=False, min_coverage=0, max_coverage=1, fold=None, aux=True, device='cpu',**kwargs):
        with open(source_path / 'dataset_metadata.json', 'r') as f:
            self.dataset_args = json.load(f)
        assert self.dataset_args['content_type'] == 'feature_vectors', 'incompatible dataset type ' + str(source_path)
        self.return_coverage = return_coverage

        # todo remove this code smell: use dataset_contents.csv from patches dataset. Encapsulate CSV read function
        patches_dataset = PatchesDataset(pathlib.Path(self.dataset_args['source_patches_dataset_path']), False,
                                         return_coverage, min_coverage, max_coverage, fold)

        self.data = patches_dataset.data
        self.index = self.data.index

        self.aux = aux
        self.fv = torch.load(source_path/'features.pth')
        if type(self.fv) == list:
            if not self.aux:
                self.fv = self.fv[0]
        else:
            self.aux = False

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        read_index = self.index[idx]
        patch = self.data.loc[read_index]
        if self.aux and type(self.fv) == list:
            fv = [fv[read_index] for fv in self.fv]
        else:
            fv = self.fv[read_index]
        # todo find usages of x y
        metadata = {'xy': pathlib.Path(patch['path']).stem.split('_')}
        if self.return_coverage:
            metadata['coverage'] = patch['coverage']
        return {'fv': fv, 'label': patch['label'], 'metadata': metadata}

    def __len__(self):
        return len(self.data)
