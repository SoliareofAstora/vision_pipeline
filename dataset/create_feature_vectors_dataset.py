import shutil
import json
import os
from datetime import datetime

import torch
import torch.utils.data

from lock import lock, unlock
from dataset.patches_dataset import PatchesDataset


def extract_feature_vectors(model, patches_dataset_path, device, batch_size=512, core_limit=8):
    patches_dataset = PatchesDataset(patches_dataset_path, augment=False)
    data_loader = torch.utils.data.DataLoader(patches_dataset, batch_size, num_workers=core_limit)

    feature_vectors = []
    with torch.no_grad():
        for batch in data_loader:
            fv = model.feature_vectors(batch['image'].to(device=device))
            if type(fv) == list:
                fv_list = []
                for e in fv:
                    fv_list.append(e.detach().cpu())
                feature_vectors.append(fv_list)
            else:
                feature_vectors.append(fv.detach().cpu())

    if type(feature_vectors[0]) == list:
        transposed = list(zip(*feature_vectors))
        return [torch.cat(transposed[i]) for i in range(len(transposed))]
    else:
        return torch.cat(feature_vectors)


def create_feature_vector_dataset(model, patches_dataset_path, args):
    dataset_metadata = {'content_type': 'feature_vectors',
                        'created': datetime.now().strftime("%d_%m_%Y-%H_%M_%S"),
                        'model_name': model.name,
                        'fv_dim': model.fv_dim,
                        'source_patches_dataset_path': str(patches_dataset_path)
                        }
    fv_dataset_path = args['data_path'] / 'datasets' / 'fv' / patches_dataset_path.name / str(args['job_id'])
    fv_dataset_path.mkdir(parents=True)

    lock(fv_dataset_path)
    with open(fv_dataset_path/'dataset_metadata.json', 'w') as f:
        json.dump(dataset_metadata, f)

    try:
        features = extract_feature_vectors(model, patches_dataset_path, args['device'], args['core_limit'])
        torch.save(features, fv_dataset_path/'features.pth')
    except Exception as e:
        shutil.rmtree(str(fv_dataset_path))
        raise e

    unlock(fv_dataset_path)

    return fv_dataset_path
