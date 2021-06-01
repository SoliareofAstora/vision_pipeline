import json
import time

from lock import lock, unlock
from dataset.create_patches_dataset import create_patches_dataset
from dataset.create_feature_vectors_dataset import create_feature_vector_dataset


# todo integrate this functionality to dataset definitions
def patches_path(args):
    datasets_path = args['data_path'] / 'datasets' / 'patches'
    dataset_args = args['dataset_args'].copy()
    if datasets_path.exists():
        datasets = [x for x in datasets_path.iterdir() if x.is_dir()]
        for dataset in datasets:
            with open(dataset / 'dataset_metadata.json', 'r') as f:
                ds_params = json.load(f)
            if ds_params['image_source'] != dataset_args['image_source']:
                continue
            if ds_params['patch_size'] != dataset_args['patch_size']:
                continue
            if 'step' in dataset_args.keys():
                if ds_params['step'] != dataset_args['step']:
                    continue
            if 'low_res' in dataset_args.keys():
                if ds_params['low_res'] != dataset_args['low_res']:
                    continue
            if 'coverage' in dataset_args.keys():
                if dataset_args['coverage']:
                    if not ds_params['coverage']:
                        continue

            lock(dataset, 600)  # wait for different process to unlock
            unlock(dataset)
            return dataset

    dataset_args['data_path'] = args['data_path']
    if 'core_limit' in args.keys():
        dataset_args['core_limit'] = args['core_limit']
    return create_patches_dataset(**dataset_args)


def feature_vectors_path(model, args):
    patches_dataset_path = patches_path(args)
    patches_fv_path = args['data_path'] / 'datasets' / 'fv' / patches_dataset_path.name
    if patches_fv_path.exists():
        datasets = [x for x in patches_fv_path.iterdir() if x.is_dir()]
        for dataset in datasets:
            with open(dataset / 'dataset_metadata.json', 'r') as f:
                ds_params = json.load(f)
            if ds_params['model_name'] != args['model_name']:
                continue

            lock(dataset, 600)  # wait for different process to unlock
            unlock(dataset)
            return dataset

    return create_feature_vector_dataset(model, patches_dataset_path, args)

