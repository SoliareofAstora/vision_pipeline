# todo rewrite this entire file to have more general use with other types of datasets
import argparse
import json
import pathlib
import random


def parse_args():
    parser = argparse.ArgumentParser(description='Split dataset into training, validation and test randomly')
    parser.add_argument('-source', help='path to dataset', type=str, required=True)
    parser.add_argument('-split', help='number of training images', type=int, default=10, required=False)
    parser.add_argument('-k', help='number of randomly generated folds', type=int, default=10, required=False)
    parser.add_argument('-seed', help='seed', type=int, default=None, required=False)
    parser.add_argument('-destination', help='path for results', type=str, required=False)
    return parser.parse_args()


def create_k_folds(dataset: dict, split: int, k: int, seed=1024):
    folds = []
    random.seed(seed)
    for i in range(k):
        fold = {'training': {}, 'validation': {}, 'test': {}}
        if split < 1:
            # todo implement
            raise NotImplementedError
        else:
            for label in dataset.keys():
                random.shuffle(dataset[label])
                fold['training'][label] = dataset[label][:split]
                fold['validation'][label] = dataset[label][-2:]
                fold['test'][label] = dataset[label][split:]
        folds.append(fold)
    return folds


def dataset_to_dict(dataset_path):
    colony_images = {}
    colonies_paths = [x for x in dataset_path.iterdir() if x.is_dir()]
    for colony_path in colonies_paths:
        colony_images[colony_path.name] = list(colony_path.glob('*'))

    if colony_images[colonies_paths[0].name][0].name.endswith('.tif'):
        print('Scanning for corrupted images. To avoid this process use patches dataset as a source')
        raise NotImplementedError
        # for colony in colony_images.keys():
        #     for tif_image_path in colony_images[colony][:]:
        #         img = cv2.imread(str(tif_image_path), -1)
        #         if img is None:
        #             print('UNABLE TO READ IMAGE: ' + str(tif_image_path))
        #             colony_images[colony].remove(tif_image_path)
        # for colony in colony_images.keys():
        #     colony_images[colony] = [x.stem for x in colony_images[colony]]
    else:
        for colony in colony_images.keys():
            colony_images[colony] = [x.name for x in colony_images[colony]]

    return colony_images


def main():
    args = parse_args()
    source_path = pathlib.Path(args.source)
    if args.destination is not None:
        dst_path = pathlib.Path(args.destination)
    else:
        dst_path = source_path.parent / 'k_folds' /('kfold' + str(args.k) + 'split' + str(args.split))
    dst_path.mkdir(parents=True, exist_ok=True)

    colony_images = dataset_to_dict(source_path)

    folds = create_k_folds(colony_images, args.split, args.k, args.seed)

    for i in range(args.k):
        with open(dst_path/f'fold{i}.json', 'w') as f:
            json.dump(folds[i], f)
    # todo add dataset metadata or smf?? Not so sure if that's necessary anymore...


if __name__ == '__main__':
    main()
