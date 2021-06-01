import argparse
from datetime import datetime
import itertools
import json
import os
from multiprocessing import Pool, cpu_count
import pathlib
import shutil
import warnings

import numpy as np
import pandas as pd
from skimage import io
import skimage.transform as transform
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray


def parse_args():
    parser = argparse.ArgumentParser(description='Create patches from raw images')
    parser.add_argument('-data_path', help='source to data folder root', type=str, required=True)
    parser.add_argument('-image_source', help='path with .tif images relative_to data_path', type=str, required=True)
    parser.add_argument('-patch_size', help='patch width and height', type=int, default=224, required=False)
    parser.add_argument('-coverage', help='save coverage info', type=bool, default=True, required=False)
    parser.add_argument('-step', help='distance ratio < 1 < pixels per step', type=float, default=0.5, required=False)
    parser.add_argument('-low_res', help='use downscaled tif image', type=bool, default=False, required=False)
    parser.add_argument('-core_limit', help='limit cpu usage', type=int, default=0, required=False)
    return parser.parse_args()


def calculate_step(patch_size, step=0.5):
    if step > 1:
        return max(int(step), 50)
    else:
        return max(int(patch_size * step), 50)


def cut_image_into_patches(img, patch_size, step, return_coverage=False):
    patches = []
    patches_metadata = []
    coverage_mask = None
    if return_coverage:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gray_image = rgb2gray(img)
        threshold = threshold_otsu(gray_image)
        coverage_mask = gray_image < threshold

    for i in range(0, img.shape[0] - patch_size, step):
        for j in range(0, img.shape[1] - patch_size, step):
            patches.append(img[i:i + patch_size, j:j + patch_size])
            metadata = [i, j]
            if return_coverage:
                metadata.append(np.sum(coverage_mask[i:i + patch_size, j:j + patch_size]) / patch_size ** 2)
            patches_metadata.append(metadata)

    return patches, patches_metadata, coverage_mask


def create_colony_patches(dst_path, colony_path, patch_size, step, low_res, coverage):
    colony_dst_path = dst_path / colony_path.name
    colony_dst_path.mkdir(exist_ok=True)
    colony_metadata = []
    tif_images_paths = list(colony_path.glob('*'))
    for tif_image_path in tif_images_paths:
        patches_dst_path = colony_dst_path / tif_image_path.stem

        try:
            img = io.imread(str(tif_image_path))
        except:
            print('UNABLE TO READ IMAGE: ' + str(tif_image_path))
            continue

        img = (img - np.amin(img)) / (np.amax(img) - np.amin(img)) * 255
        if low_res:
            img = transform.resize(img, (img.shape[0] // 2, img.shape[1] // 2), anti_aliasing=True)
        images, metadata, coverage_mask = cut_image_into_patches(img, patch_size, step, coverage)

        patches_dst_path.mkdir()
        for i in range(0, len(images)):
            image_path = patches_dst_path / f'{metadata[i][0]}_{metadata[i][1]}.png'
            colony_metadata.append([image_path, *metadata[i]])
            io.imsave(image_path, images[i].astype(np.ubyte), check_contrast=False)
        if coverage:
            io.imsave(patches_dst_path / 'coverage_mask.png', coverage_mask.astype(np.ubyte), check_contrast=False)
    return colony_metadata


def create_patches_dataset(data_path, image_source, patch_size=224, step=0.5, low_res=False, coverage=True,
                           core_limit=0, **kwargs):
    source_path = pathlib.Path(data_path / image_source)
    step = calculate_step(patch_size, step)

    version = 0
    dst_path = data_path / 'datasets' / 'patches' / (str(patch_size) + '-' + str(step) + '_v' + str(version))
    while dst_path.exists():
        version += 1
        dst_path = data_path / 'datasets' / 'patches' / (str(patch_size) + '-' + str(step) + '_v' + str(version))
    dst_path.mkdir(parents=True)

    dataset_metadata = {'content_type': 'patches',
                        'created': datetime.now().strftime("%d_%m_%Y-%H_%M_%S"),
                        'image_source': image_source,
                        'patch_size': patch_size,
                        'step': step,
                        'low_res': low_res,
                        'coverage': coverage
                        }

    with open(dst_path / 'dataset_metadata.json', 'w') as f:
        json.dump(dataset_metadata, f)

    with open(dst_path / 'IN_PROGRESS', 'w') as f:
        f.write('dataset creation in progress')
    try:
        colonies_paths = [x for x in sorted(source_path.iterdir()) if x.is_dir()]
        assert len(colonies_paths) > 0, 'No subdirectories found in source directory'
        # create_colony_patches(dst_path, colony_path, patch_size, step, low_res, coverage)
        create_colony_args = [[dst_path, colony_path, patch_size, step, low_res, coverage]
                              for colony_path in colonies_paths]
        # todo multi process per tiff image instead colony folder
        with Pool(cpu_count() if core_limit == 0 else core_limit) as p:
            data = p.starmap(create_colony_patches, create_colony_args)
        data = list(itertools.chain.from_iterable(data))

        if coverage:
            df = pd.DataFrame(data, columns=['path', 'x', 'y', 'coverage'])
        else:
            df = pd.DataFrame(data, columns=['path', 'x', 'y'])

        colony_label = {path.name: index for index, path in enumerate(colonies_paths)}
        parts = df['path'].apply(lambda x: pathlib.Path(x).parts)
        df['label'] = parts.apply(lambda x: colony_label[x[-3]])
        df['path'] = df['path'].apply(lambda x: str(x.relative_to(dst_path)))
        df.to_csv(dst_path / 'dataset_contents.csv')
    except Exception as e:
        shutil.rmtree(str(dst_path))
        raise e

    os.remove(str(dst_path / 'IN_PROGRESS'))
    return dst_path


if __name__ == '__main__':
    args = parse_args()
    create_patches_dataset(pathlib.Path(args.data_path), args.image_source, args.patch_size,
                           args.step, args.low_res, args.coverage, args.core_limit)
