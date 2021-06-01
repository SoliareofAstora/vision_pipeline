import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

from torch.utils.data import DataLoader

from feature_pooling.cnn_feature_pool import FeaturePoolingCNN

from dataset.patches_dataset import PatchesDataset

# todo remove those paths
networks = {'AD': '/workspace/other/bakterie/models/2020_07_11_model_1_AD_5000.pkl',
            'AB': '/workspace/other/bakterie/models/2020_07_11_model_1_AB_5000.pkl',
            'ABDE': '/workspace/other/bakterie/models/2020_07_11_model_1_ABDE_5000.pkl'}

features = {'AD': '/workspace/other/bakterie/features/2020_07_19_AD_features.pt',
            'AB': '/workspace/other/bakterie/features/2020_07_19_AB_features.pt',
            'ABDE': '/workspace/other/bakterie/features/2020_07_19_ABDE_features.pt'}

labels_files = {'AD': '/workspace/eeml/bakterie/labels_5splits_AD.csv',
                'AB': '/workspace/eeml/bakterie/labels_5splits_AB.csv',
                'ABDE': '/workspace/eeml/bakterie/labels_all_clones.csv'}

patches_files = {'AD': '/workspace/eeml/bakterie/patches_AD.csv',
                 'AB': '/workspace/eeml/bakterie/patches_AB.csv',
                 'ABDE': '/workspace/eeml/bakterie/patches_all_clones.csv'}

label_cm = {'AD': ['A', 'D'],
            'AB': ['A', 'B'],
            'ABDE': ['A', 'B', 'D', 'E', 'x']}

COLOR_MAP = sns.dark_palette('dark fuchsia', input='xkcd')


def show_cm(arrs, n, set_name):
    fig, ax = plt.subplots(nrows=1, ncols=n, figsize=(n * 3 + 3, 3), squeeze=False)
    for i in range(n):
        arr = arrs[i]
        sums = arr.sum(axis=1)
        arr = arr / sums[:, np.newaxis]
        arr = arr.round(2)
        curr_ax = ax[0][i]
        labels_plot = label_cm[set_name]
        sns.heatmap(arr,
                    xticklabels=labels_plot,
                    yticklabels=labels_plot,
                    cmap=COLOR_MAP,
                    square=True,
                    cbar=False,
                    annot=True,
                    ax=curr_ax)
        curr_ax.set_ylabel('True')
        curr_ax.set_xlabel('Predicted')
        curr_ax.set_title(f'Accuracy: {(sum(arr.diagonal()) / sum(sum(arr))).round(2)}')
        bottom, top = curr_ax.get_ylim()
        curr_ax.set_ylim(bottom + 0.5, top - 0.5)
    return fig


def show_avg_cm(arrs, n, set_name):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 3), squeeze=False)
    arr = np.zeros(arrs[0].shape)
    for i in range(n):
        arr += arrs[i]
    sums = arr.sum(axis=1)
    arr = arr / sums[:, np.newaxis]
    arr = arr.round(2)
    labels_plot = label_cm[set_name]
    sns.heatmap(arr,
                xticklabels=labels_plot,
                yticklabels=labels_plot,
                cmap=COLOR_MAP,
                square=True,
                cbar=False,
                annot=True)
    ax[0][0].set_ylabel('True')
    ax[0][0].set_xlabel('Predicted')
    ax[0][0].set_title(f'Accuracy: {(sum(arr.diagonal()) / sum(sum(arr))).round(2)}')
    bottom, top = ax[0][0].get_ylim()
    ax[0][0].set_ylim(bottom + 0.5, top - 0.5)
    return fig


def make_path(row):
    parts = row.image_id.split('_')
    cls = parts[0]
    isolate = f'{parts[1]}_{parts[2]}'
    image = parts[3]
    return f'/workspace/eeml/bakterie/patches/Klon_{cls}/{isolate}/{image}/{row.pos_x}_{row.pos_y}.png'


def draw_patches(patches, value):
    fig, ax = plt.subplots(nrows=1, ncols=10, figsize=(20, 3), squeeze=True)
    for j in range(10):
        try:
            ax[j].imshow(Image.open(patches.path.values[j]))
            ax[j].set_title(patches[value].values[j].round(2))
            ax[j].axis('off')
        except IndexError:
            continue
    return fig


def predict_cnn(set_name, datasets, split):
    with open('/workspace/other/bakterie/configs/feature_pooler/2020_07_11_config_1_AB.yml', 'r') as config_file:
        config = yaml.load(config_file)
    config['output_dim'] = len(label_cm[set_name])
    cnn = FeaturePoolingCNN(config)
    cnn.load_model(networks[set_name])
    ds = PatchesDataset(patches_files[set_name], labels_files[set_name], datasets, split)
    dl = DataLoader(ds, batch_size=64, shuffle=False)
    patches = []
    for X, y, image_id, pos_x, pos_y in dl:
        y_hat = cnn.predict(X)
        y_pred = np.argmax(y_hat, 1)
        for i in range(len(y_hat)):
            patches.append([y[i].item(), y_pred[i], y_hat[i], image_id[i], pos_x[i], pos_y[i]])

    results = pd.DataFrame(patches, columns=['y_true', 'y_pred', 'y_val', 'image_id', 'pos_x', 'pos_y'])
    results['path'] = results.apply(make_path, axis=1)
    results.to_csv(f'/workspace/other/bakterie/results/2020_07_25/cnn_{set_name}_results_{datasets[0]}.csv')
    del cnn
    # cm_curr = cm(results['y_true'], results['y_pred'])
    # print(set_name, datasets)
    # for i in range(len(label_cm[set_name])):
    #     if i in results.y_true.values:
    #         print(f'Class: {i}')
    #         results['curr_y_val'] = results.y_val.apply(lambda x: x[i])
    #         bottom = results[results.y_true == i].sort_values(['curr_y_val'])[:10]
    #         top = results[results.y_true == i].sort_values(['curr_y_val'])[-10:]
    #         img_top = draw_patches(top, 'curr_y_val')
    #         img_bottom = draw_patches(bottom, 'curr_y_val')
    #         img_top.savefig(
    #             f'/workspace/eeml/2020_06_30_cnn_patches_visualizations/patches_top_{set_name}_{datasets[0]}_class{i}.png')
    #         img_bottom.savefig(
    #             f'/workspace/eeml/2020_06_30_cnn_patches_visualizations/patches_bottom_{set_name}_{datasets[0]}_class{i}.png')
    #     else:
    #         print(f'No examples of class {i}')
    # return cm_curr


if __name__ == '__main__':
    for sm in ['AB', 'AD', 'ABDE']:
    # for sm in ['AD']:
        for dataset_name in ['valid', 'train']:
            print(f'{sm} ---- {dataset_name}')
            predict_cnn(sm, [dataset_name], f'SPLIT_CV1')
            # cms.append(cm1)
            # cm_all = show_cm(cms, 1, sm)
            # cm_all.savefig(f'/workspace/eeml/2020_06_30_cnn_patches_visualizations/cm_{sm}_{dataset_name}.png')
