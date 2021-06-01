import json

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataset.patches_dataset import PatchesDataset
from dataset.feature_vectors_dataset import FeatureVectorsDataset
import dataset.get_dataset_path as get_dataset_path
from model.get_model_class import get_model_class
from training.components.get_optimizer_class import get_optimizer_class
from training.components.get_scheduler_class import get_scheduler_class
from training.components.get_loss_class import get_loss_class
from dataset.create_k_folds import dataset_to_dict, create_k_folds


def initialize_components(args):
    work_dir = args['work_dir']

    # load checkpoint
    checkpoint = None
    if (work_dir / 'checkpoint.pth').exists():
        checkpoint = torch.load(work_dir / 'checkpoint.pth')

    # load model
    model_class = get_model_class(args['model_name'])
    if 'model_args' in args.keys():
        model_args = args['model_args'].copy()
        if 'num_classes' not in model_args.keys() and 'num_classes' not in args.keys():
            raise NotImplementedError

        model_args['model_name'] = args['model_name']
        if checkpoint is not None and 'pretrained' in model_args.keys():
            model_args['pretrained'] = False
        model = model_class(**model_args)
    else:
        model = model_class()
    if 'patch_size' not in args.keys():
        args['dataset_args']['patch_size'] = model.input_dim
    model.to(device=args['device'])
    model.train()

    # load optimizer
    parameters_to_train = [p for p in model.parameters() if p.requires_grad]
    optimizer_class = get_optimizer_class(args['optimizer_name'])
    if 'optimizer_args' in args.keys():
        optimizer = optimizer_class(parameters_to_train, **args['optimizer_args'])
    else:
        optimizer = optimizer_class(parameters_to_train)

    # load scheduler
    scheduler = None
    if 'scheduler_name' in args.keys():
        scheduler_class = get_scheduler_class(args['scheduler_name'])
        if 'scheduler_args' in args.keys():
            scheduler = scheduler_class(optimizer, **args['scheduler_args'])
        else:
            scheduler = scheduler_class(optimizer)

    # load loss function
    # todo add possibility to use custom loss functions
    loss_class, require_one_hot = get_loss_class(args['loss_name'])
    if 'loss_args' in args.keys():
        criterion = loss_class(**args['loss_args'])
    else:
        criterion = loss_class()

    # load state dicts from checkpoint if exist
    if checkpoint is not None:
        model_weights_path = work_dir / checkpoint['model_state_dict']
        model.load_state_dict(torch.load(model_weights_path))
        current_training_step = checkpoint['current_training_step']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    else:
        current_training_step = 0

    # get fold for this training
    # todo rewrite this. Especially this 10 training split argument (och whatever...)
    fold = create_k_folds(dataset_to_dict(get_dataset_path.patches_path(args)), 10, args['fold_id'] + 1)[
        args['fold_id']]

    return model, optimizer, scheduler, criterion, require_one_hot, current_training_step, fold


def train(args):
    work_dir = args['work_dir']
    print(f"Training {work_dir.name}")
    model, optimizer, scheduler, criterion, require_one_hot, current_training_step, fold = initialize_components(args)
    training_steps = args['training_steps']
    device = args['device']
    core_limit = args['core_limit']
    aux_weights = None
    if hasattr(model, 'aux'):
        if model.aux is not None:
            aux_weights = [1] + args['model_args']['aux_weights']

    num_classes = model.num_classes
    max_pretraining_steps = args['max_pretraining_steps'] if 'max_pretraining_steps' in args.keys() else 1000

    if current_training_step == 0:
        training_metrics = []
        validation_metrics = []
        test_metrics = []
    else:
        training_metrics = json.load(open(work_dir / 'training_metrics.json', 'r'))
        validation_metrics = json.load(open(work_dir / 'validation_metrics.json', 'r'))
        test_metrics = json.load(open(work_dir / 'test_metrics.json', 'r'))

    # use feature vectors to train classification(head) layer only
    if current_training_step == 0 and 'pretrained' in args['model_args'].keys() and args['model_args']['pretrained']:
        print("Pretraining")
        pretraining_metrics = []
        batch_size = args['pretrain_batch_size'] if 'pretrain_batch_size' in args.keys() else args['batch_size']
        fv_dataset_path = get_dataset_path.feature_vectors_path(model, args)
        train_ds = FeatureVectorsDataset(fv_dataset_path, aux=True, fold=fold['training'], device=device,
                                         **args['dataset_args'])
        train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=core_limit, drop_last=True)

        model.train()
        model.freeze_backbone()

        # load optimizer
        parameters_to_train = [p for p in model.parameters() if p.requires_grad]
        optimizer_class = get_optimizer_class(args['optimizer_name'])
        if 'optimizer_args' in args.keys():
            pre_optimizer = optimizer_class(parameters_to_train, **args['optimizer_args'])
        else:
            pre_optimizer = optimizer_class(parameters_to_train)
        pre_scheduler = torch.optim.lr_scheduler.StepLR(pre_optimizer, 1, 0.3)
        termination_counter = 0
        minimum_loss = 1000
        train_dl_iter = iter(train_dl)
        for step in range(max_pretraining_steps):
            try:
                batch = next(train_dl_iter)
            except StopIteration:
                train_dl_iter = iter(train_dl)
                batch = next(train_dl_iter)
            pre_optimizer.zero_grad()

            # todo move it to datasets
            if require_one_hot:
                labels = F.one_hot(batch['label'], num_classes).to(device=device).to(torch.float32)
            else:
                labels = batch['label'].to(device=device)

            if type(batch['fv']) == list:
                fvs = []
                for fv in batch['fv']:
                    fvs.append(fv.to(device=device))
                pred = model.forward_head(fvs)
            else:
                pred = model.forward_head(batch['fv'].to(device=device))

            if type(pred) == list:
                losses = []
                if aux_weights is not None:
                    for i in range(len(pred)):
                        losses.append(aux_weights[i] * criterion(pred[i], labels))
                else:
                    for i in range(len(pred)):
                        losses.append(criterion(pred[i], labels))
                loss = sum(losses) / len(losses)
            else:
                loss = criterion(pred, labels)

            if type(pred) == list:
                batch_acc = []
                for i in range(len(pred)):
                    batch_acc.append(torch.sum(pred[i].max(1)[1].cpu() == batch['label']).item() / batch_size)
                acc = sum(batch_acc) / len(pred)
            else:
                acc = torch.sum(pred.max(1)[1].cpu() == batch['label']).item() / batch_size
            pretraining_metrics.append([step, loss.item(), acc])

            loss.backward()
            pre_optimizer.step()

            if loss.item() < minimum_loss:
                minimum_loss = loss.item()
                termination_counter = 0
            else:
                termination_counter += 1
                if termination_counter > 20:
                    break
                pre_scheduler.step()

        model.freeze_model(False)
        json.dump(pretraining_metrics, open(work_dir / 'pretraining_metrics.json', 'w'))

    # standard training loop
    if training_steps > current_training_step:
        print('Training')
        batch_size = args['batch_size']
        patches_dataset_path = get_dataset_path.patches_path(args)
        train_ds = PatchesDataset(patches_dataset_path, fold=fold['training'], **args['dataset_args'])
        train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=core_limit, drop_last=True)

        validation_ds = PatchesDataset(patches_dataset_path, fold=fold['validation'], **args['dataset_args'])
        validation_dl = DataLoader(validation_ds, batch_size, shuffle=True, num_workers=core_limit)

        test_ds = PatchesDataset(patches_dataset_path, fold=fold['test'], **args['dataset_args'])
        test_dl = DataLoader(test_ds, batch_size, shuffle=True, num_workers=core_limit)

        model.train()
        train_dl_iter = iter(train_dl)
        for step in range(training_steps):
            try:
                batch = next(train_dl_iter)
            except StopIteration:
                train_dl_iter = iter(train_dl)
                batch = next(train_dl_iter)
            optimizer.zero_grad()

            # todo move it to datasets
            if require_one_hot:
                labels = F.one_hot(batch['label'], num_classes).to(device=device).to(torch.float32)
            else:
                labels = batch['label'].to(device=device)

            pred = model.forward(batch['image'].to(device=device))

            if type(pred) == torch.Tensor:
                loss = criterion(pred, labels)
            else:
                losses = []
                if aux_weights is not None:
                    for i in range(len(pred)):
                        losses.append(aux_weights[i] * criterion(pred[i], labels))
                else:
                    for i in range(len(pred)):
                        losses.append(criterion(pred[i], labels))
                loss = sum(losses) / len(losses)

            if type(pred) == torch.Tensor:
                acc = torch.sum(pred.max(1)[1].cpu() == batch['label']).item() / batch_size
            else:
                batch_acc = []
                for i in range(len(pred)):
                    batch_acc.append(torch.sum(pred[i].max(1)[1].cpu() == batch['label']).item() / batch_size)
                acc = sum(batch_acc) / len(pred)
            training_metrics.append([step, loss.item(), acc])

            loss.backward()
            optimizer.step()
            scheduler.step()

            if (step + 1) % args['validation_freq'] == 0:
                model.train(False)
                val_loss = []
                val_acc = []
                for batch in validation_dl:
                    pred = model.forward(batch['image'].to(device=device))
                    if type(pred) != torch.Tensor:
                        pred = pred[0]

                    if require_one_hot:
                        labels = F.one_hot(batch['label'], num_classes).to(device=device)
                    else:
                        labels = batch['label'].to(device=device)
                    val_loss.append(criterion(pred, labels).item())
                    val_acc.append(torch.sum(pred.max(1)[1].cpu() == batch['label']).item())
                validation_metrics.append([step + 1, sum(val_loss) / len(val_loss), sum(val_acc) / len(validation_ds)])
                model.train(True)

            if (step + 1) % args['checkpoint_freq'] == 0:
                model.train(False)
                test_loss = []
                test_acc = []
                for batch in test_dl:
                    pred = model.forward(batch['image'].to(device=device))
                    if type(pred) != torch.Tensor:
                        pred = pred[0]

                    if require_one_hot:
                        labels = F.one_hot(batch['label'], num_classes).to(device=device)
                    else:
                        labels = batch['label'].to(device=device)
                    test_loss.append(criterion(pred, labels).item())
                    test_acc.append(torch.sum(pred.max(1)[1].cpu() == batch['label']).item())
                test_metrics.append([step + 1, sum(test_loss) / len(test_loss), sum(test_acc) / len(test_ds)])
                model.train(True)

                # save model and statistics with test results
                current_training_step = step + 1
                checkpoint = {
                    'current_training_step': current_training_step,
                    'model_state_dict': f'{current_training_step}.pth',
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }
                torch.save(checkpoint, work_dir / 'checkpoint.pth')
                torch.save(model.state_dict(), work_dir / f'{current_training_step}.pth')
                json.dump(training_metrics, open(work_dir / 'training_metrics.json', 'w'))
                json.dump(validation_metrics, open(work_dir / 'validation_metrics.json', 'w'))
                json.dump(test_metrics, open(work_dir / 'test_metrics.json', 'w'))
