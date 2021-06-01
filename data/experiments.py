"""Performs experiments for a given number of epoches, after each epoch the validation error and loss is computed to
 monitor the training."""

from __future__ import print_function

import torch.utils.data as data_utils
from tqdm import tqdm

from data.bacteria_bag import image_collate

import time

import numpy as np
import torch
from torch.autograd import Variable


def train(args, train_loader, model, optimizer):
    # set loss to 0
    train_loss = 0.
    train_error = 0.

    # set models in training mode
    model.train(True)

    # start training
    for batch_idx, (data, label) in enumerate(train_loader):
        label = label[0]
        if args.cuda:
            data, label = data.cuda(), label.cuda()
        data, label = Variable(data), Variable(label)

        # reset gradients
        optimizer.zero_grad()
        # calculate loss and metrics
        loss, gamma, gamma_kernel = model.calculate_objective(data, label)
        train_loss += loss[0]
        train_error += model.calculate_classification_error(data, label)[0]
        # backward pass
        loss.backward()
        # optimization
        optimizer.step()

    # calculate final loss
    train_loss /= len(train_loader)
    train_error /= len(train_loader)

    return model, train_loss, train_error, gamma, gamma_kernel


def evaluate(args, model, train_loader, data_loader, mode):
    # set model to evaluation mode
    model.eval()
    # set loss to 0
    evaluate_loss = 0.
    evaluate_error = 0.

    if mode == 'validation':

        # CALCULATE classification error and log-likelihood for VALIDATION SET
        for batch_idx, (data, label) in enumerate(data_loader):
            for i in range(len(label)):
                data_one = data[i]
                label_one = label[i]
                if args.cuda:
                    data_one, label_one = data_one.cuda(), label_one.cuda()
                data_one, label_one = Variable(data_one), Variable(label_one)
                evaluate_loss += model.calculate_objective(data_one, label_one)[0]/len(label)
                evaluate_error += model.calculate_classification_error(data_one, label_one)[0]/len(label)
        # calculate final loss
        evaluate_loss /= len(data_loader)
        evaluate_error /= len(data_loader)

        # cm_valid = np.zeros((2, 2))
        # # true_neg and true_pos for train_dataset
        # for batch_idx, (data, label) in enumerate(data_loader):
        #     for i in range(len(label)):
        #         data_one = data[i]
        #         label_one = label[i]
        #         if args.cuda:
        #             data_one, label_one = data_one.cuda(), label_one.cuda()
        #         data_one, label_one = Variable(data_one), Variable(label_one)
        #         y_prob, y_hat, _, _, gamma, gamma_kernel = model.forward(data_one)
        #         cm_valid[int(label_one.cpu().detach().numpy()), int(y_hat.cpu().detach().numpy())] += 1
        #
        # cm_train = np.zeros((2, 2))
        # # true_neg and true_pos for train_dataset
        # for batch_idx, (data, label) in enumerate(train_loader):
        #     for i in range(len(label)):
        #         data_one = data[i]
        #         label_one = label[i]
        #         if args.cuda:
        #             data_one, label_one = data_one.cuda(), label_one.cuda()
        #         data_one, label_one = Variable(data_one), Variable(label_one)
        #         y_prob, y_hat, _, _, gamma, gamma_kernel = model.forward(data_one)
        #         cm_train[int(label_one.cpu().detach().numpy()), int(y_hat.cpu().detach().numpy())] += 1


    if mode == 'test':
        # set loss to 0
        train_error = 0.
        train_loss = 0.
        # CALCULATE classification error and log-likelihood for TEST SET
        t_ll_s = time.time()
        for batch_idx, (data, label) in enumerate(data_loader):
            for i in range(len(label)):
                data_one = data[i]
                label_one = label[i]
                if args.cuda:
                    data_one, label_one = data_one.cuda(), label_one.cuda()
                data_one, label_one = Variable(data_one), Variable(label_one)
                evaluate_loss += model.calculate_objective(data_one, label_one)[0]/len(label)
                evaluate_error += model.calculate_classification_error(data_one, label_one)[0]/len(label)

        t_ll_e = time.time()
        evaluate_error /= len(data_loader)
        evaluate_loss /= len(data_loader)
        print('\tTEST classification error value (time): {:.4f} ({:.2f}s)'.format(evaluate_error, t_ll_e - t_ll_s))
        print('\tTEST log-likelihood value (time): {:.4f} ({:.2f}s)\n'.format(evaluate_loss, t_ll_e - t_ll_s))

        # CALCULATE classification error and log-likelihood for TRAINING SET
        t_ll_s = time.time()
        for batch_idx, (data, label) in enumerate(train_loader):
            for i in range(len(label)):
                data_one = data[i]
                label_one = label[i]
                if args.cuda:
                    data_one, label_one = data_one.cuda(), label_one.cuda()
                data_one, label_one = Variable(data_one), Variable(label_one)
                train_loss += model.calculate_objective(data_one, label_one)[0]/len(label)
                train_error += model.calculate_classification_error(data_one, label_one)[0]/len(label)
        print(train_error, len(train_loader))
        t_ll_e = time.time()
        train_error /= len(train_loader)
        train_loss /= len(train_loader)
        print('\tTRAIN classification error value (time): {:.4f} ({:.2f}s)'.format(train_error, t_ll_e - t_ll_s))
        print('\tTRAIN log-likelihood value (time): {:.4f} ({:.2f}s)\n'.format(train_loss, t_ll_e - t_ll_s))

    if mode == 'test':
        return evaluate_loss, evaluate_error, train_loss, train_error
    else:
        return evaluate_loss, evaluate_error


def experiment(args, kwargs, current_fold, train_set, val_set, test_set, sampler, model, optimizer, scheduler, dir, sw):
    best_error = 1.
    best_loss = 1000.
    best_error_train = 1.
    e = 1
    train_loss_history = []
    train_error_history = []
    val_loss_history = []
    val_error_history = []
    time_history = []

    path_name_current_fold = dir + args.model_name + '_fold_' + str(current_fold) + '_seed_' + str(args.seed)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.train_batch_size, shuffle=False,
                                               collate_fn=image_collate, sampler=sampler, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.test_batch_size, shuffle=True,
                                               collate_fn=image_collate,  **kwargs)

    bar_tqdm = tqdm(range(1, args.epochs+1), desc="Epochs")
    for epoch in bar_tqdm:
        time_start = time.time()

        model, train_loss, train_error, gamma, gamma_kernel = train(args, train_loader, model, optimizer)
        scheduler.step()
        # val_loss, val_error, cm_train, cm_valid = evaluate(args, model, train_loader, val_loader, mode='validation')
        val_loss, val_error = evaluate(args, model, train_loader, val_loader, mode='validation')

        time_end = time.time()
        time_elapsed = time_end - time_start
        sw.add_scalar('train/loss', train_loss, epoch)
        sw.add_scalar('val/loss', val_loss, epoch)
        sw.add_scalar('train/error', train_error, epoch)
        sw.add_scalar('val/error', val_error, epoch)
        sw.add_scalar('gammma', gamma, epoch)
        sw.add_scalar('gamma_kernel', gamma_kernel, epoch)
        # sw.add_scalar('tpr/val', cm_valid[0, 0]/(cm_valid[0, 0] + cm_valid[0, 1]), epoch)
        # sw.add_scalar('tnr/val', cm_valid[1, 1]/(cm_valid[1, 1] + cm_valid[1, 0]), epoch)
        # sw.add_scalar('tpr/train', cm_train[0, 0]/(cm_train[0, 0] + cm_train[0, 1]), epoch)
        # sw.add_scalar('tnr/train', cm_train[1, 1]/(cm_train[1, 1] + cm_train[1, 0]), epoch)


        # appending history
        # train_loss_history.append(train_loss)
        # train_error_history.append(train_error)
        # val_loss_history.append(val_loss)
        # val_error_history.append(val_error)
        # time_history.append(time_elapsed)

        # printing results
        print('\tResults Epoch: {}/{} in Test-Train fold: {}/{}, Time elapsed: {:.2f}s\n'
              '\t* Train loss: {:.4f}   , error: {:.4f}\n'
              '\to Val.  loss: {:.4f}   , error: {:.4f}\n'
              '\t--> Early stopping: {}/{} (BEST: {:.4f})\n\n'.format(
            epoch, args.epochs, current_fold, args.kfold_test, time_elapsed,
            train_loss, train_error,
            val_loss, val_error,
            e, args.early_stopping_epochs, best_error
        ))

        # early-stopping
        if val_error < best_error:
            e = 0
            best_error = val_error
            best_loss = val_loss
            torch.save(model, path_name_current_fold + '.models')
            print('>>--models saved--<<')
            print(path_name_current_fold + '.models')
        elif val_error == best_error:
            if val_loss < best_loss and train_error < best_error_train:
                e = 0
                best_error = val_error
                best_loss = val_loss
                best_error_train = train_error
                torch.save(model, path_name_current_fold + '.models')
                print('>>--models saved--<<')
                print(path_name_current_fold + '.models')
            else:
                e += 1
                if e > args.early_stopping_epochs:
                    break
        else:
            e += 1
            if e > args.early_stopping_epochs:
                break

    # SAVING
    torch.save(args, path_name_current_fold + '.config')

    # FINAL EVALUATION
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False,
                                               collate_fn=image_collate, **kwargs)

    evaluate_model = torch.load(path_name_current_fold + '.models')
    print('>>--models loaded--<<')
    print(path_name_current_fold + '.models')
    objective_test, error_test, objective_train, error_train = evaluate(args,
                                                                        evaluate_model,
                                                                        train_loader,
                                                                        test_loader,
                                                                        mode='test')

    sw.add_scalar('error_test', error_test, 0)

    print('\tFINAL EVALUATION ON TEST SET OF TEST-TRAIN FOLD: {}/{}\n'
          '\tLogL (TEST): {:.4f}\n'
          '\tLogL (TRAIN): {:.4f}\n'
          '\tERROR (TEST): {:.4f}\n'
          '\tERROR (TRAIN): {:.4f}\n'.format(
        current_fold, args.kfold_test,
        objective_test,
        objective_train,
        error_test,
        error_train
    ))

    with open('experiment_log_' + args.operator + '.txt', 'a') as f:
        print('FINAL EVALUATION ON TEST SET OF TEST-TRAIN FOLD: {}/{}\n'
              'LogL (TEST): {:.4f}\n'
              'LogL (TRAIN): {:.4f}\n'
              'ERROR (TEST): {:.4f}\n'
              'ERROR (TRAIN): {:.4f}\n'
              'TRAIN TIME: {:.1f}\n'
              'TOTAL EPOCHS: {}\n'.format(
            current_fold, args.kfold_test,
            objective_test,
            objective_train,
            error_test,
            error_train,
            np.sum(np.asarray(time_history)),
            len(time_history)
        ), file=f)

    # SAVING
    # torch.save(train_loss_history, path_name_current_fold + '.train_loss')
    # torch.save(train_error_history, path_name_current_fold + '.train_error')
    # torch.save(val_loss_history, path_name_current_fold + '.val_loss')
    # torch.save(val_error_history, path_name_current_fold + '.val_error')
    # torch.save(objective_test, path_name_current_fold + '.objective_test')
    # torch.save(objective_train, path_name_current_fold + '.objective_train')
    # torch.save(error_test, path_name_current_fold + '.error_test')
    # torch.save(error_train, path_name_current_fold + '.error_train')

    return error_train, error_test


def train(args, train_loader, model, optimizer):
    # set loss to 0
    train_loss = 0.
    train_error = 0.

    # set models in training mode
    model.train(True)

    # start training
    train_loader_iter = iter(train_loader)
    for batch_idx in tqdm(range(len(train_loader)), desc='Batches'):
        data, label = next(train_loader_iter)
        # print(label)
        # label = label[0]

        # reset gradients
        optimizer.zero_grad()
        # calculate loss and metrics
        for i in range(len(label)):
            data_one = data[i]
            label_one = label[i]
            if args.cuda:
                data_one, label_one = data_one.cuda(), label_one.cuda()
            data_one, label_one = Variable(data_one), Variable(label_one)
            loss_one, gamma, gamma_kernel = model.calculate_objective(data_one, label_one)
            tmp_error = model.calculate_classification_error(data_one, label_one)[0]
            # print(loss_one.item(), tmp_error)
            train_loss += loss_one.item()/len(label)
            train_error += tmp_error/len(label)
            # backward pass
            loss_one.backward()
        optimizer.step()
    print(len(label), len(train_loader))
    # calculate final loss
    train_loss /= len(train_loader)
    train_error /= len(train_loader)

    return model, train_loss, train_error, gamma, gamma_kernel
