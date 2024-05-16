# ------------------------------------------------------------------------------
# --coding='utf-8'--
# Written by czifan (czifan@pku.edu.cn)
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from pyexpat import model
import torch
import torch.optim as optim
import prettytable as pt
import numpy as np
from networks import DeepSurv
from networks import NegativeLogLikelihood
from datasets import SurvivalDataset
from tool import generateH5
from utils import read_config
from utils import c_index
from utils import adjust_learning_rate
from utils import create_logger


import logging

def train(ini_file):
    ''' Performs training according to .ini file

    :param ini_file: (String) the path of .ini file
    :return best_c_index: the best c-index
    '''
    # reads configuration from .ini file
    config = read_config(ini_file)
    # builds network|criterion|optimizer based on configuration
    model = DeepSurv(config['network']).to(device)
    criterion = NegativeLogLikelihood(config['network']).to(device)
    optimizer = eval('optim.{}'.format(config['train']['optimizer']))(
        model.parameters(), lr=config['train']['learning_rate'])
    # constructs data loaders based on configuration
    train_dataset = SurvivalDataset(config['train']['h5_file'], is_train=True)
    test_dataset = SurvivalDataset(config['train']['h5_file'], is_train=False)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_dataset.__len__())
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_dataset.__len__())
    # training
    best_c_index = 0
    flag = 0
    for epoch in range(1, config['train']['epochs']+1):
        # adjusts learning rate
        lr = adjust_learning_rate(optimizer, epoch,
                                  config['train']['learning_rate'],
                                  config['train']['lr_decay_rate'])
        # train step
        model.train()
        for X, y, e in train_loader:
            # makes predictions
            risk_pred = model(X)
            train_loss = criterion(risk_pred, y, e, model)
            train_c = c_index(-risk_pred, y, e)
            # updates parameters
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        # valid step
        model.eval()
        for X, y, e in test_loader:
            # makes predictions
            with torch.no_grad():
                risk_pred = model(X)
                valid_loss = criterion(risk_pred, y, e, model)
                valid_c = c_index(-risk_pred, y, e)
                if best_c_index < valid_c:
                    best_c_index = valid_c
                    flag = 0
                    # saves the best model
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch}, os.path.join(models_dir, ini_file.split('\\')[-1]+'.pth'))
                else:
                    flag += 1
                    if flag >= patience:
                        return best_c_index
        # notes that, train loader and valid loader both have one batch!!!
        print('\rEpoch: {}\tLoss: {:.8f}({:.8f})\tc-index: {:.8f}({:.8f})\tlr: {:g}'.format(
            epoch, train_loss.item(), valid_loss.item(), train_c, valid_c,lr), end='', flush=False)
    return best_c_index


def predict(ini_file):
    config = read_config(ini_file)
    model = DeepSurv(config['network']).to(device)
    model_path = r'logs4\models\nasa.ini.pth'
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    test_dataset = SurvivalDataset(config['train']['h5_file'], is_train=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_dataset.__len__())
    criterion = NegativeLogLikelihood(config['network']).to(device)
    for X, y, e in test_loader:
            with torch.no_grad():
                risk_pred = model(X)
    return risk_pred
if __name__ == '__main__':
    headers = []
    values = []
    for i in range(10):
        generateH5(r'custom_data\NASA_ALL\cleaned_features_final.csv',split_size=0.3)
        logs_dir = 'logs_NASA'
        models_dir = os.path.join(logs_dir, 'models')
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        logger = create_logger(logs_dir)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        configs_dir = 'configs'
        params = [('Simulated linear', 'nasafull.ini')]
        patience = 50
        # training
        for name, ini_file in params:
            logger.info('Running {}({})...{}'.format(name, ini_file,i))
            best_c_index = train(os.path.join(configs_dir, ini_file))
            headers.append(name+":"+str(i))
            values.append('{:.6f}'.format(best_c_index))
            print('')
            logger.info("The best valid c-index: {}".format(best_c_index))
            logger.info('')
        # prints results
        tb = pt.PrettyTable()
        tb.field_names = headers
        tb.add_row(values)
        logger.info(tb)
    # patience = 50
    # logs_dir = 'logs'
    # models_dir = os.path.join(logs_dir, 'models')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # configs_dir = 'configs'
    # p_pred = predict(os.path.join(configs_dir, 'nasafull.ini'))
    # print(p_pred)
