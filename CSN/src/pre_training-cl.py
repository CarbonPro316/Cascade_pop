import copy
import pickle
import time

import torch
import torch.nn as nn
from torchsummary import summary
from models import CasEncoder, ContrastiveLoss

import numpy as np
from absl import app, flags
from utils.tools import shuffle_two
import random

# flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 64, 'Mini-batch size.')
flags.DEFINE_integer('emb_dim', 64, 'Embedding dimension.')
flags.DEFINE_integer('max_seq', 100, 'Max length of cascade sequence.')
flags.DEFINE_float('l2', 5e-4, 'L2 coefficient.')
flags.DEFINE_float('lr', 5e-4, 'Learning rate.')
flags.DEFINE_integer('model_size', 4, 'Model size.')
flags.DEFINE_integer('model_size_scale', 32, 'Model size base scale.')
flags.DEFINE_string('name', 'swj', 'Name of this experiment.')
flags.DEFINE_integer('pre_training_epochs', 30, 'Pre-training epochs.')
flags.DEFINE_string('projection_head', '4-1', 'MLP-based projection head.')
flags.DEFINE_float('temperature', .1, 'Hyper-parameter temperature for contrastive loss.')
flags.DEFINE_boolean('use_unlabel', True, 'Pre-training with unlabeled data.')
flags.DEFINE_string('cuda_device', '0', 'GPU in use')
flags.DEFINE_bool('PE',False,"use positional embedding")
flags.DEFINE_integer('pos_dim',8,"position embedding dimension")
flags.DEFINE_string('s_weight', 'none', 'soical information.')
# paths
flags.DEFINE_string('input', './datasets/weibo/', 'Pre-training data path.')
flags.DEFINE_string('weight_path', './results/pre_training_weight/', 'Path of saved encoder weights.')


def main(argv):
    start_time = time.time()
    device = torch.device("cuda:" + FLAGS.cuda_device if torch.cuda.is_available() else "cpu")
    print(device)

    # hyper-params
    pos_dim=FLAGS.pos_dim
    batch_size = FLAGS.batch_size
    if FLAGS.s_weight:
        emb_dim = FLAGS.emb_dim+1
    else:
        emb_dim = FLAGS.emb_dim
    gru_units = FLAGS.model_size * FLAGS.model_size_scale
    epochs = FLAGS.pre_training_epochs
    l2 = FLAGS.l2
    lr = FLAGS.lr
    max_seq = FLAGS.max_seq
    projection_head = FLAGS.projection_head
    temperature = FLAGS.temperature
    # hyper-params

    # build model
    casEncoder = CasEncoder(emb_dim, gru_units)

    encoder = nn.Sequential()
    mlp = nn.Sequential()

    encoder_projection = nn.Sequential()

    if projection_head[0] == '0':
        encoder.add_module("casEncoder", casEncoder)
        encoder_projection.add_module("encoder", encoder)
    elif projection_head[0] == '1':
        mlp1 = nn.Linear(gru_units * 2, gru_units * 2)
        mlp.add_module("mlp1", mlp1)
        mlp.add_module("relu1", nn.ReLU())
        if projection_head[2] == '0':
            encoder.add_module("casEncoder", casEncoder)
            encoder_projection.add_module("encoder", encoder)

            encoder_projection.add_module("mlp1", mlp1)
            encoder_projection.add_module("relu1", nn.ReLU())
        elif projection_head[2] == '1':
            encoder.add_module("casEncoder", casEncoder)
            encoder.add_module("mlp1", mlp1)
            encoder.add_module("relu1", nn.ReLU())
            encoder_projection.add_module("encoder", encoder)
        else:
            print('Wrong projection head argument, should be [0-4]-[0-4].')
    elif projection_head[0] == '2':
        mlp1 = nn.Linear(gru_units * 2, gru_units * 2)
        mlp2 = nn.Linear(gru_units * 2, gru_units * 2)
        mlp.add_module("mlp1", mlp1)
        mlp.add_module("relu1", nn.ReLU())
        mlp.add_module("mlp2", mlp2)
        mlp.add_module("relu2", nn.ReLU())
        if projection_head[2] == '0':
            encoder.add_module("casEncoder", casEncoder)
            encoder_projection.add_module("encoder", encoder)

            encoder_projection.add_module("mlp1", mlp1)
            encoder_projection.add_module("relu1", nn.ReLU())
            encoder_projection.add_module("mlp2", mlp2)
            encoder_projection.add_module("relu2", nn.ReLU())
        elif projection_head[2] == '1':
            encoder.add_module("casEncoder", casEncoder)
            encoder.add_module("mlp1", mlp1)
            encoder.add_module("relu1", nn.ReLU())
            encoder_projection.add_module("encoder", encoder)

            encoder_projection.add_module("mlp2", mlp2)
            encoder_projection.add_module("relu2", nn.ReLU())
        elif projection_head[2] == '2':
            encoder.add_module("casEncoder", casEncoder)
            encoder.add_module("mlp1", mlp1)
            encoder.add_module("relu1", nn.ReLU())
            encoder.add_module("mlp2", mlp2)
            encoder.add_module("relu2", nn.ReLU())
            encoder_projection.add_module("encoder", encoder)
        else:
            print('Wrong projection head argument, should be [0-4]-[0-4].')
    elif projection_head[0] == '3':
        mlp1 = nn.Linear(gru_units * 2, gru_units * 2)
        mlp2 = nn.Linear(gru_units * 2, gru_units * 2)
        mlp3 = nn.Linear(gru_units * 2, gru_units * 2)
        mlp.add_module("mlp1", mlp1)
        mlp.add_module("relu1", nn.ReLU())
        mlp.add_module("mlp2", mlp2)
        mlp.add_module("relu2", nn.ReLU())
        mlp.add_module("mlp3", mlp3)
        mlp.add_module("relu3", nn.ReLU())
        if projection_head[2] == '0':
            encoder.add_module("casEncoder", casEncoder)
            encoder_projection.add_module("encoder", encoder)

            encoder_projection.add_module("mlp1", mlp1)
            encoder_projection.add_module("relu1", nn.ReLU())
            encoder_projection.add_module("mlp2", mlp2)
            encoder_projection.add_module("relu2", nn.ReLU())
            encoder_projection.add_module("mlp3", mlp3)
            encoder_projection.add_module("relu3", nn.ReLU())
        elif projection_head[2] == '1':
            encoder.add_module("casEncoder", casEncoder)
            encoder.add_module("mlp1", mlp1)
            encoder.add_module("relu1", nn.ReLU())
            encoder_projection.add_module("encoder", encoder)

            encoder_projection.add_module("mlp2", mlp2)
            encoder_projection.add_module("relu2", nn.ReLU())
            encoder_projection.add_module("mlp3", mlp3)
            encoder_projection.add_module("relu3", nn.ReLU())

        elif projection_head[2] == '2':
            encoder.add_module("casEncoder", casEncoder)
            encoder.add_module("mlp1", mlp1)
            encoder.add_module("relu1", nn.ReLU())
            encoder.add_module("mlp2", mlp2)
            encoder.add_module("relu2", nn.ReLU())
            encoder_projection.add_module("encoder", encoder)

            encoder_projection.add_module("mlp3", mlp3)
            encoder_projection.add_module("relu3", nn.ReLU())
        elif projection_head[2] == '3':
            encoder.add_module("casEncoder", casEncoder)
            encoder.add_module("mlp1", mlp1)
            encoder.add_module("relu1", nn.ReLU())
            encoder.add_module("mlp2", mlp2)
            encoder.add_module("relu2", nn.ReLU())
            encoder.add_module("mlp3", mlp3)
            encoder.add_module("relu3", nn.ReLU())
            encoder_projection.add_module("encoder", encoder)
        else:
            print('Wrong projection head argument, should be [0-4]-[0-4].')
    elif projection_head[0] == '4':
        mlp1 = nn.Linear(gru_units * 2, gru_units * 2)
        mlp2 = nn.Linear(gru_units * 2, gru_units * 2)
        mlp3 = nn.Linear(gru_units * 2, gru_units * 2)
        mlp4 = nn.Linear(gru_units * 2, gru_units * 2)

        mlp.add_module("mlp1", mlp1)
        mlp.add_module("relu1", nn.ReLU())
        mlp.add_module("mlp2", mlp2)
        mlp.add_module("relu2", nn.ReLU())
        mlp.add_module("mlp3", mlp3)
        mlp.add_module("relu3", nn.ReLU())
        mlp.add_module("mlp4", mlp4)
        mlp.add_module("relu4", nn.ReLU())
        if projection_head[2] == '0':
            encoder.add_module("casEncoder", casEncoder)
            encoder_projection.add_module("encoder", encoder)

            encoder_projection.add_module("mlp1", mlp1)
            encoder_projection.add_module("relu1", nn.ReLU())
            encoder_projection.add_module("mlp2", mlp2)
            encoder_projection.add_module("relu2", nn.ReLU())
            encoder_projection.add_module("mlp3", mlp3)
            encoder_projection.add_module("relu3", nn.ReLU())
            encoder_projection.add_module("mlp4", mlp4)
            encoder_projection.add_module("relu4", nn.ReLU())
        elif projection_head[2] == '1':
            encoder.add_module("casEncoder", casEncoder)
            encoder.add_module("mlp1", mlp1)
            encoder.add_module("relu1", nn.ReLU())
            encoder_projection.add_module("encoder", encoder)

            encoder_projection.add_module("mlp2", mlp2)
            encoder_projection.add_module("relu2", nn.ReLU())
            encoder_projection.add_module("mlp3", mlp3)
            encoder_projection.add_module("relu3", nn.ReLU())
            encoder_projection.add_module("mlp4", mlp4)
            encoder_projection.add_module("relu4", nn.ReLU())
        elif projection_head[2] == '2':
            encoder.add_module("casEncoder", casEncoder)
            encoder.add_module("mlp1", mlp1)
            encoder.add_module("relu1", nn.ReLU())
            encoder.add_module("mlp2", mlp2)
            encoder.add_module("relu2", nn.ReLU())
            encoder_projection.add_module("encoder", encoder)

            encoder_projection.add_module("mlp3", mlp3)
            encoder_projection.add_module("relu3", nn.ReLU())
            encoder_projection.add_module("mlp4", mlp4)
            encoder_projection.add_module("relu4", nn.ReLU())
        elif projection_head[2] == '3':
            encoder.add_module("casEncoder", casEncoder)
            encoder.add_module("mlp1", mlp1)
            encoder.add_module("relu1", nn.ReLU())
            encoder.add_module("mlp2", mlp2)
            encoder.add_module("relu2", nn.ReLU())
            encoder.add_module("mlp3", mlp3)
            encoder.add_module("relu3", nn.ReLU())
            encoder_projection.add_module("encoder", encoder)

            encoder_projection.add_module("mlp4", mlp4)
            encoder_projection.add_module("relu4", nn.ReLU())
        elif projection_head[2] == '4':
            encoder.add_module("casEncoder", casEncoder)
            encoder.add_module("mlp1", mlp1)
            encoder.add_module("relu1", nn.ReLU())
            encoder.add_module("mlp2", mlp2)
            encoder.add_module("relu2", nn.ReLU())
            encoder.add_module("mlp3", mlp3)
            encoder.add_module("relu3", nn.ReLU())
            encoder.add_module("mlp4", mlp4)
            encoder.add_module("relu4", nn.ReLU())
            encoder_projection.add_module("encoder", encoder)
        else:
            print('Wrong projection head argument, should be [0-4]-[0-4].')
    else:
        print("Wrong projection head argument, should be [0-4]-[0-4].")

    encoder_projection = encoder_projection.to(device)

    summary(encoder, input_size=(max_seq, emb_dim), batch_size=batch_size)
    summary(encoder_projection, input_size=(max_seq, emb_dim), batch_size=batch_size)
    # optimize
    casEncoder_optimizer = torch.optim.Adam(casEncoder.parameters(), lr=lr, weight_decay=l2)
    mlp_optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)

    # criterion
    criterion = ContrastiveLoss(batch_size, temperature).to(device)

    # load data
    with open(FLAGS.input + 'data_aug_1_04.pkl', 'rb') as f:
        aug_1_04 = pickle.load(f)
    with open(FLAGS.input + 'data_aug_1_06.pkl', 'rb') as f:
        aug_1_06 = pickle.load(f)
    with open(FLAGS.input + 'data_aug_1_08.pkl', 'rb') as f:
        aug_1_08 = pickle.load(f)
    with open(FLAGS.input + 'data_aug_2_05.pkl', 'rb') as f:
        aug_2_05 = pickle.load(f)
    with open(FLAGS.input + 'data_aug_2_07.pkl', 'rb') as f:
        aug_2_07 = pickle.load(f)
    with open(FLAGS.input + 'data_aug_2_085.pkl', 'rb') as f:
        aug_2_085 = pickle.load(f)
    if FLAGS.use_unlabel:
        with open(FLAGS.input + 'data_unlabel_aug_1_04.pkl', 'rb') as f:
            unlabeled_aug_1_04 = pickle.load(f)
            aug_1_04.extend(unlabeled_aug_1_04)
        with open(FLAGS.input + 'data_unlabel_aug_1_06.pkl', 'rb') as f:
            unlabeled_aug_1_06 = pickle.load(f)
            aug_1_06.extend(unlabeled_aug_1_06)
        with open(FLAGS.input + 'data_unlabel_aug_1_08.pkl', 'rb') as f:
            unlabeled_aug_1_08 = pickle.load(f)
            aug_1_08.extend(unlabeled_aug_1_08)
        with open(FLAGS.input + 'data_unlabel_aug_2_05.pkl', 'rb') as f:
            unlabeled_aug_2_05 = pickle.load(f)
            aug_2_05.extend(unlabeled_aug_2_05)
        with open(FLAGS.input + 'data_unlabel_aug_2_07.pkl', 'rb') as f:
            unlabeled_aug_2_07 = pickle.load(f)
            aug_2_07.extend(unlabeled_aug_2_07)
        with open(FLAGS.input + 'data_unlabel_aug_2_085.pkl', 'rb') as f:
            unlabeled_aug_2_085 = pickle.load(f)
            aug_2_085.extend(unlabeled_aug_2_085)

    # print data information
    dataset_info = '# pre-training samples {}'
    print(dataset_info.format(len(aug_1_04)))

    # pre-training

    best_train_loss = 1000
    l1=[aug_1_04,aug_1_06,aug_1_08]
    l2=[aug_2_05,aug_2_07,aug_2_085]

    for epoch in range(epochs):
        time_start = time.time()
        #aug_1, aug_2 = shuffle_two(aug_1, aug_2)
        losses=[]
        random.shuffle(l1)
        random.shuffle(l2)
        l3=[list(t) for t in zip(l1,l2)]
        casEncoder_optimizer.zero_grad()
        mlp_optimizer.zero_grad()

        for aug_1,aug_2 in l3:
            aug_1,aug_2=shuffle_two(aug_1,aug_2)
            for i in range(len(aug_1) // batch_size):
                
                batch_aug_1 = copy.deepcopy(aug_1[batch_size * i:batch_size * i + batch_size])
                batch_aug_2 = copy.deepcopy(aug_2[batch_size * i:batch_size * i + batch_size])
                for batch_cascade_1 in batch_aug_1:
                    while len(batch_cascade_1) < max_seq:
                        batch_cascade_1.append(np.zeros(emb_dim))
                for batch_cascade_2 in batch_aug_2:
                    while len(batch_cascade_2) < max_seq:
                        batch_cascade_2.append(np.zeros(emb_dim))

                batch_aug_1 = torch.from_numpy(np.array(batch_aug_1)).float().to(device)
                batch_aug_2 = torch.from_numpy(np.array(batch_aug_2)).float().to(device)
                z_1 = encoder_projection(batch_aug_1)
                z_2 = encoder_projection(batch_aug_2)

                loss = criterion(z_1, z_2)
                loss.backward()
                casEncoder_optimizer.step()
                mlp_optimizer.step()
                losses.append(loss.item())
        epoch_loss = np.mean(losses)

        time_now = time.strftime("%Y-%m-%d, %H:%M", time.localtime())
        template = '{}: Pre-training Epoch {:3}, Time: {:.3f}s, {}, Train Loss: {:.4f}'
        print(template.format(FLAGS.name, epoch + 1, time.time() - time_start, time_now, epoch_loss))

        if epoch_loss < best_train_loss:
            best_train_loss = epoch_loss
            # save model
            torch.save({'model': encoder.state_dict()}, FLAGS.weight_path + FLAGS.name + '.pth')
            print('Model saved!')
    print('Finished! Time used: {:.3f}min'.format((time.time() - start_time) / 60))


if __name__ == '__main__':
    app.run(main)
