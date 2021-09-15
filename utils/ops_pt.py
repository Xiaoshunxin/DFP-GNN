'''
@Project: DFP-GNN
@File   : ops_pt
@Time   : 2021/8/26 21:13
@Author : Shunxin Xiao
@Email  : xiaoshunxin.tj@gmail
@Desc
    Module of the pretraining process
'''
import os
import time
import torch
import numpy as np
import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F
from torch.optim import RMSprop
from sklearn.cluster import KMeans

from models.sgae import SGAE
from models.dfp_gnn import DFPGNN
from utils.ops_io import load_data, construct_adjacency_matrix, load_embedded_combined_data


def pretraining(args):
    labels, feature_list, adj_wave_list, adj_hat_list, norm_list, weight_tensor_list = load_data(
        direction_path=args.direction, dataset_name=args.dataset_name, normalization=args.normalization,
        load_saved=False, k_nearest_neighobrs=args.pm_knns, prunning_one=args.pm_pruning_one,
        prunning_two=args.pm_pruning_two, common_neighbors=args.pm_common_neighbors)

    args.num_classes = len(np.unique(labels))
    args.num_views = len(feature_list)
    view_dims = []
    for j in range(args.num_views):
        view_dims.append(feature_list[j].shape[1])
    print(view_dims)
    pm_hidden_dims = [args.pm_first_dim, args.pm_second_dim, args.pm_third_dim]
    sm_hidden_dims = [args.sm_first_dim, args.sm_second_dim, args.sm_third_dim]

    # pretraining or loading the weights of DFP-GNN
    ec_feat_save_direction = './data/ec_feature/'
    if not os.path.exists(ec_feat_save_direction):
        os.makedirs(ec_feat_save_direction)
    ec_feat_save_path = ec_feat_save_direction + args.dataset_name + '.npy'
    pt_weight_save_direction = './data/pt_weight/'
    if not os.path.exists(pt_weight_save_direction):
        os.makedirs(pt_weight_save_direction)
    pt_weight_save_path = pt_weight_save_direction + args.dataset_name + '.pkl'

    print("############### begin to pretraining all submodules ###############")
    model = DFPGNN(view_dims, pm_hidden_dims, sm_hidden_dims, args.num_classes)

    pt_begin_time = time.time()
    ec_feature = pre_training_pm(model=model, feature_list=feature_list, adj_wave_list=adj_wave_list,
                                 adj_hat_list=adj_hat_list, norm_list=norm_list,
                                 weight_tensor_list=weight_tensor_list, optimizer_type=args.pt_pm_optimizer,
                                 learning_rate=args.pt_pm_lr, momentum=args.pt_pm_momentum,
                                 weight_decay=args.pt_pm_weight_decay, num_epochs=args.pt_pm_num_epochs,
                                 sp_weight=args.pt_pm_sp_weight, max_loss_patience=args.pt_pm_loss_patience,
                                 show_patience=args.pt_pm_show_patience)

    # saving and loading the embedded combine0d feature
    np.save(ec_feat_save_path, ec_feature)

    # constructing the adjacent matrix from the embedded combined feature
    af_adj_hat = pre_training_fm(feature=ec_feature, dataset_name=args.dataset_name,
                                 pruning_one=args.fm_pruning_one,
                                 k_nearest_neighbors=args.fm_knns, pruning_two=args.fm_pruning_two,
                                 common_neighbors=args.fm_common_neighbors, adj_hat_list=adj_hat_list)

    # pretraining the shared module
    pre_training_sm(model=model, feature=ec_feature, labels=labels, dataset_name=args.dataset_name,
                    optimizer_type=args.pt_sm_optimizer, learning_rate=args.pt_sm_lr, momentum=args.pt_sm_momentum,
                    weight_decay=args.pt_sm_weight_decay, num_epochs=args.pt_sm_num_epochs,
                    max_loss_patience=args.pt_sm_loss_patience, show_patience=args.pt_sm_show_patience,
                    sp_weight=args.pt_sm_sp_weight, af_adj_hat=af_adj_hat)
    pt_cost_time = time.time() - pt_begin_time
    print("Pretraining time: ", pt_cost_time)
    # saving the weights of DFP-GNN
    torch.save(model.state_dict(), pt_weight_save_path)


def pre_training_pm(model, feature_list, adj_wave_list, adj_hat_list, norm_list, weight_tensor_list, optimizer_type,
                    learning_rate, momentum, weight_decay, num_epochs, sp_weight, max_loss_patience, show_patience):

    print("###################### Pretraining the preliminary module ######################")
    embedding_list = []
    for i in range(model.num_views):
        print("Begin to train the " + str(i+1) + " initial autoencoder ...")
        feature = feature_list[i].cuda()
        adj_wave = adj_wave_list[i].cuda()
        adj_hat = adj_hat_list[i].cuda()
        norm = norm_list[i]
        weight_tensor = weight_tensor_list[i].cuda()

        # construct the dimension list for each view
        temp_dims = []
        temp_dims.append(model.view_dims[i])
        temp_dims.extend(model.pm_hidden_dims)

        # construct the SGAE model for each view
        sgae = SGAE(dims=temp_dims, act_func=nn.ReLU()).cuda()

        # construct the optimizer and reconstructed loss function
        if optimizer_type == "RMSprop":
            optimizer = RMSprop(sgae.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
        else:
            optimizer = RMSprop(sgae.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
        loss_function = nn.MSELoss()

        # begin to train...
        best_loss = float("inf")
        loss_patience = 0
        best_epoch = 0
        for epoch in range(num_epochs):
            sgae.train()
            hidden, X_bar, A_bar = sgae(feature, adj_hat)

            optimizer.zero_grad()
            loss = sp_weight * norm * F.binary_cross_entropy(A_bar.view(-1), adj_wave.to_dense().view(-1), weight=weight_tensor)
            loss += loss_function(X_bar, feature)
            loss.backward()
            optimizer.step(closure=None)

            loss_value = float(loss.item())
            if loss_value < best_loss:
                best_loss = loss_value
                best_epoch = epoch + 1
                loss_patience = 0
            else:
                loss_patience += 1

            if loss_patience > max_loss_patience:
                print("Break by loss patience!")
                print("Best Epoch:", '%04d' % (best_epoch), "best loss=", "{:.5f}".format(best_loss))
                break
            if (epoch + 1) % show_patience == 0:
                print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss_value))

        # obtain the embedded feature
        sgae.eval()
        hidden, _, _ = sgae(feature, adj_hat)
        embedding = hidden.detach().cpu().numpy()
        embedding_list.append(embedding)

        # copy the trained weights to the corresponding part of the DFP-GNN model
        model_dict = model.get_preliminary_ae(i).state_dict()
        model_dict.update(sgae.encoder.state_dict())
        model.get_preliminary_ae(i).load_state_dict(model_dict)

    np_list = np.array(embedding_list)
    ec_feature = np.mean(np_list, axis=0)
    return ec_feature


def pre_training_fm(feature, dataset_name, adj_hat_list, k_nearest_neighbors, pruning_one, pruning_two, common_neighbors):
    print("###################### Pretraining the fused module ######################")
    save_direction = './data/adj_matrix/' + dataset_name + '/'
    if not os.path.exists(save_direction):
        os.makedirs(save_direction)

    print("Constructing the adjacency matrix of " + dataset_name + " for the embedded combined feature ......")
    adj, adj_wave, adj_hat = construct_adjacency_matrix(feature, k_nearest_neighbors, pruning_one,
                                                        pruning_two, common_neighbors)
    # save these scale and matrix
    print("Saving the adjacency matrix to " + save_direction)
    sp.save_npz(save_direction + 'ec_adj.npz', adj)
    sp.save_npz(save_direction + 'ec_adj_wave.npz', adj_wave)
    sp.save_npz(save_direction + 'ec_adj_hat.npz', adj_hat)

    af_adj_hat = adj_hat_list[0]
    for i in range(1, len(adj_hat_list)):
        af_adj_hat += adj_hat_list[i]
    af_adj_hat = af_adj_hat / len(adj_hat_list)
    return af_adj_hat


def pre_training_sm(model, feature, labels, dataset_name, optimizer_type, learning_rate, momentum, weight_decay,
                    num_epochs, max_loss_patience, show_patience, sp_weight, af_adj_hat):
    print("###################### Pretraining the shared module ######################")
    # obtaining all related data
    adj_wave, adj_hat, norm, weight_tensor = load_embedded_combined_data(dataset_name)
    adj_wave = adj_wave.cuda()
    adj_hat = adj_hat.cuda()
    weight_tensor = weight_tensor.cuda()
    feature = torch.from_numpy(feature).float().cuda()
    af_adj_hat = af_adj_hat.cuda()

    # construct the dimension list for shared module
    temp_dims = []
    temp_dims.append(model.pm_hidden_dims[-1])
    temp_dims.extend(model.sm_hidden_dims)

    # construct the SGAE model for shared module
    sgae = SGAE(dims=temp_dims, act_func=nn.ReLU()).cuda()

    # construct the optimizer and reconstructed loss function
    if optimizer_type == "RMSprop":
        optimizer = RMSprop(sgae.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    else:
        optimizer = RMSprop(sgae.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    loss_function = nn.MSELoss()

    # begin to train...
    best_loss = float("inf")
    loss_patience = 0
    best_epoch = 0
    for epoch in range(num_epochs):
        sgae.train()
        hidden, X_bar, A_bar = sgae(feature, af_adj_hat)

        optimizer.zero_grad()
        loss = sp_weight * norm * F.binary_cross_entropy(A_bar.view(-1), adj_wave.to_dense().view(-1), weight=weight_tensor)
        loss += loss_function(X_bar, feature)
        loss.backward()
        optimizer.step(closure=None)

        loss_value = float(loss.item())
        if loss_value < best_loss:
            best_loss = loss_value
            best_epoch = epoch + 1
            loss_patience = 0
        else:
            loss_patience += 1

        if loss_patience > max_loss_patience:
            print("Break by loss patience!")
            print("Best Epoch:", '%04d' % (best_epoch), "best loss=", "{:.5f}".format(best_loss))
            break
        if (epoch + 1) % show_patience == 0:
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss_value))

    sgae.eval()
    hidden, _, _ = sgae(feature, adj_hat)
    embedding = hidden.detach().cpu().cpu()
    kmeans = KMeans(n_clusters=len(np.unique(labels)), n_init=5, n_jobs=8)
    _ = kmeans.fit_predict(embedding)

    # copy the trained weights to the corresponding part of the DFP-GNN model
    model.cluster_layer.network.data = torch.tensor(kmeans.cluster_centers_).float().cuda()
    model_dict = model.shared_module.state_dict()
    model_dict.update(sgae.state_dict())
    model.shared_module.load_state_dict(model_dict)