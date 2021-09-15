'''
@Project: DFP-GNN
@File   : ops_ft
@Time   : 2021/8/26 21:10
@Author : Shunxin Xiao
@Email  : xiaoshunxin.tj@gmail
@Desc
    Finetune module
'''
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import RMSprop
from sklearn.cluster import KMeans
from utils.ops_al import target_distribution
from utils.ops_ev import get_evaluation_results


def finetune(model, feature_list, adj_hat_list, ec_feature, adj_hat, adj_wave, norm, weight_tensor, optimizer_type,
             learning_rate, weight_decay, momentum, num_epochs, sp_weight, show_patience, update_interval,
             cl_weight, tolerance, max_loss_patience, labels, save_patience, dataset_name):
    print("###################### Finetune the whole DFP-GNN model ######################")
    model.cuda()
    for i in range(model.num_views):
        feature_list[i] = feature_list[i].cuda()
        adj_hat_list[i] = adj_hat_list[i].to_dense().cuda()
    adj_hat = adj_hat.cuda()
    adj_wave = adj_wave.cuda()
    weight_tensor = weight_tensor.cuda()
    ec_feature = torch.from_numpy(ec_feature).float().cuda()

    # construct the optimizer and reconstructed loss function
    if optimizer_type == "RMSprop":
        optimizer = RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    else:
        optimizer = RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    loss_function = nn.MSELoss()

    model.eval()
    hidden, _, _, _ = model(feature_list, adj_hat_list)
    embedding = hidden.detach().cpu().cpu()
    kmeans = KMeans(n_clusters=len(np.unique(labels)), n_init=5, n_jobs=8)
    y_pred = kmeans.fit_predict(embedding)
    ACC, NMI, Purity, ARI, P, R, F1 = get_evaluation_results(labels.numpy(), y_pred)
    print("Initial ACC score={:.4f}".format(ACC))
    y_pred_last = y_pred

    # begin to train...
    best_loss = float("inf")
    loss_patience = 0
    best_epoch = 0

    List_ID = []
    List_loss = []
    List_lr = []
    List_ls = []
    List_lc = []
    List_ACC = []
    List_NMI = []
    List_Purity = []
    List_ARI = []
    List_P = []
    List_R = []
    List_F = []

    for epoch in range(num_epochs):
        model.train()

        if epoch % update_interval == 0:
            _, _, _, tmp_q = model(feature_list, adj_hat_list)

            # update target distribution p
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)

            # evaluate clustering performance
            y_pred = tmp_q.cpu().numpy().argmax(1)
            # print(",".join(str(x) for x in y_pred))
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
            y_pred_last = y_pred

            ACC, NMI, Purity, ARI, P, R, F1 = get_evaluation_results(labels.numpy(), y_pred)
            print('Iter {}'.format(epoch), ':Acc {:.4f}'.format(ACC), ', f1 {:.4f}'.format(F1))

            if epoch > 0 and delta_label < tolerance:
                print('delta_label {:.4f}'.format(delta_label), '< tol', tolerance)
                print('Reached tolerance threshold. Stopping training.')
                break

        hidden, X_bar, A_bar, q = model(feature_list, adj_hat_list)

        optimizer.zero_grad()
        loss_ls = sp_weight * norm * F.binary_cross_entropy(A_bar.view(-1), adj_wave.to_dense().view(-1),
                                                         weight=weight_tensor)
        loss_lr = loss_function(X_bar, ec_feature)
        loss_lc = cl_weight * F.kl_div(q.log(), p)
        loss = loss_ls + loss_lr +loss_lc
        loss.backward()
        optimizer.step(closure=None)

        loss_value = float(loss.item())
        value_lr = float(loss_lr.item())
        value_ls = float(loss_ls.item())
        value_lc = float(loss_lc.item())

        if (epoch+1) % save_patience == 0:
            List_ID.append(epoch+1)
            List_loss.append(loss_value)
            List_lr.append(value_lr)
            List_ls.append(value_ls)
            List_lc.append(value_lc)
            with torch.no_grad():
                _, _, _, pred_q = model(feature_list, adj_hat_list)
                predicted = pred_q.data.cpu().numpy().argmax(1)
                ACC, NMI, Purity, ARI, P, R, F1 = get_evaluation_results(labels.numpy(), predicted)
                List_ACC.append(ACC)
                List_NMI.append(NMI)
                List_Purity.append(Purity)
                List_ARI.append(ARI)
                List_P.append(P)
                List_R.append(R)
                List_F.append(F1)

        # print(loss_value)
        if (epoch + 1) % show_patience == 0:
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss_value))

    model.eval()
    _, _, _, q = model(feature_list, adj_hat_list)

    # saving the training process
    # save_training_process(dataset_name, List_ID, List_loss, List_lr, List_ls, List_lc, List_ACC, List_NMI, List_Purity, List_ARI, List_P, List_R, List_F)

    return q.data.cpu().numpy().argmax(1)