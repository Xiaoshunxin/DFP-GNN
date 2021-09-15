'''
@Project: DFP-GNN
@File   : test_train
@Time   : 2021/9/12 15:10
@Author : Shunxin Xiao
@Email  : xiaoshunxin.tj@gmail
@Desc
    main function
'''
import os
import time
import torch
import random
import warnings
warnings.filterwarnings('ignore')
import argparse
import numpy as np

from models.dfp_gnn import DFPGNN
from utils.ops_ft import finetune
from utils.ops_pt import pretraining
from utils.ops_ev import get_evaluation_results
from utils.ops_io import load_data, load_embedded_combined_data


if __name__ == '__main__':
    # Configuration settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=True, help='Disables CUDA training.')
    parser.add_argument('--cuda_device', type=str, default='2', help='The number of cuda device.')
    parser.add_argument('--n_repeated', type=int, default=5, help='Number of repeated experiments')
    parser.add_argument('--direction', type=str, default='./data/datasets/', help='direction of datasets')
    parser.add_argument('--dataset_name', type=str, default='BBCSport', help='The dataset used for training/testing')
    parser.add_argument('--normalization', type=str, default='normalize', help='default normalize')

    parser.add_argument('--pm_first_dim', type=int, default=512, help='the dim of the first layer in PM')
    parser.add_argument('--pm_second_dim', type=int, default=2048, help='the dim of the second layer in PM')
    parser.add_argument('--pm_third_dim', type=int, default=256, help='the dim of the third layer in PM')
    parser.add_argument('--pm_knns', type=int, default=50, help='the number of nearest neighbors in PM')
    parser.add_argument('--pm_pruning_one', action='store_true', default=True, help='Whether to use prunning one in PM')
    parser.add_argument('--pm_pruning_two', action='store_true', default=True, help='Whether to use prunning two in PM')
    parser.add_argument('--pm_common_neighbors', type=int, default=2, help='threshold of common neighbors in PM')
    parser.add_argument('--pt_pm_optimizer', type=str, default='RMSprop', help='The optimizer type in pretraining stage')
    parser.add_argument('--pt_pm_lr', type=float, default=0.00001, help='learning rate in pretraining stage.')
    parser.add_argument('--pt_pm_momentum', type=float, default=0.9, help='value of pretraining momentum.')
    parser.add_argument('--pt_pm_weight_decay', type=float, default=0.000001, help='value of layer-wise weight decay.')
    parser.add_argument('--pt_pm_num_epochs', type=int, default=20000, help='number of layer-wise training epochs.')
    parser.add_argument('--pt_pm_loss_patience', type=int, default=100, help='value of loss patience in pretraining')
    parser.add_argument('--pt_pm_show_patience', type=int, default=100, help='value of show patience in pretraining')
    parser.add_argument('--pt_pm_sp_weight', type=float, default=0.001, help='weight of structure preservation loss')

    parser.add_argument('--sm_first_dim', type=int, default=256, help='the dim of the first layer in SM')
    parser.add_argument('--sm_second_dim', type=int, default=64, help='the dim of the second layer in SM')
    parser.add_argument('--sm_third_dim', type=int, default=16, help='the dim of the third layer in SM')
    parser.add_argument('--fm_knns', type=int, default=40, help='the number of nearest neighbors in FM')
    parser.add_argument('--fm_pruning_one', action='store_true', default=True, help='Whether to use prunning one in FM')
    parser.add_argument('--fm_pruning_two', action='store_true', default=True, help='Whether to use prunning two in FM')
    parser.add_argument('--fm_common_neighbors', type=int, default=2, help='threshold of common neighbors in FM')
    parser.add_argument('--pt_sm_optimizer', type=str, default='RMSprop', help='The optimizer type in pretraining stage')
    parser.add_argument('--pt_sm_lr', type=float, default=0.00001, help='learning rate in pretraining stage.')
    parser.add_argument('--pt_sm_momentum', type=float, default=0.9, help='value of pretraining momentum.')
    parser.add_argument('--pt_sm_weight_decay', type=float, default=0.000001, help='value of layer-wise weight decay.')
    parser.add_argument('--pt_sm_num_epochs', type=int, default=20000, help='number of layer-wise training epochs.')
    parser.add_argument('--pt_sm_loss_patience', type=int, default=100, help='value of loss patience in pretraining')
    parser.add_argument('--pt_sm_show_patience', type=int, default=100, help='value of show patience in pretraining')
    parser.add_argument('--pt_sm_sp_weight', type=float, default=0.001, help='weight of structure preservation loss')

    parser.add_argument('--ft_optimizer', type=str, default='RMSprop', help='The optimizer type in pretraining stage')
    parser.add_argument('--ft_lr', type=float, default=0.00001, help='learning rate in pretraining stage.')
    parser.add_argument('--ft_momentum', type=float, default=0.9, help='value of pretraining momentum.')
    parser.add_argument('--ft_weight_decay', type=float, default=0.00001, help='value of layer-wise weight decay.')
    parser.add_argument('--ft_num_epochs', type=int, default=20000, help='number of layer-wise training epochs.')
    parser.add_argument('--ft_loss_patience', type=int, default=100, help='value of show patience in pretraining')
    parser.add_argument('--ft_show_patience', type=int, default=100, help='value of show patience in pretraining')
    parser.add_argument('--ft_sp_weight', type=float, default=0.00001, help='weight of structure preservation loss')
    parser.add_argument('--ft_update_interval', type=int, default=50, help='weight of structure preservation loss')
    parser.add_argument('--ft_cl_weight', type=float, default=1.0, help='weight of clustering loss')
    parser.add_argument('--ft_tolerance', type=float, default=0.0000001, help='weight of clustering loss')

    parser.add_argument('--save_patience', type=int, default=20001, help='the frequency about saving the result')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device

    all_ACC = []
    all_NMI = []
    all_Purity = []
    all_ARI = []
    all_F = []
    all_P = []
    all_R = []
    all_PT_TIME = []
    all_FT_TIME = []

    for i in range(args.n_repeated):

        pretraining(args)

        labels, feature_list, adj_wave_list, adj_hat_list, norm_list, weight_tensor_list = load_data(
            direction_path=args.direction, dataset_name=args.dataset_name, normalization=args.normalization,
            load_saved=True, k_nearest_neighobrs=args.pm_knns, prunning_one=args.pm_pruning_one,
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
        pt_weight_save_path = pt_weight_save_direction +args.dataset_name + '.pkl'

        # exit()
        print("############### loading the pretrained wieghts.... ###############")
        model = DFPGNN(view_dims, pm_hidden_dims, sm_hidden_dims, args.num_classes)
        model.load_state_dict(torch.load(pt_weight_save_path))
        ec_feature = np.load(ec_feat_save_path)
        adj_wave, adj_hat, norm, weight_tensor = load_embedded_combined_data(args.dataset_name)

        # finetune the whole DFP-GNN model
        ft_begin_time = time.time()
        predicted = finetune(model=model, feature_list=feature_list, adj_hat_list=adj_hat_list, ec_feature=ec_feature,
                             adj_hat=adj_hat, adj_wave=adj_wave, norm=norm, weight_tensor=weight_tensor,
                             optimizer_type=args.ft_optimizer, learning_rate=args.ft_lr, momentum=args.ft_momentum,
                             weight_decay=args.ft_weight_decay, num_epochs=args.ft_num_epochs, sp_weight=args.ft_sp_weight,
                             show_patience=args.ft_show_patience, update_interval=args.ft_update_interval,
                             cl_weight=args.ft_cl_weight, labels=labels, tolerance=args.ft_tolerance,
                             max_loss_patience=args.ft_loss_patience, save_patience=args.save_patience,
                             dataset_name=args.dataset_name)
        ft_cost_time = time.time() - ft_begin_time
        ACC, NMI, Purity, ARI, P, R, F1 = get_evaluation_results(labels.numpy(), predicted)

        # pred_save_path = './data/pred/' + args.dataset_name + '.mat'
        # sio.savemat(pred_save_path, {'pred': predicted})

        all_ACC.append(ACC)
        all_NMI.append(NMI)
        all_Purity.append(Purity)
        all_ARI.append(ARI)
        all_P.append(P)
        all_R.append(R)
        all_F.append(F1)
        all_FT_TIME.append(ft_cost_time)

    # append result to .txt file
    fp = open("results.txt", "a+", encoding="utf-8")
    # fp = open("results_" + args.dataset_name + ".txt", "a+", encoding="utf-8")
    fp.write("dataset_name: {}\n".format(args.dataset_name))
    fp.write("ft_sp_weight: {}\n".format(args.ft_sp_weight))
    fp.write("ft_cl_weight: {}\n".format(args.ft_cl_weight))
    fp.write("ACC: {:.2f}\t{:.2f}\n".format(np.mean(all_ACC) * 100, np.std(all_ACC) * 100))
    fp.write("NMI: {:.2f}\t{:.2f}\n".format(np.mean(all_NMI) * 100, np.std(all_NMI) * 100))
    fp.write("Purity: {:.2f}\t{:.2f}\n".format(np.mean(all_Purity) * 100, np.std(all_Purity) * 100))
    fp.write("ARI: {:.2f}\t{:.2f}\n".format(np.mean(all_ARI) * 100, np.std(all_ARI) * 100))
    fp.write("P: {:.2f}\t{:.2f}\n".format(np.mean(all_P) * 100, np.std(all_P) * 100))
    fp.write("R: {:.2f}\t{:.2f}\n".format(np.mean(all_R) * 100, np.std(all_R) * 100))
    fp.write("F: {:.2f}\t{:.2f}\n".format(np.mean(all_F) * 100, np.std(all_F) * 100))
    # fp.write("Pretrain Time: {:.2f}\t{:.2f}\n".format(np.mean(all_PT_TIME), np.std(all_PT_TIME)))
    fp.write("Finetune Time: {:.2f}\t{:.2f}\n\n".format(np.mean(all_FT_TIME), np.std(all_FT_TIME)))
    fp.close()
'''
@Project: DFP-GNN
@File   : test_train
@Time   : 2021/9/12 15:10
@Author : Shunxin Xiao
@Email  : xiaoshunxin.tj@gmail
@Desc
    main function
'''
import os
import time
import torch
import random
import warnings
warnings.filterwarnings('ignore')
import argparse
import numpy as np

from models.dfp_gnn import DFPGNN
from utils.ops_ft import finetune
from utils.ops_pt import pretraining
from utils.ops_ev import get_evaluation_results
from utils.ops_io import load_data, load_embedded_combined_data


if __name__ == '__main__':
    # Configuration settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=True, help='Disables CUDA training.')
    parser.add_argument('--cuda_device', type=str, default='2', help='The number of cuda device.')
    parser.add_argument('--n_repeated', type=int, default=5, help='Number of repeated experiments')
    parser.add_argument('--direction', type=str, default='./data/datasets/', help='direction of datasets')
    parser.add_argument('--dataset_name', type=str, default='BBCSport', help='The dataset used for training/testing')
    parser.add_argument('--normalization', type=str, default='normalize', help='default normalize')

    parser.add_argument('--pm_first_dim', type=int, default=512, help='the dim of the first layer in PM')
    parser.add_argument('--pm_second_dim', type=int, default=2048, help='the dim of the second layer in PM')
    parser.add_argument('--pm_third_dim', type=int, default=256, help='the dim of the third layer in PM')
    parser.add_argument('--pm_knns', type=int, default=50, help='the number of nearest neighbors in PM')
    parser.add_argument('--pm_pruning_one', action='store_true', default=True, help='Whether to use prunning one in PM')
    parser.add_argument('--pm_pruning_two', action='store_true', default=True, help='Whether to use prunning two in PM')
    parser.add_argument('--pm_common_neighbors', type=int, default=2, help='threshold of common neighbors in PM')
    parser.add_argument('--pt_pm_optimizer', type=str, default='RMSprop', help='The optimizer type in pretraining stage')
    parser.add_argument('--pt_pm_lr', type=float, default=0.00001, help='learning rate in pretraining stage.')
    parser.add_argument('--pt_pm_momentum', type=float, default=0.9, help='value of pretraining momentum.')
    parser.add_argument('--pt_pm_weight_decay', type=float, default=0.000001, help='value of layer-wise weight decay.')
    parser.add_argument('--pt_pm_num_epochs', type=int, default=20000, help='number of layer-wise training epochs.')
    parser.add_argument('--pt_pm_loss_patience', type=int, default=100, help='value of loss patience in pretraining')
    parser.add_argument('--pt_pm_show_patience', type=int, default=100, help='value of show patience in pretraining')
    parser.add_argument('--pt_pm_sp_weight', type=float, default=0.001, help='weight of structure preservation loss')

    parser.add_argument('--sm_first_dim', type=int, default=256, help='the dim of the first layer in SM')
    parser.add_argument('--sm_second_dim', type=int, default=64, help='the dim of the second layer in SM')
    parser.add_argument('--sm_third_dim', type=int, default=16, help='the dim of the third layer in SM')
    parser.add_argument('--fm_knns', type=int, default=40, help='the number of nearest neighbors in FM')
    parser.add_argument('--fm_pruning_one', action='store_true', default=True, help='Whether to use prunning one in FM')
    parser.add_argument('--fm_pruning_two', action='store_true', default=True, help='Whether to use prunning two in FM')
    parser.add_argument('--fm_common_neighbors', type=int, default=2, help='threshold of common neighbors in FM')
    parser.add_argument('--pt_sm_optimizer', type=str, default='RMSprop', help='The optimizer type in pretraining stage')
    parser.add_argument('--pt_sm_lr', type=float, default=0.00001, help='learning rate in pretraining stage.')
    parser.add_argument('--pt_sm_momentum', type=float, default=0.9, help='value of pretraining momentum.')
    parser.add_argument('--pt_sm_weight_decay', type=float, default=0.000001, help='value of layer-wise weight decay.')
    parser.add_argument('--pt_sm_num_epochs', type=int, default=20000, help='number of layer-wise training epochs.')
    parser.add_argument('--pt_sm_loss_patience', type=int, default=100, help='value of loss patience in pretraining')
    parser.add_argument('--pt_sm_show_patience', type=int, default=100, help='value of show patience in pretraining')
    parser.add_argument('--pt_sm_sp_weight', type=float, default=0.001, help='weight of structure preservation loss')

    parser.add_argument('--ft_optimizer', type=str, default='RMSprop', help='The optimizer type in pretraining stage')
    parser.add_argument('--ft_lr', type=float, default=0.00001, help='learning rate in pretraining stage.')
    parser.add_argument('--ft_momentum', type=float, default=0.9, help='value of pretraining momentum.')
    parser.add_argument('--ft_weight_decay', type=float, default=0.00001, help='value of layer-wise weight decay.')
    parser.add_argument('--ft_num_epochs', type=int, default=20000, help='number of layer-wise training epochs.')
    parser.add_argument('--ft_loss_patience', type=int, default=100, help='value of show patience in pretraining')
    parser.add_argument('--ft_show_patience', type=int, default=100, help='value of show patience in pretraining')
    parser.add_argument('--ft_sp_weight', type=float, default=0.00001, help='weight of structure preservation loss')
    parser.add_argument('--ft_update_interval', type=int, default=50, help='weight of structure preservation loss')
    parser.add_argument('--ft_cl_weight', type=float, default=1.0, help='weight of clustering loss')
    parser.add_argument('--ft_tolerance', type=float, default=0.0000001, help='weight of clustering loss')

    parser.add_argument('--save_patience', type=int, default=20001, help='the frequency about saving the result')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device

    all_ACC = []
    all_NMI = []
    all_Purity = []
    all_ARI = []
    all_F = []
    all_P = []
    all_R = []
    all_PT_TIME = []
    all_FT_TIME = []

    for i in range(args.n_repeated):

        pretraining(args)

        labels, feature_list, adj_wave_list, adj_hat_list, norm_list, weight_tensor_list = load_data(
            direction_path=args.direction, dataset_name=args.dataset_name, normalization=args.normalization,
            load_saved=True, k_nearest_neighobrs=args.pm_knns, prunning_one=args.pm_pruning_one,
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
        pt_weight_save_path = pt_weight_save_direction +args.dataset_name + '.pkl'

        # exit()
        print("############### loading the pretrained wieghts.... ###############")
        model = DFPGNN(view_dims, pm_hidden_dims, sm_hidden_dims, args.num_classes)
        model.load_state_dict(torch.load(pt_weight_save_path))
        ec_feature = np.load(ec_feat_save_path)
        adj_wave, adj_hat, norm, weight_tensor = load_embedded_combined_data(args.dataset_name)

        # finetune the whole DFP-GNN model
        ft_begin_time = time.time()
        predicted = finetune(model=model, feature_list=feature_list, adj_hat_list=adj_hat_list, ec_feature=ec_feature,
                             adj_hat=adj_hat, adj_wave=adj_wave, norm=norm, weight_tensor=weight_tensor,
                             optimizer_type=args.ft_optimizer, learning_rate=args.ft_lr, momentum=args.ft_momentum,
                             weight_decay=args.ft_weight_decay, num_epochs=args.ft_num_epochs, sp_weight=args.ft_sp_weight,
                             show_patience=args.ft_show_patience, update_interval=args.ft_update_interval,
                             cl_weight=args.ft_cl_weight, labels=labels, tolerance=args.ft_tolerance,
                             max_loss_patience=args.ft_loss_patience, save_patience=args.save_patience,
                             dataset_name=args.dataset_name)
        ft_cost_time = time.time() - ft_begin_time
        ACC, NMI, Purity, ARI, P, R, F1 = get_evaluation_results(labels.numpy(), predicted)

        # pred_save_path = './data/pred/' + args.dataset_name + '.mat'
        # sio.savemat(pred_save_path, {'pred': predicted})

        all_ACC.append(ACC)
        all_NMI.append(NMI)
        all_Purity.append(Purity)
        all_ARI.append(ARI)
        all_P.append(P)
        all_R.append(R)
        all_F.append(F1)
        all_FT_TIME.append(ft_cost_time)

    # append result to .txt file
    fp = open("results.txt", "a+", encoding="utf-8")
    # fp = open("results_" + args.dataset_name + ".txt", "a+", encoding="utf-8")
    fp.write("dataset_name: {}\n".format(args.dataset_name))
    fp.write("ft_sp_weight: {}\n".format(args.ft_sp_weight))
    fp.write("ft_cl_weight: {}\n".format(args.ft_cl_weight))
    fp.write("ACC: {:.2f}\t{:.2f}\n".format(np.mean(all_ACC) * 100, np.std(all_ACC) * 100))
    fp.write("NMI: {:.2f}\t{:.2f}\n".format(np.mean(all_NMI) * 100, np.std(all_NMI) * 100))
    fp.write("Purity: {:.2f}\t{:.2f}\n".format(np.mean(all_Purity) * 100, np.std(all_Purity) * 100))
    fp.write("ARI: {:.2f}\t{:.2f}\n".format(np.mean(all_ARI) * 100, np.std(all_ARI) * 100))
    fp.write("P: {:.2f}\t{:.2f}\n".format(np.mean(all_P) * 100, np.std(all_P) * 100))
    fp.write("R: {:.2f}\t{:.2f}\n".format(np.mean(all_R) * 100, np.std(all_R) * 100))
    fp.write("F: {:.2f}\t{:.2f}\n".format(np.mean(all_F) * 100, np.std(all_F) * 100))
    # fp.write("Pretrain Time: {:.2f}\t{:.2f}\n".format(np.mean(all_PT_TIME), np.std(all_PT_TIME)))
    fp.write("Finetune Time: {:.2f}\t{:.2f}\n\n".format(np.mean(all_FT_TIME), np.std(all_FT_TIME)))
    fp.close()
