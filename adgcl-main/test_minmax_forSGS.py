import argparse
import logging
import random

import numpy as np
import torch
from ogb.graphproppred import Evaluator
from ogb.graphproppred import PygGraphPropPredDataset
from sklearn.linear_model import Ridge, LogisticRegression
from torch_geometric.data import DataLoader
from torch_geometric.transforms import Compose
from torch_scatter import scatter
import torch.nn as nn
import time

from unsupervised.embedding_evaluation import EmbeddingEvaluation
from unsupervised.encoder import MoleculeEncoder
from torch_geometric.loader import ClusterData, ClusterLoader

from sklearn.metrics import f1_score


from unsupervised.learning import GInfoMinMax
from unsupervised.utils import initialize_edge_weight
from unsupervised.view_learner import ViewLearner

import DeviceDir

DIR, RESULTS_DIR = DeviceDir.get_directory()
device, NUM_PROCESSORS = DeviceDir.get_device()
from ipynb.fs.full.SGSLoadDataset import LOAD_DATASET
from unsupervised.encoder.feed_forward_encoder import EncoderFeedForward



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def calculate_f1(logits, labels, mask):
    preds = logits[mask].argmax(dim=1)
    f1 = f1_score(labels[mask].cpu(), preds.cpu(), average='micro')
    return f1


def evaluate(model, cluster_loader, device, q=500, mode=None):
    model.eval()
    total_train_f1, total_val_f1, total_test_f1 = 0, 0, 0
    num_train_examples, num_val_examples, num_test_examples = 0, 0, 0

    with torch.no_grad():
        for batch in cluster_loader:
            batch = batch.to(device)
            
            x, out = model.encoder(batch, batch.x, batch.edge_index)
            
            if batch.train_mask.any():
                train_f1 = calculate_f1(out, batch.y, batch.train_mask)
                total_train_f1 += train_f1 * batch.train_mask.sum().item()
                num_train_examples += batch.train_mask.sum().item()

            if batch.val_mask.any():
                val_f1 = calculate_f1(out, batch.y, batch.val_mask)
                total_val_f1 += val_f1 * batch.val_mask.sum().item()
                num_val_examples += batch.val_mask.sum().item()

            if batch.test_mask.any():
                test_f1 = calculate_f1(out, batch.y, batch.test_mask)
                total_test_f1 += test_f1 * batch.test_mask.sum().item()
                num_test_examples += batch.test_mask.sum().item()

    avg_train_f1 = total_train_f1 / num_train_examples if num_train_examples > 0 else 0
    avg_val_f1 = total_val_f1 / num_val_examples if num_val_examples > 0 else 0
    avg_test_f1 = total_test_f1 / num_test_examples if num_test_examples > 0 else 0

    return avg_train_f1, avg_val_f1, avg_test_f1

import os

def get_loader(data, DATASET_NAME, sample_percent=0.20):
    use_metis = data.edge_index.shape[1]>=100000 # or data.num_nodes >= 20000

    # use_metis = True

    if use_metis:
        num_edges_per_partition = 100000
        num_parts = int(np.ceil(data.edge_index.shape[1] / num_edges_per_partition))    
        
        # num_parts = 256
        # num_edges_per_partition = int(data.edge_index.shape[1] / num_parts)

        sample_size = int(num_edges_per_partition * sample_percent)
        print("Using METIS with num_parts:", num_parts, "avg edges per partition:", num_edges_per_partition, "sample size:", sample_size)
    else:
        print("Graph has fewer than 10,000 nodes; don't do graph partition")
        sample_size = int(data.edge_index.shape[1] * sample_percent)

    # Partition data using METIS clustering or load the entire graph as a single batch
    cluster_loader = [data]
    
    if use_metis:
        start = time.time()
        cluster_dir = DIR + 'tmp/' + DATASET_NAME
        print(cluster_dir)
        if not os.path.exists(cluster_dir):
            os.makedirs(cluster_dir)
        
        cluster_data = ClusterData(data, num_parts=num_parts, recursive=False, save_dir=cluster_dir, log=True)
        print("METIS Partition time:", time.time() - start)
        cluster_loader = ClusterLoader(cluster_data, batch_size=1, shuffle=True)

    
    return cluster_loader


def run(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Using Device: %s" % device)
    logging.info("Seed: %d" % args.seed)
    logging.info(args)
    

    #evaluator = Evaluator(name=args.dataset)
    #my_transforms = Compose([initialize_edge_weight])
    #dataset = PygGraphPropPredDataset(name=args.dataset, root='/scratch/gilbreth/das90/Dataset/', transform=my_transforms)

    DATASET_NAME = args.dataset
    data, dataset  = LOAD_DATASET(DIR, DATASET_NAME)
    num_classes = max(data.y).item()+1

    print(data, num_classes)


    # split_idx = dataset.get_idx_split()
    # train_loader = DataLoader(dataset[split_idx["train"]], batch_size=128, shuffle=True)
    # valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=128, shuffle=False)
    # test_loader = DataLoader(dataset[split_idx["test"]], batch_size=128, shuffle=False)

    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # model = GInfoMinMax(MoleculeEncoder(emb_dim=args.emb_dim, num_gc_layers=args.num_gc_layers, drop_ratio=args.drop_ratio, pooling_type=args.pooling_type),
    #                 proj_hidden_dim=args.emb_dim).to(device)


    encoder = EncoderFeedForward(num_features=data.x.shape[1], dim = 256, num_gc_layers=2, num_fc_layers=2, out_features=num_classes, dropout=0.2)
    # print(encoder)

    model = GInfoMinMax(encoder).to(device)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=args.model_lr)

    gEncoder = EncoderFeedForward(num_features=data.x.shape[1], dim = 256, num_gc_layers=2, num_fc_layers=2, out_features=num_classes, dropout=0.2)

    # print(gEncoder)

    view_learner = ViewLearner(gEncoder, mlp_edge_model_dim=64).to(device)
    view_optimizer = torch.optim.Adam(view_learner.parameters(), lr=args.view_lr)

    # if 'classification' in dataset.task_type:
    #     ee = EmbeddingEvaluation(LogisticRegression(dual=False, fit_intercept=True, max_iter=5000),
    #                              evaluator, dataset.task_type, dataset.num_tasks, device, params_dict=None,
    #                              param_search=True)
    # elif 'regression' in dataset.task_type:
    #     ee = EmbeddingEvaluation(Ridge(fit_intercept=True, normalize=True, copy_X=True, max_iter=5000),
    #                              evaluator, dataset.task_type, dataset.num_tasks, device, params_dict=None,
    #                              param_search=True)
    # else:
    #     raise NotImplementedError

    print(model)
    print(view_learner)

    # model.eval()
    # train_score, val_score, test_score = ee.embedding_evaluation(model.encoder, train_loader, valid_loader, test_loader)
    # logging.info(
    #     "Before training Embedding Eval Scores: Train: {} Val: {} Test: {}".format(train_score, val_score,
    #                                                                                      test_score))

    model_losses = []
    view_losses = []
    view_regs = []
    valid_curve = []
    test_curve = []
    train_curve = []

    dataloader = get_loader(data, DATASET_NAME)
    criterion = nn.CrossEntropyLoss()

    best_test_f1 = 0

    for epoch in range(1, args.epochs + 1):

        model_loss_all = 0
        view_loss_all = 0
        reg_all = 0


        for batch in dataloader:
            # set up
            batch = batch.to(device)

            # train view to maximize contrastive loss
            view_learner.train()
            view_learner.zero_grad()
            
            model.eval()
            x, _ = model(batch.batch, batch.x, batch.edge_index, batch.edge_attr, None)

            # print(x.shape)
            # return 

            edge_logits = view_learner(batch.batch, batch.x, batch.edge_index, batch.edge_attr)

            temperature = 1.0
            bias = 0.0 + 0.0001  # If bias is 0, we run into problems
            eps = (bias - (1 - bias)) * torch.rand(edge_logits.size()) + (1 - bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = gate_inputs.to(device)
            gate_inputs = (gate_inputs + edge_logits) / temperature
            batch_aug_edge_weight = torch.sigmoid(gate_inputs).squeeze()

            x_aug, out = model(batch.batch, batch.x, batch.edge_index, batch.edge_attr, batch_aug_edge_weight)

            # regularization

            edge_drop_out_prob = 1 - batch_aug_edge_weight
            reg = edge_drop_out_prob.mean()

            # row, col = batch.edge_index
            # edge_batch = batch.batch[row]
            

            # uni, edge_batch_num = edge_batch.unique(return_counts=True)
            # sum_pe = scatter(edge_drop_out_prob, edge_batch, reduce="sum")

            # reg = []
            # for b_id in range(args.batch_size):
            #     if b_id in uni:
            #         num_edges = edge_batch_num[uni.tolist().index(b_id)]
            #         reg.append(sum_pe[b_id] / num_edges)
            #     else:
            #         # means no edges in that graph. So don't include.
            #         pass
            # num_graph_with_edges = len(reg)
            # reg = torch.stack(reg)
            # reg = reg.mean()


            # view_loss = model.calc_loss(x, x_aug)

            view_loss = model.calc_loss(x, x_aug) - (args.reg_lambda * reg)
            view_loss_all += view_loss.item()
            #reg_all += reg.item()
        
            # gradient ascent formulation
            (-view_loss).backward()
            view_optimizer.step()


            # train (model) to minimize contrastive loss
            model.train()
            view_learner.eval()
            model.zero_grad()


            x, _ = model(batch.batch, batch.x, batch.edge_index, batch.edge_attr, None)
            edge_logits = view_learner(batch.batch, batch.x, batch.edge_index, batch.edge_attr)


            temperature = 1.0
            bias = 0.0 + 0.0001  # If bias is 0, we run into problems
            eps = (bias - (1 - bias)) * torch.rand(edge_logits.size()) + (1 - bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = gate_inputs.to(device)
            gate_inputs = (gate_inputs + edge_logits) / temperature
            batch_aug_edge_weight = torch.sigmoid(gate_inputs).squeeze().detach()

            x_aug, out = model(batch.batch, batch.x, batch.edge_index, batch.edge_attr, batch_aug_edge_weight)
            
            task_loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])


            model_loss = 0.1* model.calc_loss(x, x_aug) + task_loss
            model_loss_all += model_loss.item()
            # standard gradient descent formulation
            model_loss.backward()
            model_optimizer.step()

        fin_model_loss = model_loss_all / len(dataloader)
        fin_view_loss = view_loss_all / len(dataloader)
        fin_reg = reg_all / len(dataloader)

        if epoch%10==0:
            logging.info('Epoch {}, Model Loss {}, View Loss {}, Reg {}'.format(epoch, fin_model_loss, fin_view_loss, fin_reg))
        
        model_losses.append(fin_model_loss)
        view_losses.append(fin_view_loss)
        view_regs.append(fin_reg)

        #model.eval()

        train_f1, val_f1, test_f1 = evaluate(model, dataloader, device)

        if test_f1>best_test_f1:
            best_test_f1 = test_f1



        # train_score, val_score, test_score = ee.embedding_evaluation(model.encoder, train_loader, valid_loader,
        #                                                              test_loader)

        # logging.info(
        #     "Metric: {} Train: {} Val: {} Test: {}".format(train_f1,val_f1,test_f1))

        print(f'Train {train_f1:0.4f} Val {val_f1:0.4f} Test {test_f1:0.4f}')

        # train_curve.append(train_score)
        # valid_curve.append(val_score)
        # test_curve.append(test_score)

    # if 'classification' in dataset.task_type:
    #     best_val_epoch = np.argmax(np.array(valid_curve))
    #     best_train = max(train_curve)
    # else:
    #     best_val_epoch = np.argmin(np.array(valid_curve))
    #     best_train = min(train_curve)

    # logging.info('FinishedTraining!')
    # logging.info('BestEpoch: {}'.format(best_val_epoch))
    # logging.info('BestTrainScore: {}'.format(best_train))
    # logging.info('BestValidationScore: {}'.format(valid_curve[best_val_epoch]))
    # logging.info('FinalTestScore: {}'.format(test_curve[best_val_epoch]))

    # return valid_curve[best_val_epoch], test_curve[best_val_epoch]

    return best_test_f1


def arg_parse():
    parser = argparse.ArgumentParser(description='AD-GCL ogbg-mol*')

    parser.add_argument('--dataset', type=str, default='ogbg-molesol',
                        help='Dataset')
    parser.add_argument('--model_lr', type=float, default=0.001,
                        help='Model Learning rate.')
    parser.add_argument('--view_lr', type=float, default=0.001,
                        help='View Learning rate.')
    parser.add_argument('--num_gc_layers', type=int, default=5,
                        help='Number of GNN layers before pooling')
    parser.add_argument('--pooling_type', type=str, default='standard',
                        help='GNN Pooling Type Standard/Layerwise')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimension')
    parser.add_argument('--mlp_edge_model_dim', type=int, default=64,
                        help='embedding dimension')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--drop_ratio', type=float, default=0.0,
                        help='Dropout Ratio / Probability')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Train Epochs')
    parser.add_argument('--reg_lambda', type=float, default=5.0, help='View Learner Edge Perturb Regularization Strength')

    parser.add_argument('--seed', type=int, default=0)

    return parser.parse_args()


if __name__ == '__main__':

    args = arg_parse()

    setup_seed(args.seed)
    
    best_test_f1 = []

    for i in range(10):

        print("="*50,"run", "="*50, i)

        test_f1 = run(args)
        best_test_f1.append(test_f1)
    
    print(f'Mean std f1 {np.mean(best_test_f1)*100:.2f} ± {np.std(best_test_f1)*100:.2f}')

