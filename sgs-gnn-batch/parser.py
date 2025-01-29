import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


datasets = ['Cornell',
            'Texas',
            'Wisconsin',
            'reed98',
            'amherst41',
            'penn94',
            'Roman-empire',
            'cornell5',
            'Squirrel',
            'johnshopkins55',
            'Actor',
            'Minesweeper',
            'Questions',
            'Chameleon',
            'Tolokers',
            'Amazon-ratings',
            'genius',
            'pokec',
            'arxiv-year',
            'snap-patents',
            'Cora',
            'DBLP',
            'Computers',
            'PubMed',
            'Cora_ML',
            'SmallCora',
            'CS',
            'Photo',
            'Physics',
            'CiteSeer',
            'wiki',
            'Reddit',
            'ogbn-proteins',
            'Reddit0.1',
            'Reddit0.2',
            'Reddit0.3',
            'Reddit0.4',
            'Reddit0.5',
            'Reddit0.6',
            'Reddit0.7',
            'Moon',
            'Karate']

GNNs = ['GCN','GIN','GAT','Cheb']
EDGE_MLPs= ['MLP','GSAGE','GCN']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--GNN', type=str, default='GCN', choices=GNNs, help=GNNs)
    parser.add_argument('--edge_mlp_type', type=str, default='GCN', choices=EDGE_MLPs, help=EDGE_MLPs)    
    #parser.add_argument('--kr', type=float, default='0.5',help='proportion of average degree for EDGEmlpSAGE')    
    parser.add_argument('--sparse_edge_mlp', type=str2bool, nargs='?', const=False, default=False, help='Feed Sparse Graph at EDGE MLP for very large graphs')    
    parser.add_argument('--conditional', type=str2bool, nargs='?', const=True, default=True, help='Conditional update')    
    parser.add_argument('--eval', type=str2bool, nargs='?', const=True, default=True, help='evaluate at every epoch')    
    parser.add_argument('--runs', type=int, default=1, help='# of independent model runs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--dataset', type=str, default='SmallCora', choices=datasets, help='dataset')
    parser.add_argument('--mode', type=str, default='learned',choices = ['learned','edge','random','full'])
    parser.add_argument('--lr', type=float, default=0.001,  help='learning rate')
    parser.add_argument('--drop_rate', type=float, default=0.3,  help='dropout rate')
    parser.add_argument('--weight_decay', type=float, default=0.0005,  help='weight decay rate')
    parser.add_argument('--epochs', type=int, default=200,  help='epochs')
    parser.add_argument('--sample_perc',type=float,default = 0.20, help='% of nodes to sample')
    parser.add_argument('--metis_threshold',type=int,default = 500000, help='maximum number of nodes in each partition')
    parser.add_argument('--t_init', type=float, default=0.7,  help='initial temperature in gumbel-softmax temperature annealing')
    parser.add_argument('--t_min', type=float, default=0.5,  help='min temperature in gumbel-softmax temperature annealing')
    parser.add_argument('--regularizer1_coef', type=float, default=1.0,  help='coefficient of homophily regularizer')
    parser.add_argument('--reg1', type=str2bool, nargs='?', const=True, default=True, help='use assortative loss')
    parser.add_argument('--reg2', type=str2bool, nargs='?', const=True, default=True, help='use consistency loss')
    parser.add_argument('--consist_reg_coef', type=float, default=0.5,  help='coefficient of consistency regularizer')
    parser.add_argument('--degree_bias_coef', type=float, default=0.3,  help='coefficient of degree bias to combine with learned sampling probability')
    parser.add_argument('--nhid', type=int, default=256,  help='nhid')
    parser.add_argument('--num_samples_eval',type = int, default = 11, help = 'number of samples to take during ensemble eval')
    parser.add_argument('--device', type=str,default = 'cuda:0',help='cuda:0/cuda:1/...')
    parser.add_argument('--save_csv', type=str2bool, nargs='?', const=True, default=True, help='save csv')    
    parser.add_argument('--plot_curve', type=str2bool, nargs='?', const=False, default=False, help='plot curve')
    parser.add_argument('--log', type=str2bool, nargs='?', const=False, default=False, help='show log')
    parser.add_argument('--convergence',type=float,default = 0.0001) ##Recommended 0.001
    parser.add_argument('--ER', type=str2bool, nargs='?', const=False, default=False, help='use effective resistance')
    parser.add_argument('--ERcompute', type=str2bool, nargs='?', const=False, default=False, help='recompute effective resistance')    
    parser.add_argument('--syn', type=str2bool, nargs='?', const=False, default=False, help='generate synthetic graph')    
    parser.add_argument('--degree',type=int,default = 100)
    parser.add_argument('--train',type=float,default = 0.2)
    parser.add_argument('--hn',type=float,default = 0.1) #nodehomophily

    return parser.parse_known_args()