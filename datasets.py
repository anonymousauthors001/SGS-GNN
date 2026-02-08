import torch_geometric
from torch_geometric.datasets import KarateClub, Reddit, Reddit2, Flickr, Yelp, AmazonProducts, PPI,  OGB_MAG,  FakeDataset, Amazon,Coauthor,HeterophilousGraphDataset,LINKXDataset,CitationFull
from torch_geometric.datasets import WikipediaNetwork,Planetoid, Reddit,WikiCS
import torch 
from torch_geometric.typing import Adj, SparseTensor
from torch_geometric.utils import scatter
from ogb.nodeproppred import PygNodePropPredDataset
from sklearn.model_selection import train_test_split
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import to_scipy_sparse_matrix, homophily
from Notebooks.DeviceDir import get_device, get_directory
from torch_geometric.utils import to_undirected
from ipynb.fs.full.EffectiveResistanceWeights import EffectiveRessistance
import os
import scipy.sparse as sp
import torch.nn.functional as F

from sklearn.decomposition import TruncatedSVD

def adj_feature(data):    
    adj_mat = torch.zeros((data.num_nodes,data.num_nodes))
    edges = data.edge_index.t()
    adj_mat[edges[:,0], edges[:,1]] = 1
    adj_mat[edges[:,1], edges[:,0]] = 1
    
    #return adj_mat
    
#     n_components = data.x.shape[1]
    n_components = min(256, data.x.shape[1], data.num_nodes)

    svd = TruncatedSVD(n_components=n_components)
    x = svd.fit_transform(adj_mat)
    
    x = torch.Tensor(x)
    
    return x

# data.x = torch.cat((data.x, adj_feature(data)), dim=1)
# data.x = adj_feature(data)
# print(data.x.shape)
# data.x

def load_dataset(dataset_name, DIR = '', verbose = True):
    DATASET_NAME = dataset_name

    if dataset_name == "karate":
        dataset = KarateClub()
    elif dataset_name == "moon":
        from ipynb.fs.full.Moon import MoonDataset
        dataset = MoonDataset(n_samples=1000, degree=4, train=0.2, h = 0.2)
    elif dataset_name == "SmallCora":
        dataset = Planetoid(root=DIR+'/tmp/Cora', name='Cora')
    elif dataset_name in ["Cora", "Cora_ML", "CiteSeer", "DBLP", "PubMed"]:
        dataset = CitationFull(root=DIR+'/tmp/Citation/'+dataset_name, name=dataset_name)
    elif dataset_name == 'Amazon-ratings':
        dataset = HeterophilousGraphDataset(root=DIR+'/tmp/amazon_ratings', name = dataset_name)
#     elif dataset_name == 'Roman-empire':
#         from torch_geometric.datasets import LINKXDataset
#         dataset = LINKXDataset(DIR+'/tmp/Roman_empire', dataset_name)
    elif dataset_name == "Reddit":
        dataset = Reddit(root=DIR+'/tmp/Reddit')
    elif dataset_name == 'penn94':
        dataset = LINKXDataset(root=DIR+'/tmp/LINKX', name=dataset_name)
    elif dataset_name == 'wiki':
        dataset = WikiCS(root=DIR+'/tmp/WikiCS')
    elif dataset_name == 'Photo':
        dataset = Amazon(root=DIR+'/tmp/Photo', name='Photo')
    elif dataset_name == 'CiteSeer':
        dataset = Planetoid(root=DIR+'/tmp/CiteSeer', name=DATASET_NAME)
    elif dataset_name == 'CS':
        dataset = Coauthor(root=DIR+'/tmp/CS', name = DATASET_NAME)
    elif dataset_name == 'Physics':
        dataset = Coauthor(root=DIR+'/tmp/Physics', name = DATASET_NAME)
    elif dataset_name == 'Minesweeper':
        dataset   = HeterophilousGraphDataset(root=DIR+'/tmp/Mine',name=dataset_name)
#     elif dataset_name == 'pokec':
#         print("HELLOWROLD ")
    elif dataset_name == 'ogbn-proteins':
        dataset = PygNodePropPredDataset(name='ogbn-proteins', root=DIR+'/tmp/ogbn-proteins')
        data = dataset[0]
        data.node_species = None
        data.y = data.y.to(torch.float)

        # Initialize features of nodes by aggregating edge features.
        row, col = data.edge_index
        data.x = scatter(data.edge_attr, col, dim_size=data.num_nodes, reduce='sum')
        labels = data.y.argmax(dim=1)
        data.y = labels
        
        return dataset, data
    elif dataset_name == 'squirrel':
        dataset = WikipediaNetwork(root=DIR+'/tmp/squirrel', name='Squirrel')
    elif dataset_name == 'AmazonProducts':
        dataset = AmazonProducts(root=DIR+'/tmp/AmazonProducts')
    else:        
        from ipynb.fs.full.Dataset import get_data
        data, dataset = get_data(dataset_name, DIR=DIR, log=False, h_score = False, split_no = 0)


        
        return dataset, data
    
    if verbose: print(dataset)
    data = dataset[0]
    if verbose: print(data)
        
    return dataset, data

def train_val_test_mask(data, train=0.4, val=0.3, test=0.3, random_state=False):

    if isinstance(data.x, SparseTensor):
        N = data.x.size(0)
        data.num_nodes = N
    else:
        N = data.x.shape[0]

    indexs = list(range(N))

    if random_state:
        train_index, test_index = train_test_split(indexs, test_size=val+test)
        val_index, test_index = train_test_split(test_index, test_size=test/(val+test))
    else:
        train_index, test_index = train_test_split(indexs, test_size=val+test, random_state=1)
        val_index, test_index = train_test_split(test_index, test_size=test/(val+test), random_state=1)



    train_mask = torch.zeros(N, dtype=bool)
    train_mask[train_index]=True
    val_mask = torch.zeros(N, dtype=bool)
    val_mask[val_index]=True
    test_mask = torch.zeros(N, dtype=bool)
    test_mask[test_index]=True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data

def add_degree(data):
    N = data.num_nodes
    E = data.num_edges

    adj = SparseTensor(row=data.edge_index[0], col=data.edge_index[1],
                value=torch.arange(E, device=data.edge_index.device),sparse_sizes=(N, N))

    row, col, _ = adj.coo()

    deg_in = 1. / adj.storage.colcount()
    deg_out = 1. / adj.storage.rowcount()
    prob = (1. / deg_in[row]) + (1. / deg_out[col])
    prob = 1. / (prob + 1e-10)
    #prob = F.softmax(prob/100, dim=0)
    prob = F.softmax(prob*len(prob)**-0.5,dim=0) #normalize (low variance)
    data.prob = prob


def add_ER(data, DIR, dataset_name, recompute = False):
    E_filename = DIR+dataset_name+'_erweight.pt'
    
    weight = None
    
    if os.path.exists(E_filename)==True and recompute == False:
        weight = torch.load(E_filename)
    else:
        er = EffectiveRessistance(data, eps=0.1, lmbda=0.1)
        weight  = er.er_weight()
        torch.save(weight, E_filename)

    weight = F.softmax(weight*len(weight)**-0.5,dim=0) #normalize (low variance)
    
    data.prob = weight


def get_dataset(args, dataset_name="SmallCora",add_components=False):
    # Example usage
    # dataset_name = "SmallCora"
    # DATASET_NAME = dataset_name
    DIR, RESULTS_DIR = get_directory()
    dataset, data = load_dataset(dataset_name, DIR)

    if args.syn == True:
        from ipynb.fs.full.Dataset import generate_synthetic
        print("Synethtic")
        data = generate_synthetic(data, d=args.degree, h=args.hn, train=args.train, random_state=0, log=True, balance = False)
        print(data)

    if data.is_undirected() == False:
        data.edge_index = to_undirected(data.edge_index, reduce = "mean")

    
    if dataset_name in ['Squirrel','Chameleon','Amazon-ratings','reed98']:

        x = adj_feature(data)
        data.x = torch.cat((data.x, x), dim=1)

        
    try:
        if "val_mask" not in data.__dict__['_store']:    
            data = train_val_test_mask(data, train=0.2, val=0.4, test=0.4)
    except:    
        try:
            if "val_mask" not in data.__dict__['data']:
                data = train_val_test_mask(data, train=0.2, val=0.4, test=0.4)            
        except:
            None
    if dataset_name in ['wiki']:
        data = train_val_test_mask(data, train=0.2, val=0.4, test=0.4)
    else:
        if len(data.train_mask.shape) > 1 and len(data.val_mask.shape) > 1 and len(data.test_mask.shape) > 1:    
            try:
                split_index = 2
                data.train_mask = data.train_mask[:,split_index]
                data.val_mask = data.val_mask[:,split_index]
                data.test_mask = data.test_mask[:,split_index]

            except:        
                data = train_val_test_mask(data, train=0.2, val=0.4, test=0.4)
    # print(data)
    data.num_classes = int(max(data.y)+1)
    He = homophily(data.edge_index, data.y, method='edge')
    data.He = He
    if args.ER:
        add_ER(data, DIR, dataset_name, args.ERcompute)
    else:
        add_degree(data)
    adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)
    if add_components:
        num_components, component = sp.csgraph.connected_components(adj, connection='weak')
        data.num_components = num_components
    return dataset, data 

def print_stats(dataset, data):
    print("""Stats.....""")
    print(f'Dataset: {dataset}:')
    print('======================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {data.num_classes}')
    print()
    print(data)
    print('===========================================================================================================')

    # Gather some statistics about the graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Number of training nodes: {data.train_mask.sum()}')
    print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')