from parser import parse_args
from utils import *
from model import * 
from torch_geometric.utils import to_scipy_sparse_matrix
import scipy.sparse as sp
import os 
from datasets import *  
from evaluate import evaluate,ensemble_evaluate 
from training import train
import time
from torch_geometric.loader import ClusterData, ClusterLoader
import pandas as pd 
from Notebooks.DeviceDir import get_directory

 
if __name__=='__main__':
    args,_ = parse_args()
    fix_seeds(args.seed)
    DIR, RESULTS_DIR = get_directory()

    #print(DIR)
    print(args.dataset)

    dataset, data = get_dataset(args, args.dataset)
    # print_stats(dataset,data)
    # Configuration
    plot = args.plot_curve
    RUNS = args.runs
    NUM_EPOCHS = args.epochs
    partition_threshold = args.metis_threshold
    

    log = args.log
    mode = args.mode  
    alternate_frequency = 0
    sample_percent = args.sample_perc
    figure_log = False
    GNNConv = args.GNN

    # Check if partitioning is needed
    use_metis = data.edge_index.shape[1]>=partition_threshold # or data.num_nodes >= 20000

    if use_metis:
        num_edges_per_partition = partition_threshold
        num_parts = int(np.ceil(data.edge_index.shape[1] / num_edges_per_partition))    

    #     num_parts = 1500
    #     num_edges_per_partition = int(data.edge_index.shape[1] / num_parts)

        sample_size = int(num_edges_per_partition * sample_percent)
        print("Using METIS with num_parts:", num_parts, "avg edges per partition:", num_edges_per_partition, "sample size:", sample_size)
    else:
        print("Graph has fewer than "+str(args.metis_threshold)+" nodes; don't do graph partition")
        sample_size = int(data.edge_index.shape[1] * sample_percent)

    # Partition data using METIS clustering or load the entire graph as a single batch
    if use_metis:
        start = time.time()
        cluster_dir = DIR+'tmp/' + args.dataset
        if not os.path.exists(cluster_dir):
            os.makedirs(cluster_dir)
        
        cluster_data = ClusterData(data, num_parts=num_parts, recursive=False, save_dir=cluster_dir, log=True)
        print("METIS Partition time:", time.time() - start)
        cluster_loader = ClusterLoader(cluster_data, batch_size=1, shuffle=True)
    else:
        cluster_loader = [data]

    # Training and evaluation loop
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model_save_path = DIR +'tmp/'+ args.dataset+'_'+args.mode+'_best_model.pth'

    train_figures = []
    iterations = []

    RunTimes = []

    best_test_f1s = []
    best_test_f1s_at_best_vals = []
    max_test_f1s_in_run = []

    edge_updated = np.zeros(RUNS)
    sgs_udpated = np.zeros(RUNS) 


    for run in range(RUNS):
        print("Run:", run)
        
        train_figures = []
        EpochTimes = []
        c_updates = 0
        t_updates = 0        

        # Initialize model and optimizers
        # optimizer_gnn = torch.optim.Adam([param for name, param in model.named_parameters()], lr=args.lr)
        #model = GNNModel(in_channels=data.x.shape[1], hidden_dim=args.nhid, num_classes=data.num_classes,dropout_prob=args.drop_rate).to(device)

        if GNNConv == "GCN":    
            model = GNNModel(in_channels=data.x.shape[1], hidden_dim=args.nhid, num_classes=data.num_classes,dropout_prob=args.drop_rate, edge_mlp_type=args.edge_mlp_type).to(device)
            optimizer_gnn = torch.optim.Adam([param for name, param in model.named_parameters() if 'gcn' in name], lr=args.lr)            
        elif GNNConv == "GIN":    
            model = GINModel(in_channels=data.x.shape[1], hidden_dim=args.nhid, num_classes=data.num_classes,dropout_prob=args.drop_rate, edge_mlp_type=args.edge_mlp_type).to(device)
            optimizer_gnn = torch.optim.Adam([param for name, param in model.named_parameters() if 'GIN' in name], lr=args.lr)
        elif GNNConv == "GAT":
            model = GATModel(in_channels=data.x.shape[1], hidden_dim=args.nhid,  num_classes=data.num_classes,dropout_prob=args.drop_rate, edge_mlp_type=args.edge_mlp_type).to(device)
            optimizer_gnn = torch.optim.Adam([param for name, param in model.named_parameters() if 'GAT' in name], lr=args.lr)            
        elif GNNConv == "Cheb":
            model = ChebModel(in_channels=data.x.shape[1], hidden_dim=args.nhid, num_classes=data.num_classes,dropout_prob=args.drop_rate, edge_mlp_type=args.edge_mlp_type).to(device)
            optimizer_gnn = torch.optim.Adam([param for name, param in model.named_parameters() if 'gcn' in name], lr=args.lr)
        else:
            raise NotImplemented
        
        if args.log:
            print(model)

        gpu_profiler = GpuMemoryProfiler(enabled=args.gpu_profile, device=device)
        if gpu_profiler.enabled:
            model.gpu_profiler = gpu_profiler
            model.edge_prob_mlp.gpu_profiler = gpu_profiler
        
        #optimizer_gnn = torch.optim.Adam([param for name, param in model.named_parameters()], lr=args.lr)
        optimizer_edge_prob = torch.optim.Adam([param for name, param in model.named_parameters() if 'edge_prob_mlp' in name], lr=args.lr)    
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
        
        criterion = nn.CrossEntropyLoss()

        # Initialize F1 score trackers
        train_f1_scores = np.zeros(NUM_EPOCHS)
        val_f1_scores = np.zeros(NUM_EPOCHS)
        test_f1_scores = np.zeros(NUM_EPOCHS)
        loss_scores = np.zeros(NUM_EPOCHS)
        
        best_test_f1 = 0
        best_val_f1 = 0
        test_at_best_val_f1 = 0
        best_Temperture = 0

        train_losses = []
        num_iteration = NUM_EPOCHS

        # Training over epochs
        peak_memory_bytes = 0
        if (args.stats or args.gpu_profile) and torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        for epoch in range(NUM_EPOCHS):
            # Alternate training of GNN and edge probability model based on frequency
            start = time.time()
            if (args.stats or args.gpu_profile) and torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            if gpu_profiler.enabled:
                gpu_profiler.start_epoch(epoch)
            loss, current_temp, c_update, t_update = train(
                args,
                epoch,
                NUM_EPOCHS,
                model,
                optimizer_gnn,
                optimizer_edge_prob,
                optimizer,
                criterion,
                cluster_loader,
                q=sample_size,
                alternate_frequency=alternate_frequency
            )
            
            EpochTimes.append(time.time()-start)
            if args.stats and torch.cuda.is_available() and device.type == "cuda":
                peak_memory_bytes = max(peak_memory_bytes, torch.cuda.max_memory_allocated(device))
            if gpu_profiler.enabled:
                epoch_peak_bytes = 0
                total_bytes = 0
                if torch.cuda.is_available() and device.type == "cuda":
                    epoch_peak_bytes = torch.cuda.max_memory_allocated(device)
                    total_bytes = torch.cuda.get_device_properties(device).total_memory
                stats = gpu_profiler.summarize_epoch(epoch)
                def _fmt_seg(name):
                    seg = stats.get(name)
                    if not seg:
                        return f"{name}=n/a"
                    peak_pct = (seg["max_peak_inc_bytes"] / epoch_peak_bytes * 100.0) if epoch_peak_bytes else 0.0
                    total_pct = (seg["max_peak_inc_bytes"] / total_bytes * 100.0) if total_bytes else 0.0
                    return (
                        f"{name}: peak_inc_mb={seg['max_peak_inc_mb']:.2f} "
                        f"({peak_pct:.2f}% of epoch_peak, {total_pct:.2f}% of device) "
                        f"alloc_after_mb={seg['max_alloc_after_mb']:.2f} "
                        f"calls={seg['calls']}"
                    )
                epoch_peak_mb = epoch_peak_bytes / (1024 ** 2) if epoch_peak_bytes else 0.0
                print(
                    "[gpu-profile] "
                    f"epoch={epoch} peak_epoch_mb={epoch_peak_mb:.2f} "
                    + " | ".join(
                        [
                            _fmt_seg("edge_mlp_pre"),
                            _fmt_seg("edge_score"),
                            _fmt_seg("gnn_forward"),
                            _fmt_seg("backward"),
                        ]
                    )
                )
                if torch.cuda.is_available() and device.type == "cuda":
                    alloc = torch.cuda.memory_allocated(device) / (1024 ** 2)
                    reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
                    print(f"[mem] allocated_mb={alloc:.1f} reserved_mb={reserved:.1f}")
                gpu_profiler.end_epoch()

            c_updates+=c_update
            t_updates+=t_update

            train_losses.append(loss)

            if args.eval:
            
                # Evaluate model performance on train, validation, and test sets
                # train_f1, val_f1, test_f1 = evaluate(args, model, cluster_loader, device, q=int(sample_size), mode=mode)
                train_f1, val_f1, test_f1 = ensemble_evaluate(args, model, cluster_loader, device, q=int(sample_size), mode=mode, temperature=current_temp)

                # Store F1 scores
                train_f1_scores[epoch] = train_f1
                val_f1_scores[epoch] = val_f1
                test_f1_scores[epoch] = test_f1
                loss_scores[epoch] = loss

                # Save model if validation F1 improves
                if val_f1 >= best_val_f1:
                    best_val_f1 = val_f1   
                    test_at_best_val_f1 = test_f1             
                    
                    torch.save(model.state_dict(), model_save_path)
                    
                    best_Temperture = current_temp
                    if log:
                        print(f"*Epoch {epoch}, model saved with Loss: {loss:.4f}, Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}, Test F1: {test_f1:.4f}")

        
                if test_f1 > best_test_f1:
                    best_test_f1 = test_f1
        
                if log and epoch % 100 == 0:
                    print(f'Epoch {epoch}, Loss: {loss:.4f}, Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}, Test F1: {test_f1:.4f}')
                    
                if figure_log:
                    train_figures.append(visualize(istest=False,show=False))
            
            if epoch>=5 and np.std(train_losses[-5:]) < args.convergence:
                num_iteration = epoch+1
                break
        
        if args.eval == False:
            torch.save(model.state_dict(), model_save_path)
            
        # Load the best model for evaluation
        iterations.append(num_iteration)
        RunTimes.append(np.mean(EpochTimes))
        edge_updated[run]=c_updates
        sgs_udpated[run]=t_updates
        print(f'Mean epoch time of run {np.mean(EpochTimes):.4f}')
        print('Iteration: ',num_iteration)
        print(f'EdgeMLP updated {c_updates}/{t_updates}')
        print(f'Best Test F1 throughout: {best_test_f1:.4f}')
        print(f'Loading best model for final evaluation at val: {best_val_f1:.4f}')
        model.load_state_dict(torch.load(model_save_path))

        max_test_f1s_in_run.append(best_test_f1) ## just for checking what's the best we could do

        #best_train_f1, best_val_f1, best_test_f1 = evaluate(args, model, cluster_loader, device, q=sample_size, mode=mode)
        best_train_f1, best_val_f1, best_test_f1 = ensemble_evaluate(args, model, cluster_loader, device, q=sample_size, mode=mode,temperature=best_Temperture)
        print(f'Best Test F1 after loading saved model: {best_test_f1:.4f}')

        if args.stats:
            train_time_sec = float(np.sum(EpochTimes))
            if torch.cuda.is_available() and device.type == "cuda":
                peak_mem_mb = peak_memory_bytes / (1024 ** 2)
                print(
                    f"[stats] pipeline={args.pipeline} run={run} "
                    f"train_time_sec={train_time_sec:.4f} peak_gpu_mem_mb={peak_mem_mb:.2f} "
                    f"best_val_f1={best_val_f1:.4f} best_test_f1={best_test_f1:.4f}"
                )
            else:
                print(
                    f"[stats] pipeline={args.pipeline} run={run} "
                    f"train_time_sec={train_time_sec:.4f} peak_gpu_mem_mb=NA "
                    f"best_val_f1={best_val_f1:.4f} best_test_f1={best_test_f1:.4f}"
                )
        
        best_test_f1s.append(best_test_f1)
        best_test_f1s_at_best_vals.append(test_at_best_val_f1)
        


        if plot:
            plot_learning_curves(run, train_f1_scores,val_f1_scores,test_f1_scores)
        if args.save_csv:
            output = {'run':run, 'iter':num_iteration, 'he':data.He, 'mode': mode, 'loss': loss, 'train_f1':best_train_f1, 'val_f1':best_val_f1, 'test_f1':best_test_f1}
            csv_name = "Results/"+args.dataset+'/'
            os.system('mkdir -p '+csv_name)
            csv_name += str(sample_percent)+'.csv'
            if os.path.exists(csv_name):
                result_df = pd.read_csv(csv_name)
            else:
                result_df = pd.DataFrame()
            # result_df = pd.DataFrame()
            result = pd.concat([result_df, pd.DataFrame(output,index = [0])])
            result.to_csv(csv_name, header=True, index=False)

    # Report the mean and standard deviation of the best test F1 scores over runs
    print(f'---------------Stats-----------')
    print(f'Mean training epoch runtime: {np.mean(RunTimes):.4f}')
    print(f'Mean convergence number: {np.mean(iterations):.4f} +/- {np.std(iterations):.4f}, {iterations}')
    if args.mode == "learned":
        print(f'EdgeMLP updated/Total GNN updates {np.round(np.mean(edge_updated))}/{np.round(np.mean(sgs_udpated))}')
    print(f'Mean Std of Best Test we could do F1 Score: {np.mean(max_test_f1s_in_run):.4f} +/- {np.std(max_test_f1s_in_run):.4f}')
    print(f'Mean Std of Test at best Val F1 Score: {np.mean(best_test_f1s_at_best_vals):.4f} +/- {np.std(best_test_f1s_at_best_vals):.4f}')
    print(f'Mean Std of Loaded best Val model Test F1 Score: {np.mean(best_test_f1s):.4f} +/- {np.std(best_test_f1s):.4f}')
    
    print(f'-------------------------------')

    os.system('rm '+model_save_path)
    
