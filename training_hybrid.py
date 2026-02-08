from sampling import *
from utils import consistency_loss, calculate_f1
import torch.nn.functional as F
from tqdm import tqdm

# Training function with alternating updates (hybrid pipeline)
def train(args, epoch, max_epoch, model, optimizer_gnn, optimizer_edge_prob, optimizer, criterion, cluster_loader,\
           q=500, alternate_frequency=1):
    device = args.device
    mode = args.mode
    use_checkpoint = bool(getattr(args, "hybrid_checkpoint", False))
    if epoch == 0:
        print(f"[hybrid] checkpoint={'on' if use_checkpoint else 'off'}")
    model.train()
    profiler = getattr(model, "gpu_profiler", None)
    total_loss = 0
    iteration = 0
    temperature = 1.0
    condtional_update = 0
    total_update = 0 

    def _backward(loss):
        if profiler is not None:
            profiler.begin("backward")
        loss.backward()
        if profiler is not None:
            profiler.end("backward")

    for batch in cluster_loader:
        if not batch.train_mask.any():
            continue
        
        total_update+=1

        optimizer_edge_prob.zero_grad()
        optimizer_gnn.zero_grad()
            
        # Select edges based on mode
        if mode == 'learned':

            if batch.edge_index.shape[1] > q:                                          
                batch = batch.to(device)

                random_sampled_edge_index = None
                if args.conditional or args.sparse_edge_mlp:
                    random_samples = F.softmax(batch.prob, dim=-1)
                    random_edge_sample = torch.multinomial(random_samples, q, replacement=False)
                    random_sampled_edge_index = batch.edge_index[:, random_edge_sample]

                # pass 1: edge probability for sampling (grad enabled)
                if random_sampled_edge_index is not None:
                    edge_probs_full = model.edge_prob_mlp(
                        batch.x,
                        batch.edge_index,
                        random_sampled_edge_index,
                        use_checkpoint=use_checkpoint,
                    ).squeeze()
                else:
                    edge_probs_full = model.edge_prob_mlp(
                        batch.x,
                        batch.edge_index,
                        None,
                        use_checkpoint=use_checkpoint,
                    ).squeeze()

                # sample : sampling sparse subgraph using edge probs
                t_init = args.t_init 
                t_min = args.t_min
                r = (t_init - t_min)/max_epoch # r = 0.01
                temperature = max(t_min, t_init - epoch*r) # Annealing (improves very little)
                
                sampled_edge_indices, _ = gumbel_softmax_sampling(
                    batch, 
                    edge_probs_full.detach(),
                    batch.edge_index, 
                    q=q, 
                    temperature=temperature,
                    degree_bias_coef= args.degree_bias_coef,
                    log=(epoch==max_epoch-1),
                    epoch=epoch
                )
                
                sampled_edge_index = batch.edge_index[:, sampled_edge_indices]

                # pass: use sampled indices with full edge probs (grad enabled)
                edge_probs_sampled = edge_probs_full[sampled_edge_indices]
                edge_probs_for_loss = edge_probs_sampled
                learned_out = model(batch, sampled_edge_index, edge_probs_sampled)
                
                update_edge_mlp = True
                # Calculate F1 scores for learned and random sampling
                if args.conditional:
                    random_out = model(batch,random_sampled_edge_index)                    
                    learned_f1 = calculate_f1(learned_out, batch.y, batch.train_mask)
                    random_f1 = calculate_f1(random_out, batch.y, batch.train_mask)
                    # Compute loss for learned edges            
                
                    if learned_f1 > random_f1:
                        update_edge_mlp = True                        
                    else:
                        update_edge_mlp = False

                if update_edge_mlp:
                    condtional_update+= 1
                    loss = criterion(learned_out[batch.train_mask], batch.y[batch.train_mask])

                    if args.reg1 == True:
                        #######---regularizer 1 (sampled edges only)
                        edge_labels = torch.full((sampled_edge_index.size(1),), -1, dtype=torch.long, device=device)
                        train_indices = torch.nonzero(batch.train_mask).squeeze()
                        src = sampled_edge_index[0]
                        dst = sampled_edge_index[1]
                        train_edge_mask = torch.isin(src, train_indices) & torch.isin(dst, train_indices)

                        # Assign labels based on the class of the endpoints
                        same_class_mask = batch.y[src] == batch.y[dst]
                        edge_labels[train_edge_mask & same_class_mask] = 1
                        edge_labels[train_edge_mask & ~same_class_mask] = 0

                        # Filter out the edges with labels -1
                        valid_edge_mask = edge_labels != -1
                        valid_edge_probs = edge_probs_for_loss[valid_edge_mask]
                        valid_edge_labels = edge_labels[valid_edge_mask]

                        if torch.sum(valid_edge_labels).item() > 1:            
                            loss2 = F.binary_cross_entropy(valid_edge_probs, valid_edge_labels.float().to(device))
                        else:
                            loss2 = 0
                        loss = loss+ args.regularizer1_coef*loss2
                    
                    if args.reg2 == True:                        
                        global_consistency_loss = consistency_loss(edge_probs_for_loss, sampled_edge_index, learned_out)
                        loss = loss + args.consist_reg_coef *global_consistency_loss

                    _backward(loss)
                    optimizer_edge_prob.step()
                    optimizer_gnn.step()                              
                else:
                    loss = criterion(random_out[batch.train_mask], batch.y[batch.train_mask])
                    _backward(loss)
                    optimizer_gnn.step()                    
            else:
                batch = batch.to(device)
                out = model(batch, batch.edge_index)
                loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])
                _backward(loss)
                optimizer_gnn.step()
    
        elif mode == 'random':
            batch = batch.to(device)
            if batch.edge_index.shape[1] > q:
                sampled_edge_index = random_edge_sampling(batch.edge_index, q=q)
                out = model(batch, sampled_edge_index)
            else:
                out = model(batch, batch.edge_index)
            
            loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])
            _backward(loss)
            optimizer.step()
                
        elif mode == 'edge':
            batch = batch.to(device)
            if batch.edge_index.shape[1] > q:
                samples = F.softmax(batch.prob, dim=-1)
                edge_sample = torch.multinomial(samples, q, replacement=False) #nothing                
                sampled_edge_index = batch.edge_index[:,edge_sample]
                out = model(batch, sampled_edge_index)
            else:
                out = model(batch, batch.edge_index)
            
            loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])
            _backward(loss)
            optimizer.step()
            
        elif mode == 'full':
            batch = batch.to(device)
            out = model(batch, batch.edge_index)
            loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])
            _backward(loss)
            optimizer.step()
            
        else:
            raise ValueError("Invalid mode. Choose 'learned', 'random', or 'full'.")
        

        total_loss += loss.item()
        iteration += 1

    return total_loss / len(cluster_loader), temperature, condtional_update, total_update
