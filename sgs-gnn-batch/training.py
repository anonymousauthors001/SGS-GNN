from sampling import *
from utils import consistency_loss, calculate_f1
import torch.nn.functional as F
from tqdm import tqdm

# Training function with alternating updates
def train(args, epoch, max_epoch, model, optimizer_gnn, optimizer_edge_prob, optimizer, criterion, cluster_loader,\
           q=500, alternate_frequency=1):
    device = args.device
    mode = args.mode
    model.train()
    total_loss = 0
    iteration = 0
    temperature = 1.0
    condtional_update = 0
    total_update = 0 

    for batch in cluster_loader:
        if not batch.train_mask.any():
            continue
        
        total_update+=1

        optimizer_edge_prob.zero_grad()
        optimizer_gnn.zero_grad()
            
        # Select edges based on mode
        if mode == 'learned':

            if batch.edge_index.shape[1] > q:

                #### from prior ######## reduce batch size                                
                batch = batch.to(device)
                
                if args.conditional or args.sparse_edge_mlp:
                    random_samples = F.softmax(batch.prob, dim=-1)
                    random_edge_sample = torch.multinomial(random_samples, q, replacement=False) #nothing                
                    random_sampled_edge_index = batch.edge_index[:,random_edge_sample]                

                ## for more efficiency
                if args.sparse_edge_mlp:                    
                    edge_probs = model.edge_prob_mlp(batch.x, batch.edge_index, random_sampled_edge_index).squeeze()                   
                else:
                    edge_probs = model.edge_prob_mlp(batch.x, batch.edge_index, random_sampled_edge_index=None).squeeze()                   

                t_init = args.t_init 
                t_min = args.t_min
                r = (t_init - t_min)/max_epoch # r = 0.01
                temperature = max(t_min, t_init - epoch*r) # Annealing (improves very little)
                #print(temperature)                
                sampled_edge_indices, sampled_edge_weight = gumbel_softmax_sampling(batch, edge_probs, batch.edge_index, q=q, temperature=temperature, \
                                                                                    degree_bias_coef= args.degree_bias_coef, log=(epoch==max_epoch-1),epoch=epoch)
                sampled_edge_index = batch.edge_index[:, sampled_edge_indices]      
                learned_out = model(batch, sampled_edge_index, sampled_edge_weight) 
                #out = model(batch, sampled_edge_index, None)                                                       
        
                # random_edge_index = random_edge_sampling(batch.edge_index,q)
                # random_out = model(batch,random_edge_index)
                
                ###- firx prior---
                #with torch.no_grad():
                # random_samples = F.softmax(batch.prob, dim=-1)
                # random_edge_sample = torch.multinomial(random_samples, q, replacement=False) #nothing                
                # random_sampled_edge_index = batch.edge_index[:,random_edge_sample]                
                
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
                        # Compute node classification loss
                        # loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])
                        # loss = loss + 0.001 * torch.sum(torch.square(edge_probs))

                        #######---regularizer 1
                        edge_labels = torch.full((batch.edge_index.size(1),), -1, dtype=torch.long, device=device)
                        train_indices = torch.nonzero(batch.train_mask).squeeze()
                        train_edge_mask = torch.isin(batch.edge_index[0], train_indices) & torch.isin(batch.edge_index[1], train_indices)

                        # Assign labels based on the class of the endpoints
                        same_class_mask = batch.y[batch.edge_index[0]] == batch.y[batch.edge_index[1]]
                        edge_labels[train_edge_mask & same_class_mask] = 1
                        edge_labels[train_edge_mask & ~same_class_mask] = 0

                        # Filter out the edges with labels -1
                        valid_edge_mask = edge_labels != -1
                        valid_edge_probs = edge_probs[valid_edge_mask]
                        valid_edge_labels = edge_labels[valid_edge_mask]

                        #print(torch.sum(valid_edge_labels))
                        if torch.sum(valid_edge_labels).item() > 1:            
                            loss2 = F.binary_cross_entropy(valid_edge_probs, valid_edge_labels.float().to(device))
                        else:
                            loss2 = 0
                        loss = loss+ args.regularizer1_coef*loss2
                    
                    if args.reg2 == True:                        
                        global_consistency_loss = consistency_loss(edge_probs,sampled_edge_index, learned_out)
                        loss = loss + args.consist_reg_coef *global_consistency_loss


                    ########---regularizer 3 ---- not used in paper #######
                    # if args.reg3 == True:
                    #     train_indices = torch.nonzero(batch.train_mask).squeeze()            
                    #     num_nodes = train_indices.size(0)
                    #     #num_edges = min(num_nodes*num_nodes,q)  # Adjust the number of edges as needed
                    #     num_edges = num_nodes*num_nodes

                    #     # Generate random edges
                    #     src = torch.randint(0, num_nodes, (num_edges,))
                    #     dst = torch.randint(0, num_nodes, (num_edges,))

                    #     # Map to actual node indices
                    #     src_nodes = train_indices[src]
                    #     dst_nodes = train_indices[dst]

                    #     # Create edge_index tensor
                    #     reg_edge_index = torch.stack([src_nodes, dst_nodes], dim=0).to(device)
                    #     reg_edge_label = (batch.y[src_nodes] == batch.y[dst_nodes]).float().to(device)            
                    #     reg_edge_probs = model.edge_prob_mlp(batch.x, reg_edge_index).squeeze()
                    #     loss4 = F.binary_cross_entropy(reg_edge_probs, reg_edge_label)
                    #     #print(reg_edge_index)
                    #     #print(reg_edge_label)

                    #     loss = loss+loss4

                    
                    loss.backward()
                    optimizer_edge_prob.step()
                    optimizer_gnn.step()                              
                    #print("learned..")
                else:
                    loss = criterion(random_out[batch.train_mask], batch.y[batch.train_mask])
                    loss.backward()
                    optimizer_gnn.step()                    
                    #print("Random..")
            else:
                batch = batch.to(device)
                out = model(batch, batch.edge_index)
                loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])
                loss.backward()
                optimizer_gnn.step()
                #print("whole batch")
    
        # loss.backward()
        # importance_score = edge_probs.grad.abs().detach().cpu().numpy()

#         # Update parameters selectively
#         if update_gnn:
#             optimizer_gnn.step()
#         if update_edge_prob:
#             optimizer_edge_prob.step()

        # optimizer.step()
    
    
        elif mode == 'random':
            batch = batch.to(device)
            if batch.edge_index.shape[1] > q:
                sampled_edge_index = random_edge_sampling(batch.edge_index, q=q)
                
#                 adj = to_scipy_sparse_matrix(sampled_edge_index, num_nodes=data.num_nodes)
#                 num_components, component = sp.csgraph.connected_components(adj, connection='weak')
#                 print("Components in random: ", num_components)
                
                out = model(batch, sampled_edge_index)
            else:
                out = model(batch, batch.edge_index)
            
            loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])
            loss.backward()
            optimizer.step()
                
        elif mode == 'edge':
            batch = batch.to(device)
            if batch.edge_index.shape[1] > q:
                
                samples = F.softmax(batch.prob, dim=-1)
                # samples = batch.prob/batch.prob.sum()
                edge_sample = torch.multinomial(samples, q, replacement=False) #nothing                
                sampled_edge_index = batch.edge_index[:,edge_sample]                
                #print(sampled_edge_index.shape)
                
#                 adj = to_scipy_sparse_matrix(sampled_edge_index, num_nodes=data.num_nodes)
#                 num_components, component = sp.csgraph.connected_components(adj, connection='weak')
#                 print("Components in edge: ", num_components)
                
                out = model(batch, sampled_edge_index)
            else:
                out = model(batch, batch.edge_index)
            
            loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])
            loss.backward()
            optimizer.step()
            
        elif mode == 'full':
            batch = batch.to(device)
            out = model(batch, batch.edge_index)
            loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])
            loss.backward()
            optimizer.step()
            
        else:
            raise ValueError("Invalid mode. Choose 'learned', 'random', or 'full'.")
        

        total_loss += loss.item()
        iteration += 1

    return total_loss / len(cluster_loader), temperature, condtional_update, total_update

