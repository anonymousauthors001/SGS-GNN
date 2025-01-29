import torch
# Gumbel-Softmax trick for edge sampling
# def gumbel_softmax_sampling(batch, edge_probs, edge_index, q=500, temperature=1.0, degree_bias_coef = 0.3, log=False, istest=False, epoch=-1):
        
#     if istest:
#         # samples = torch.exp(edge_probs)/torch.exp(edge_probs).sum()
#         samples = edge_probs/edge_probs.sum()
#         samples = (1-degree_bias_coef) * samples + degree_bias_coef *batch.prob
#         top_k_values, sampled_edges = torch.topk(samples, q, dim=-1, largest=True, sorted=True)
#         #sampled_edges = torch.multinomial(samples, q, replacement=False) #nothing
        
#         # samples = F.softmax(edge_probs/temperature, dim=-1)
#         #top_k_values, sampled_edges = torch.topk(edge_probs, q, dim=-1, largest=True, sorted=True)
#         # edge_gradients = edge_probs.grad
#         # sampled_edges = torch.topk(edge_gradients.abs(),k=q)
#     else:
#         # warmup_epochs = 20  # Number of epochs for the warm-up phase
#         # if epoch< warmup_epochs:
#             # samples = F.softmax(batch.prob, dim=-1)
#             # sampled_edges = torch.multinomial(samples, q, replacement=False) #nothing               
#         # else:
#         # samples = torch.exp(edge_probs)/torch.exp(edge_probs).sum()
#         # edge_probs += torch.ones_like(edge_probs)/len(edge_probs)
#         samples =  edge_probs/edge_probs.sum()
#         # samples = F.softmax(edge_probs/temperature, dim=-1)
#         # samples = 0.1*samples + 0.9*torch.rand_like(samples)
#         samples = (1-degree_bias_coef) * samples + degree_bias_coef *batch.prob
#         sampled_edges = torch.multinomial(samples, q, replacement=False) #nothing
# #         gumbels = -torch.empty_like(edge_probs).exponential_().log()
# #         gumbels = (gumbels - gumbels.min()) / (gumbels.max() - gumbels.min())
        
# #         #noise = 1.0 + (epoch/100)*9.0
# #         noise = 1.0
        
# #         gumbels = gumbels/noise
        
    
# # #         gumbels = 0

# #         logits = (edge_probs + gumbels) / temperature    
# #         samples = F.softmax(logits, dim=-1)

#         #top_k_values, sampled_edges = torch.topk(samples, q, dim=-1, largest=True, sorted=True) # 50       
        
    
#         #print(f'{torch.mean(samples).item():.4f}, {torch.var(samples).item():.4f}, {torch.std(samples).item():.4f}')    
#         #print("EP: ",torch.mean(edge_probs).item(), torch.var(edge_probs).item())
#         #print("SP: ",torch.mean(samples).item(), torch.var(samples).item())
    
#     # if epoch%10 == 0:
    
# #     print("Test" if istest == True else "Train")
# #     print(edge_probs)
# #     print(samples)   
#         #plot_probs(edge_probs, samples)
# #         plot_hist(edge_probs, samples, edge_probs[sampled_edges],samples[sampled_edges])
    
# #     adj = to_scipy_sparse_matrix(data.edge_index[:,sampled_edges], num_nodes=data.num_nodes)
# #     num_components, component = sp.csgraph.connected_components(adj, connection='weak')
# #     print("Components: ", num_components)
    
# #     sampled_edges = torch.multinomial(samples, q, replacement=False)
# #     top_k_values, sampled_edges = torch.topk(edge_probs, q, dim=-1, largest=True, sorted=True)    

    
#     one_hot = torch.zeros_like(samples)
#     one_hot.scatter_(0, sampled_edges, 1.0)     
    
#     indexs = (one_hot - samples).detach() + samples
    
#     edge_probs = edge_probs*indexs
#     indexs = indexs.bool()
    
    
# #     if log:        
# #         print("edge_probs: ",edge_probs)
# # #         print("gumbels: ", gumbels)
# # #         print("logits: ",logits)
# #         print("samples: ",samples)
# #         print("sampled_edges: ",sampled_edges)
# #         print("One hot: ", one_hot)
# #         print("Indexs: ",indexs)
# #         print("edge_probs[indexs]:", edge_probs[indexs])

    
# #     return indexs, edge_probs[indexs]
    
#     return indexs, edge_probs[indexs]+0.01
#     #return indexs, torch.clamp(edge_probs[indexs]+10, max=1.0)

def gumbel_softmax_sampling(batch, edge_probs, edge_index, q=500, temperature=1.0, degree_bias_coef = 0.3, log=False, istest=False, epoch=-1):
        
    if istest:
        # samples = torch.exp(edge_probs)/torch.exp(edge_probs).sum()
        samples = edge_probs/edge_probs.sum()
        # samples = F.softmax(edge_probs/temperature, dim=-1)
        #top_k_values, sampled_edges = torch.topk(edge_probs, q, dim=-1, largest=True, sorted=True)
        sampled_edges = torch.multinomial(samples, q, replacement=False) #nothing
        # edge_gradients = edge_probs.grad
        # sampled_edges = torch.topk(edge_gradients.abs(),k=q)
    else:
        # warmup_epochs = 20  # Number of epochs for the warm-up phase
        # if epoch< warmup_epochs:
            # samples = F.softmax(batch.prob, dim=-1)
            # sampled_edges = torch.multinomial(samples, q, replacement=False) #nothing               
        # else:
        # samples = torch.exp(edge_probs)/torch.exp(edge_probs).sum()
        # edge_probs += torch.ones_like(edge_probs)/len(edge_probs)
        samples =  edge_probs/edge_probs.sum()
        # samples = F.softmax(edge_probs/temperature, dim=-1)
        # samples = 0.1*samples + 0.9*torch.rand_like(samples)
        samples = (1-degree_bias_coef) * samples + degree_bias_coef *batch.prob
        sampled_edges = torch.multinomial(samples, q, replacement=False) #nothing
#         gumbels = -torch.empty_like(edge_probs).exponential_().log()
#         gumbels = (gumbels - gumbels.min()) / (gumbels.max() - gumbels.min())
        
#         #noise = 1.0 + (epoch/100)*9.0
#         noise = 1.0
        
#         gumbels = gumbels/noise
        
    
# #         gumbels = 0

#         logits = (edge_probs + gumbels) / temperature    
#         samples = F.softmax(logits, dim=-1)

        #top_k_values, sampled_edges = torch.topk(samples, q, dim=-1, largest=True, sorted=True) # 50       
        
    
        #print(f'{torch.mean(samples).item():.4f}, {torch.var(samples).item():.4f}, {torch.std(samples).item():.4f}')    
        #print("EP: ",torch.mean(edge_probs).item(), torch.var(edge_probs).item())
        #print("SP: ",torch.mean(samples).item(), torch.var(samples).item())
    
    # if epoch%10 == 0:
    
#     print("Test" if istest == True else "Train")
#     print(edge_probs)
#     print(samples)   
        # plot_probs(edge_probs, samples)
        # plot_hist(edge_probs, samples, edge_probs[sampled_edges],samples[sampled_edges])
    
#     adj = to_scipy_sparse_matrix(data.edge_index[:,sampled_edges], num_nodes=data.num_nodes)
#     num_components, component = sp.csgraph.connected_components(adj, connection='weak')
#     print("Components: ", num_components)
    
#     sampled_edges = torch.multinomial(samples, q, replacement=False)
#     top_k_values, sampled_edges = torch.topk(edge_probs, q, dim=-1, largest=True, sorted=True)    

    
    one_hot = torch.zeros_like(samples)
    one_hot.scatter_(0, sampled_edges, 1.0)     
    
    indexs = (one_hot - samples).detach() + samples
    
    edge_probs = edge_probs*indexs
    indexs = indexs.bool()
    
    
#     if log:        
#         print("edge_probs: ",edge_probs)
# #         print("gumbels: ", gumbels)
# #         print("logits: ",logits)
#         print("samples: ",samples)
#         print("sampled_edges: ",sampled_edges)
#         print("One hot: ", one_hot)
#         print("Indexs: ",indexs)
#         print("edge_probs[indexs]:", edge_probs[indexs])

    
#     return indexs, edge_probs[indexs]
    
    return indexs, edge_probs[indexs]+0.01
    #return indexs, torch.clamp(edge_probs[indexs]+10, max=1.0)


# Random edge sampling
def random_edge_sampling(edge_index, q):
    num_edges = edge_index.shape[1]
    sampled_indices = torch.randperm(num_edges)[:q]
    sampled_edge_index = edge_index[:, sampled_indices]
    return sampled_edge_index
