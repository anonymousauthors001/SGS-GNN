import torch 
from utils import * 
from sampling import * 

# Evaluation function remains unchanged
def evaluate(args, model, cluster_loader, device, q=500, mode=None, temperature=1.0):
    model.eval()
    total_train_f1, total_val_f1, total_test_f1 = 0, 0, 0
    num_train_examples, num_val_examples, num_test_examples = 0, 0, 0

    with torch.no_grad():
        for batch in cluster_loader:
            batch = batch.to(device)
            if mode == 'learned':
                if batch.edge_index.shape[1] > q:
                    edge_probs = model.edge_prob_mlp(batch.x, batch.edge_index).squeeze()                    
                    sampled_edge_indices, sampled_edge_weight = gumbel_softmax_sampling(batch, edge_probs, batch.edge_index, degree_bias_coef = args.degree_bias_coef,\
                                                                                         q=q, istest=True, temperature = temperature)
                    sampled_edge_index = batch.edge_index[:, sampled_edge_indices]
                    out = model(batch, sampled_edge_index, sampled_edge_weight)                                        
                    #out = model(batch, sampled_edge_index, None)                                        
                else:
                    out = model(batch, batch.edge_index)
            elif mode == 'random':
                if batch.edge_index.shape[1] > q:
                    sampled_edge_index = random_edge_sampling(batch.edge_index, q=q)
                    out = model(batch, sampled_edge_index)
                else:
                    out = model(batch, batch.edge_index)
                    
            elif mode == 'edge':
                if batch.edge_index.shape[1] > q:
                    samples = F.softmax(batch.prob, dim=-1)
                    edge_sample = torch.multinomial(samples, q, replacement=False) #nothing                
                    sampled_edge_index = batch.edge_index[:,edge_sample]                
                    #print(sampled_edge_index.shape)
                    out = model(batch, sampled_edge_index)                
                else:
                    out = model(batch, batch.edge_index)
                    
            
            elif mode == 'full':
                out = model(batch, batch.edge_index)
            else:
                raise ValueError("Invalid mode. Choose 'learned', 'random', or 'full'.")

            # Calculate F1 scores for train, val, and test sets if masks are present
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

# Evaluation function remains unchanged
def ensemble_evaluate(args, model, cluster_loader, device, q=500, mode=None,  temperature=1.0):
    model.eval()
    total_train_f1, total_val_f1, total_test_f1 = 0, 0, 0
    num_train_examples, num_val_examples, num_test_examples = 0, 0, 0

    with torch.no_grad():
        for batch in cluster_loader:
            batch = batch.to(device)
            
            outs = []
            
            for i in range(args.num_samples_eval):                    
                if mode == 'learned':
                    if batch.edge_index.shape[1] > q:
                        edge_probs = model.edge_prob_mlp(batch.x, batch.edge_index).squeeze()                    
                        sampled_edge_indices, sampled_edge_weight = gumbel_softmax_sampling(batch, edge_probs, batch.edge_index, degree_bias_coef = args.degree_bias_coef, \
                                                                                            q=q, istest=True,temperature=temperature)
                        sampled_edge_index = batch.edge_index[:, sampled_edge_indices]
                        out = model(batch, sampled_edge_index, sampled_edge_weight)                                        
                        #out = model(batch, sampled_edge_index, None)                                        
                    else:
                        out = model(batch, batch.edge_index)
                elif mode == 'random':
                    if batch.edge_index.shape[1] > q:
                        sampled_edge_index = random_edge_sampling(batch.edge_index, q=q)
                        out = model(batch, sampled_edge_index)
                    else:
                        out = model(batch, batch.edge_index)

                elif mode == 'edge':
                    if batch.edge_index.shape[1] > q:
                        samples = F.softmax(batch.prob, dim=-1)
                        edge_sample = torch.multinomial(samples, q, replacement=False) #nothing                
                        sampled_edge_index = batch.edge_index[:,edge_sample]                
                        #print(sampled_edge_index.shape)
                        out = model(batch, sampled_edge_index)                
                    else:
                        out = model(batch, batch.edge_index)


                elif mode == 'full':
                    out = model(batch, batch.edge_index)
                else:
                    raise ValueError("Invalid mode. Choose 'learned', 'random', or 'full'.")
                    
                    
                outs.append(out)
            
            
#             #ensembel prediction
            
#             train_preds = []
#             val_preds = []
#             test_preds = []
            
#             for logits in outs:                   
#                 train_preds.append(logits[batch.train_mask].argmax(dim=1))
#                 val_preds.append(logits[batch.val_mask].argmax(dim=1))
#                 test_preds.append(logits[batch.test_mask].argmax(dim=1))
                
#             train_y, _ = torch.mode(torch.stack(train_preds), dim=0)
#             val_y, _ = torch.mode(torch.stack(val_preds), dim=0)
#             test_y, _ = torch.mode(torch.stack(test_preds), dim=0)
            
#             train_f1 = f1_score(batch.y[batch.train_mask].cpu(), train_y.cpu(), average='micro')
#             val_f1 = f1_score(batch.y[batch.val_mask].cpu(), val_y.cpu(), average='micro')
#             test_f1 = f1_score(batch.y[batch.test_mask].cpu(), test_y.cpu(), average='micro')
            
#             total_train_f1 += train_f1 * batch.train_mask.sum().item()
#             num_train_examples += batch.train_mask.sum().item()
            
#             total_val_f1 += val_f1 * batch.val_mask.sum().item()
#             num_val_examples += batch.val_mask.sum().item()
            
#             total_test_f1 += test_f1 * batch.test_mask.sum().item()
#             num_test_examples += batch.test_mask.sum().item()
            
            # average embedding prediction
            
            out = torch.mean(torch.stack(outs, dim=0), dim=0)                
                
            # Calculate F1 scores for train, val, and test sets if masks are present
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