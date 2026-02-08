from training_two_pass import train as train_two_pass
from training_straight_through import train as train_straight_through
from training_hybrid import train as train_hybrid


def train(args, epoch, max_epoch, model, optimizer_gnn, optimizer_edge_prob, optimizer, criterion, cluster_loader,\
           q=500, alternate_frequency=1):
    pipeline = getattr(args, "pipeline", "two_pass")
    if pipeline == "straight_through":
        return train_straight_through(
            args,
            epoch,
            max_epoch,
            model,
            optimizer_gnn,
            optimizer_edge_prob,
            optimizer,
            criterion,
            cluster_loader,
            q=q,
            alternate_frequency=alternate_frequency,
        )
    if pipeline == "hybrid":
        return train_hybrid(
            args,
            epoch,
            max_epoch,
            model,
            optimizer_gnn,
            optimizer_edge_prob,
            optimizer,
            criterion,
            cluster_loader,
            q=q,
            alternate_frequency=alternate_frequency,
        )
    return train_two_pass(
        args,
        epoch,
        max_epoch,
        model,
        optimizer_gnn,
        optimizer_edge_prob,
        optimizer,
        criterion,
        cluster_loader,
        q=q,
        alternate_frequency=alternate_frequency,
    )
