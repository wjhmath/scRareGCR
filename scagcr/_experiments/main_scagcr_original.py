import argparse
import warnings
import torch
import numpy as np
from sklearn.cluster import KMeans

from utils import setup_seed, loader_construction, evaluate, device
from model import Model, ZINBLoss
from config import config


def collect_embeddings(model, data_loader, device):
    embeddings = []
    labels = []
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.float().to(device)
            batch_z, _, _, _, _, _, _, _ = model(batch_x)
            embeddings.append(batch_z.cpu().numpy())
            labels.append(batch_y)
    return np.vstack(embeddings), np.hstack(labels)


def initialize_cluster_centers(model, train_loader, n_clusters, seed, device):
    z_train, _ = collect_embeddings(model, train_loader, device)
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20).fit(z_train)
    model.cluster_centers.data.copy_(torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=device))


def train(train_loader, test_loader, input_dim, graph_head, phi, gcn_dim, mlp_dim,
          prob_feature, prob_edge, tau, alpha, beta, lambda_cl, lambda_cluster,
          cluster_alpha, lambda_zinb, dropout, lr, seed, epochs, pretrain_epochs,
          save_model_path, n_clusters, device, use_graph=True):
    model = Model(input_dim, graph_head, phi, gcn_dim, mlp_dim, prob_feature, prob_edge,
                  tau, alpha, beta, dropout, n_clusters, cluster_alpha, use_graph).to(device)
    zinb_loss_fn = ZINBLoss().to(device)
    opt_model = torch.optim.Adam(model.parameters(), lr=lr)
    setup_seed(seed)
    best_epoch = 0
    best_ari = -1.0
    z_test_epoch, y_test_epoch = [], []
    mae_f = torch.nn.L1Loss(reduction='mean')

    for each_epoch in range(epochs):
        if each_epoch == pretrain_epochs:
            initialize_cluster_centers(model, train_loader, n_clusters, seed, device)

        model.train()
        for batch_x, _ in train_loader:
            batch_x = batch_x.float().to(device)
            outputs = model(batch_x)
            batch_z, x_imp, loss_cl, loss_cluster, mean, disp, pi, _ = outputs
            mask = torch.where(batch_x != 0, torch.ones_like(batch_x), torch.zeros_like(batch_x))
            loss_mae = mae_f(mask * x_imp, mask * batch_x)
            counts = torch.expm1(torch.clamp(batch_x, min=0.0, max=20.0))
            size_factor = torch.clamp(counts.sum(1), min=1.0)
            loss_zinb = zinb_loss_fn(counts, mean, disp, pi, size_factor)
            cluster_weight = 0.0 if each_epoch < pretrain_epochs else lambda_cluster
            loss = loss_mae + lambda_cl * loss_cl + cluster_weight * loss_cluster + lambda_zinb * loss_zinb
            opt_model.zero_grad()
            loss.backward()
            opt_model.step()

        with torch.no_grad():
            model.eval()
            batch_losses, z_test, y_test = [], [], []
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.float().to(device)
                outputs = model(batch_x)
                batch_z, x_imp, loss_cl, loss_cluster, mean, disp, pi, _ = outputs
                mask = torch.where(batch_x != 0, torch.ones_like(batch_x), torch.zeros_like(batch_x))
                loss_mae = mae_f(mask * x_imp, mask * batch_x)
                counts = torch.expm1(torch.clamp(batch_x, min=0.0, max=20.0))
                size_factor = torch.clamp(counts.sum(1), min=1.0)
                loss_zinb = zinb_loss_fn(counts, mean, disp, pi, size_factor)
                cluster_weight = 0.0 if each_epoch < pretrain_epochs else lambda_cluster
                loss = loss_mae + lambda_cl * loss_cl + cluster_weight * loss_cluster + lambda_zinb * loss_zinb
                batch_losses.append(loss.cpu().detach().numpy())
                z_test.append(batch_z.cpu().detach().numpy())
                y_test.append(batch_y)

            z_test_stack = np.vstack(z_test)
            y_test_stack = np.hstack(y_test)
            y_pred = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20).fit_predict(z_test_stack)
            acc, f1, nmi, ari, homo, comp = evaluate(y_test_stack, y_pred)

            z_test_epoch.append(z_test)
            y_test_epoch.append(y_test)
            if ari > best_ari:
                best_ari = float(ari)
                best_epoch = each_epoch
                torch.save({'net': model.state_dict(), 'optimizer': opt_model.state_dict(), 'epoch': each_epoch, 'best_ari': best_ari}, save_model_path)

    return best_epoch, best_ari, z_test_epoch, y_test_epoch


def test(z_test_epoch, y_test_epoch, best_epoch, n_clusters, seed):
    z_test = np.vstack(z_test_epoch[best_epoch])
    y_test = np.hstack(y_test_epoch[best_epoch])
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20).fit(z_test)
    y_kmeans_test = kmeans.labels_
    acc, f1, nmi, ari, homo, comp = evaluate(y_test, y_kmeans_test)
    return {'CA': acc, 'NMI': nmi, 'ARI': ari}


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_head', type=int, default=config['graph_head'])
    parser.add_argument('--phi', type=float, default=config['phi'])
    parser.add_argument('--gcn_dim', type=int, default=config['gcn_dim'])
    parser.add_argument('--mlp_dim', type=int, default=config['mlp_dim'])
    parser.add_argument('--prob_feature', type=float, default=config['prob_feature'])
    parser.add_argument('--prob_edge', type=float, default=config['prob_edge'])
    parser.add_argument('--tau', type=float, default=config['tau'])
    parser.add_argument('--alpha', type=float, default=config['alpha'])
    parser.add_argument('--beta', type=float, default=config['beta'])
    parser.add_argument('--lambda_cl', type=float, default=config['lambda_cl'])
    parser.add_argument('--lambda_cluster', type=float, default=config['lambda_cluster'])
    parser.add_argument('--cluster_alpha', type=float, default=config['cluster_alpha'])
    parser.add_argument('--lambda_zinb', type=float, default=config['lambda_zinb'])
    parser.add_argument('--dropout', type=float, default=config['dropout'])
    parser.add_argument('--n_clusters', type=int)
    parser.add_argument('--lr', type=float, default=config['lr'])
    parser.add_argument('--pretrain_epochs', type=int, default=config['pretrain_epochs'])
    parser.add_argument('--seed', type=int, default=config['seed'])
    parser.add_argument('--epochs', type=int, default=config['epochs'])
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--save_model_path', type=str)
    parser.add_argument('--no_graph', action='store_true')
    args = parser.parse_args()
    train_loader, test_loader, input_dim = loader_construction(args.data_path)
    best_epoch, best_ari, z_test_epoch, y_test_epoch = train(
        train_loader, test_loader, input_dim, args.graph_head, args.phi, args.gcn_dim,
        args.mlp_dim, args.prob_feature, args.prob_edge, args.tau, args.alpha, args.beta,
        args.lambda_cl, args.lambda_cluster, args.cluster_alpha, args.lambda_zinb,
        args.dropout, args.lr, args.seed, args.epochs, args.pretrain_epochs, args.save_model_path, args.n_clusters, device, not args.no_graph)
    results = test(z_test_epoch, y_test_epoch, best_epoch, args.n_clusters, args.seed)
    print(results)
