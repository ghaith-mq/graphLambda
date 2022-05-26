import torch.nn.functional as F
import torch

# Training function
def train(model, train_loader,epoch,device='cpu',optimizer,scheduler=None):
    model.train()
    loss_all = 0
    error = 0
    for data in train_loader:
        optimizer.zero_grad()
        loss = F.mse_loss(model(data), data[0].y)
        loss.backward(retain_graph=True)
        loss_all += loss.item() * data[0].num_graphs
        error += (model(data) - data[0].y).abs().sum().item()  # MAE error
        torch.nn.utils.clip_grad_value_(model.parameters(), 1)
        optimizer.step()
    return loss_all / len(train_loader.dataset), error / len(train_loader.dataset)

# test function
def test(model, loader,device):
    model.eval()
    error = 0

    for data in loader:
        error += (model(data) - data[0].y).abs().sum().item()  # MAE error
    return error / len(loader.dataset)

# Test function for benchmarks
def test_predictions(model, loader):
    model.eval()
    pred = []
    true = []
    for data in loader:
        pred += model(data).detach().cpu().numpy().tolist()
        true += data[0].y.detach().cpu().numpy().tolist()
    return pred, true
