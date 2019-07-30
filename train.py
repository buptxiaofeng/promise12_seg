import torch 
import torch.nn as nn
import torch.backends.cudnn as cudnn
import json
from loss import dice_loss
from model import VNet
from dataset import PromiseDataset
from tqdm import tqdm

def train():
    json_file = open("parameters.json")
    parameters = json.load(json_file)
    json_file.close()
    net = VNet()
    net = torch.nn.DataParallel(net, device_ids=[0, 1, 2, 3])
    net.cuda()
    cudnn.benchmark = True

    optimizer = torch.optim.Adam(net.parameters(), lr = parameters["lr"])
    criterion = nn.BCELoss()
    promise_dataset = PromiseDataset(is_train = True)
    train_loader = torch.utils.data.DataLoader(dataset = promise_dataset, batch_size = parameters["batch_size"])
    for epoch in range(parameters["num_epochs"]):
        net.train()
        for i, (data, label) in enumerate(train_loader):
            optimizer.zero_grad()
            output = net(data)
            loss = dice_loss(output, label)
            loss.backward()
            optimizer.step()

        print ('Epoch [{}/{}], Loss: {:.4f}' .format(epoch+1, parameters["num_epochs"], loss.item()))

    torch.save(net.state_dict(), "weights/promise12_weight.pth")

if __name__ == "__main__":
    train()
