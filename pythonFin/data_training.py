import data_building
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import CNN
import torch.nn.functional


DEFAULT_MODEL = "retrain.pkl"
model_save = "data/model"
BATCH_SIZE = 10
EPOCHS = 3
LR = 0.01


# get the 32*32 pixel normalized picture with Center cut # Unify the pictures that need to be input to the model
def image_process():
    return transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32),
                               transforms.ToTensor(), transforms.Normalize(mean=[0.4, 0.4, 0.4],
                                                                           std=[0.2, 0.2, 0.2])
    ])


def loading(batch_size=10, num_workers=2):
    train_data = ImageFolder(root=data_building.train_set, transform=image_process())
    test_data = ImageFolder(root=data_building.test_set, transform=image_process())
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return train_loader, test_loader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model():
    train_loader, test_loader = loading(batch_size=BATCH_SIZE)
    net = CNN.CNN().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)  # Use Adam optimizer
    for epoch in range(EPOCHS):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            output = net(x)
            loss = torch.nn.functional.nll_loss(output, y)  # Use maximum likelihood / log likelihood cost function
            optimizer.zero_grad()  # Pytorch will accumulate the gradient, so the gradient needs to be cleared # waste my lots of time
            loss.backward()
            optimizer.step()  # Use Adam for gradient update
            if (step + 1) % 3 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch + 1, step * len(x), len(train_loader.dataset),
                    100. * step / len(train_loader), loss.item()))
    test(net, test_loader)  # Check model accuracy
    torch.save(net.state_dict(), os.path.join(model_save, DEFAULT_MODEL))  # Save the model weights to the model directory
    return net


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            test_loss += torch.nn.functional.nll_loss(output, y, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(y.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\ntest loss={:.4f}, accuracy={:.4f}\n'.format(test_loss, float(correct) / len(test_loader.dataset)))


