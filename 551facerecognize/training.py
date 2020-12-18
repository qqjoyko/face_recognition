import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import CNN
import torch.nn.functional

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def trainingCNN():
    # get the 32*32 pixel normalized picture with Center cut # Unify the pictures that need to be input to the model
    image_process = transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32),
                                        transforms.ToTensor(), transforms.Normalize(mean=[0.4, 0.4, 0.4],
                                                                                    std=[0.2, 0.2, 0.2])])
    train_data = ImageFolder(root="data/train_set", transform=image_process)
    test_data = ImageFolder(root="data/test_set", transform=image_process)
    train_loader = DataLoader(train_data, batch_size=10, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_data, batch_size=5, shuffle=True, num_workers=1)
    cnn = CNN.CNN().to(device)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)  # Use Adam optimizer
    for epoch in range(3):  # edit epoch here
        for step, (img, label) in enumerate(train_loader):
            img, label = img.to(device), label.to(device)
            sort = cnn(img)
            loss = torch.nn.functional.nll_loss(sort, label)  # Use maximum likelihood / log likelihood cost function
            optimizer.zero_grad()  # Pytorch will accumulate the gradient, so the gradient needs to be cleared # waste my a lot of time!
            loss.backward()
            optimizer.step()  # Use Adam for gradient update
            if (step + 1) % 3 == 0:
                print(
                    f'Train Epoch: {epoch + 1} [{step * len(img)}/{len(train_loader.dataset)} ({100. * step / len(train_loader):.0f}%)]\tLoss: {loss.item():.4f}')

    cnn.eval()  # Check model accuracy
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for img, label in test_loader:
            img, label = img.to(device), label.to(device)
            sort = cnn(img)
            test_loss += torch.nn.functional.nll_loss(sort, label, reduction='sum').item()
            pred = sort.max(1, keepdim=True)[1]
            correct += pred.eq(label.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print(f'\ntest loss={test_loss:.4f}, accuracy={float(correct) / len(test_loader.dataset):.4f}\n')
    try:
        os.makedirs("data/model/")
    except OSError as e:
        if e.errno == 17:  # 17 means this file are already exists
            pass
        else:
            raise
    torch.save(cnn.state_dict(), "data/model/trained.pkl")  # Save the model weights to the model directory


if __name__ == '__main__':
    trainingCNN()
