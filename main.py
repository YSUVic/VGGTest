import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from parameters import args_parser
from torchvision import datasets, transforms as T

class VggNet(nn.Module):
    def __init__(self, num_classes=3):
        super(VggNet, self).__init__()
        self.Conv = torch.nn.Sequential(
            # 3*224*224  conv1
            torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # 64*112*112   conv2
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # 128*56*56    conv3
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # 256*28*28    conv4
            torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        self.Classes = torch.nn.Sequential(
            torch.nn.Linear(14 * 14 * 512, 1060),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(1060, 1060),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(1060, num_classes))

    def forward(self, inputs):
        x = self.Conv(inputs)
        x = x.view(-1, 14 * 14 * 512)
        x = self.Classes(x)
        return x


# """
# The main CovidAID and CheXNet implementation
# """
# class DenseNet121(nn.Module):
#     """Model modified.
#     The architecture of our model is the same as standard DenseNet121
#     except the classifier layer which has an additional sigmoid function.
#     """
#     def __init__(self, out_size):
#         super(DenseNet121, self).__init__()
#         self.densenet121 = torchvision.models.densenet121(pretrained=True)
#         num_ftrs = self.densenet121.classifier.in_features
#         self.densenet121.classifier = nn.Sequential(
#             nn.Linear(num_ftrs, out_size),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         x = self.densenet121(x)
#         return x
#
# class CovidAID(DenseNet121):
#     """
#     Modified DenseNet network with 4 classes
#     """
#     def __init__(self):
#         NUM_CLASSES = 3
#         super(CovidAID, self).__init__(NUM_CLASSES)



if __name__ == '__main__':
    # parse args
    args = args_parser()

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"

#—数据—————————————————————————————————————————————————————————————————————————————————————————
    normalize = T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        normalize
    ])
    train_data = ImageFolder('../data/Train', transform=transform)
    test_data = ImageFolder('../data/Test', transform=transform)
# ——————————————————————————————————————————————————————————————————————————————————————————
    train_dataloaders = DataLoader(train_data, batch_size=args.batchsize)
    test_dataloader = DataLoader(test_data, batch_size=args.batchsize * 2)

    model=VggNet().to(device)
    loss_fn=nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum,
                                          weight_decay=args.weight_decay)
    for t in range(args.epoch):
        size = len(train_dataloaders.dataset)
        model.train()
        for batch, (X, y) in enumerate(train_dataloaders):
            X, y = X.to(device), y.to(device)
            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            if (batch+1) % 10 == 0:
                loss, current = loss.item(), (batch+1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    model.eval()
    test_loss, correct, f1 = 0, 0, 0
    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            test_loss += loss_fn(y_pred, y).item()
            correct += (y_pred.argmax(1) == y).type(torch.float).sum().item()
            f1 += f1_score(y.cpu(), y_pred.argmax(1).cpu(), average='micro')
    test_loss /= num_batches
    correct /= size
    f1 /= num_batches
    print(f"Test Error: \n Accuracy: {correct:>8f}, Avg loss: {test_loss:>8f}, F1: {f1:>8f} \n")

