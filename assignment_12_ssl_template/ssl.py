import torch, torchvision
from torchvision import transforms
import torch.optim as optim, torch.nn as nn, torch.utils.data as data
import random, numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
batch_size = 512
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class MultiViewDataset(data.Dataset):
    """ Wraps a torchvision dataset to return two augmented views """
    def __init__(self, dataset):
        self.dataset = dataset

        # define the random augmentations for training - your code
        # ........
        # self.transform =
        # ........

    def __getitem__(self, index):
        img, _ = self.dataset[index] # Get image, ignore original label
        img = img.convert('RGB') # Assume conversion needed if not RGB

        # transforms the image into two augmented views - your code
        # ........
        # ........
        # ........

        return view1, view2

    def __len__(self):
        return len(self.dataset)


def extract_images(model, loader):
    """ Extract all imges from loader via model. This function does not normalize the descriptors! """

    model.eval()

    des, labels = [], []
    with torch.no_grad():
        for data, target in loader:
            des.append(model(data.to(device)))
            labels.append(target)

    des = torch.cat(des, dim=0)
    labels = torch.cat(labels, dim=0)
    return des, labels


class Encoder(nn.Module):
    """ Encoder Network: Backbone + Projection head (MLP) """
    def __init__(self, backbone, backbone_dim=1280, projection_size=128, projection_hidden_size=2048):
        super().__init__()
        self.backbone = nn.Sequential(*list(backbone.children())[:-1], nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())

        # implement the MLP projection head - your code
        # ........
        # self.projector =
        # ........

    def forward(self, x):

        # encode and project the input images x - your code
        # ........
        # ........
        # ........

        return projections


def test(model, train_loader_knn, test_loader_knn, knn=1):
    """
    k-NN classification of the given model

    Args:
        model: backbone of the encoder
        train_loader_knn: DataLoader for the support set of the test dataset
        test_loader_knn: DataLoader for the query set of the test dataset
        knn: number of nearest neighbors to use for classification

    Returns:
        accuracy: a single value [0, 1] showcasing the model's performance
    """

    encoder.eval()

    train_features, train_labels = extract_images(encoder, train_loader_knn)
    test_features, test_labels = extract_images(encoder, test_loader_knn)

    # find the label of the 1-NN per test sample and compute accuracy - your code
    # ........
    # ........
    # ........

    return accuracy


def xent_loss(z1, z2, temperature=0.1):
    """
    NT-Xent loss (InfoNCE) for SimCLR

    Args:
        z1, z2: projections from the two views
        temperature: temperature parameter for scaling

    Returns:
        loss value
    """
    # compute the NT-Xent loss for the **full batch** - your code
    # ........
    # ........
    # ........

    return loss


def train(encoder, train_loader, optimizer, scheduler, temperature):
    """
    Training loop for one epoch

    Args:
        encoder: Encoder class wrapper for the backbone
        train_loader: DataLoader for the training set
        optimizer: optimizer for the encoder
        scheduler: learning rate scheduler
        temperature: temperature parameter of the loss

    Returns:
        avg_loss: average loss scalar for the epoch
    """

    # training loop - your code
    # ........
    # ........
    # ........

    return avg_loss


def main():

    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    ## --- Hyperparameters ---
    lr = 0.3
    epochs = 20
    temperature = 0.1

    ## --- Load Data ---
    imagenette_train = torchvision.datasets.Imagenette(root='./data', download=False)
    trainset = MultiViewDataset(imagenette_train)

    cifar_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_test)
    cifar_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    ## --- Setup ---
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size // 2, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    knn_train_loader = torch.utils.data.DataLoader(cifar_train, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    knn_test_loader = torch.utils.data.DataLoader(cifar_test, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    backbone = torchvision.models.mobilenet_v2()
    encoder = Encoder(backbone).to(device)

    initial_knn_acc = test(encoder.backbone, knn_train_loader, knn_test_loader)
    print(f'Before training, CIFAR-10 1-NN accuracy: {initial_knn_acc:.4f}', flush=True)
    best_knn_acc = initial_knn_acc

    optimizer = optim.AdamW(list(encoder.parameters()), lr=lr, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(trainloader))

    ## --- Training Loop ---
    for epoch in range(1, epochs + 1):
        print(f'Epoch {epoch}', flush=True)
        loss = train(encoder, trainloader, optimizer, scheduler, temperature)
        print(f'Epoch {epoch} average loss: {loss:.4f}')

        if epoch % 2 == 0 or epoch == epochs:
            knn_acc = test(encoder.backbone, knn_train_loader, knn_test_loader)
            print(f'Epoch {epoch}, CIFAR-10 1-NN accuracy: {knn_acc:.4f}', flush=True)

            # If the validation accuracy is higher, save the best performing model so far
            if knn_acc > best_knn_acc:
                best_knn_acc = knn_acc
                torch.save({'epoch': epoch,
                            'knn_acc': knn_acc,
                            'state_dict': encoder.backbone.state_dict()},
                            'model.pth')
if __name__ == '__main__':
    main()
