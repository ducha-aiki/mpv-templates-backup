import torch, torchvision
from torchvision import transforms
import torch.optim as optim, torch.nn as nn, torch.nn.functional as F, torch.utils.data as data
import numpy as np, ssl
from PIL import Image

ssl._create_default_https_context = ssl._create_unverified_context # fix needed for downloading resnet weights
device = torch.device('cuda:0') # gpu training is typically much faster if available
# device = torch.device('cpu')
batch_size = 300

# read the dataset files and return the list of images and list of class labels
def readCUB(n=1000):
    cub_folder = 'CUB_200_2011/'
    f = open(cub_folder + "images.txt", 'r')
    cub_imgfn = [a.split(' ')[::-1] for a in f.read().split('\n')][:-1]
    cub_imgfn = np.array([(cub_folder + '/images/' + x[0]) for x in cub_imgfn])
    cub_label = np.loadtxt(cub_folder + "image_class_labels.txt", delimiter=" ", unpack=False)[:, 1].astype(int)
    cub_label = np.array(cub_label)

    train_mask = cub_label <= 100
    train_imgfn, train_label = cub_imgfn[train_mask], cub_label[train_mask]

    idx = []
    for i in range(0, max(train_label) + 1):
        idx = idx + ([x for (x, val) in enumerate(train_label) if val == i][0:n])
    cub_train = ([train_imgfn[x] for x in idx], train_label[idx])

    val_mask = (cub_label > 100) & (cub_label <= 150)
    cub_val = (cub_imgfn[val_mask], cub_label[val_mask])

    return cub_train, cub_val

# dataset structure used for testing/evaluation
class CUB(data.Dataset):
    
    def __init__(self, data, transform = None):
        self.img, self.labels = data
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.img[index]).convert('RGB')
        label =  self.labels[index].item()
        if self.transform is not None:
            img = self.transform(img)
        return img, label
            
    def __len__(self):
        return len(self.img)

# dataset structure used for training
class CUBtriplet(data.Dataset):
    
    def __init__(self, data, transform = None):

        self.img, self.labels = data
        self.transform = transform
        self.class_idx = np.split(np.arange(len(self.labels)), np.unique(self.labels, return_index=True)[1])
        self.hardneg = np.zeros(self.__len__(), np.uint32)


    def minehard(self, model):
        temp_loader = torch.utils.data.DataLoader(CUB((self.img, self.labels), transform=self.transform),
                                                  batch_size=batch_size, shuffle=False)
        des, labels = extract_images(model, temp_loader)
        # mine hard negatives and store them in "self.hardneg" - your code
        # ........
        # ........
        # ........
        # ........
        # ........

    # this is used to iterate over triplets 
    # and is called by the data-loader for batch construction
    def __getitem__(self, index):

        img1, label1 = Image.open(self.img[index]).convert('RGB'), self.labels[index].item()
        img1 = self.transform(img1)

        # img1 is the anchor, pick a positve and a hard negative - your code
        # ........
        # ........
        # ........
        # ........
        # ........
        
        return (img1, img2, img3)
    
    # this lets the data-loader know how many items are there to use
    # note that number of triplets = number of training images, since we are using each image once as an anchor
    def __len__(self):
        return len(self.img)

def extract_images(model, loader):
    model.eval()

    des, labels = [], []
    with torch.no_grad():
        for data, target in loader:
            des.append(model(data.to(device)))
            labels.append(target)

    des = torch.cat(des, dim=0)
    labels = torch.cat(labels, dim=0)
    return des, labels

class GDextractor(nn.Module):
    """
    Create A network that maps an image to an embedding (descriptor) with global pooling
    """    
    def __init__(self, input_net, dim, usemax = False):
        """
        Contructor takes a CNN as input
        input_net: a MobileNetV2 or ResNet18
        dim: output embedding dimensionality
        usemax: do MAC (max pooling) if true, otherwise SPoC (average pooling)
        """
        super(GDextractor, self).__init__()
        self.dim = dim
        self.usemax = usemax
        input_net = list(input_net.children())[:-1]
        self.net = nn.Sequential(*input_net)
        self.fc = nn.Linear(1280, dim)

        # create the network structure by using part of input_net - your code
        # ........
        # ........
        # ........

    def forward(self, x, eps = 1e-6):

        # x are the input image in the batch, extract the global descriptor - your code
        # ........
        # ........
        # ........
        # ........
        # z = .... # global descriptor

        return z

     
def test(model, test_loader):
    """
    Compute accuracy on the test set
    model: network
    test_loader: test_loader loading images and labels in batches
    """

    model.eval()
    des, labels = extract_images(model, test_loader)
    # Calculate all pairwise similarities
    similarities = des @ des.t()
    similarities.fill_diagonal_(float('-inf'))

    # sort similarities and see if the label of the top-ranked image is correct
    nn_inds = torch.argmax(similarities, dim=1).cpu()
    correct_matches = (labels == labels[nn_inds]).sum().item()

    return correct_matches / des.shape[0]


def triplet_loss(distances_pos, distances_neg, margin):
    # input: pos. and neg. distances per triplet in the batch
    # compute and return the loss per triplet - your code
    # ........

def train(model, train_loader, optimizer, margin = 0.5):
    """
    Training of an epoch with Triplet loss and triplets with hard-negatives
    model: network
    train_loader: train_loader loading triplets of the form (a,p,n) in batches. 
    optimizer: optimizer to use in the training
    margin: triplet loss margin
    """
    
    model.train() # first put the model into training mode
    model.apply(set_batchnorm_eval) # do not update the Batch-Norm running avg/std - helps to get improvements in a couple of epochs
    total_loss = 0
    
    for batch_idx, data in enumerate(train_loader): # iterate over batches
        # call the model and get global descriptors v1,v2,v3 for all anchors, positives and negatives in the batch
        v1, v2, v3 = model(data[0].to(device)), model(data[1].to(device)), model(data[2].to(device))
        
        # compute the required distances - your code
        # ........
        # ........
        loss = triplet_loss(distances_pos, distances_neg, margin)

        # update the network with back-propagation - your code
        # ........
        # ........
        
        total_loss = total_loss + loss.mean().cpu().item()
        
    print('Epoch average loss {:.6f}'.format(total_loss/batch_idx))


# sets the BN layers to eval mode
# so that the running statistics will not get updated
def set_batchnorm_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def main():
    ## input transformations for training (image augmentations) and testing
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # these hyper-parameters worked reasonably well for us
    lr=0.00001
    margin=.1

    cub_train, cub_val = readCUB(n=20) # keep only n images per class, lower number speeds up training
    trainset = CUBtriplet(cub_train, transform = transform_train)
    valset = CUB(cub_val, transform = transform_test)

    torch.manual_seed(0); np.random.seed(0)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size // 3, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False)

    # pre_model = torchvision.models.resnet18(pretrained=True) # load pre-trained ResNet18
    pre_model = torchvision.models.mobilenet_v2(pretrained=True) # load pre-trained MobileNetV2
    dim = 512  # dimensionality of the global descriptor
    model = GDextractor(pre_model, dim) # construct the network that extracts descriptors
    model.to(device)

    best_mp = test(model, valloader)
    torch.save({'epoch': 0,'val_mp': best_mp,'state_dict': model.state_dict()}, 'bestmodel.pth.tar')
    print('Before training, precision@1: {}'.format(np.round(best_mp, 4)), flush=True)

    optimizer = optim.AdamW(model.parameters(), lr=lr)

    print('Mining...')
    trainset.minehard(model)
    print('Training...')
    for epoch in range(1, 10 + 1):
            print('Epoch {}'.format(epoch), flush=True)
            train(model, trainloader, optimizer, margin)

            if epoch % 1 == 0:
                mp = test(model, valloader)
                print('Epoch {}, precision@1: {}'.format(epoch, np.round(mp, 4)), flush=True)

                if mp > best_mp:
                    best_mp = mp
                    torch.save({'epoch': epoch,'val_mp': mp,'state_dict': model.state_dict()}, 'bestmodel.pth.tar')

if __name__ == '__main__':
    main()
