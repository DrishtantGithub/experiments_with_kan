import torch
import torchvision
import torchvision.transforms as transforms
import ssl

# Fix macOS SSL issue
ssl._create_default_https_context = ssl._create_unverified_context

def get_cifar10_dataloaders(batch_size=128, quick=False, num_workers=2):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([transforms.ToTensor(),])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )

    if quick:
        trainset.data = trainset.data[:2000]
        trainset.targets = trainset.targets[:2000]
        testset.data = testset.data[:500]
        testset.targets = testset.targets[:500]

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return trainloader, testloader
