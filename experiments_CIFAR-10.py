num_classes = 10
batch_size = 256

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

cifar10_train_dataset = dsets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

cifar10_test_dataset = dsets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

cifar10_train_loader = torch.utils.data.DataLoader(cifar10_train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True)

cifar10_test_loader = torch.utils.data.DataLoader(cifar10_test_dataset, batch_size=batch_size,
                                                  shuffle=False)

cifar10_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def train_cifar10(n_blocks, n_channels, reg, lr, n_epochs):
    model = ResNet(10, block=BottleneckBlock, n_blocks=n_blocks, in_channels=3, n_channels=n_channels, reg=reg).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = ReduceLROnPlateau(opt, 'min', patience=3)
    train(model, opt, sch, n_epochs, cifar10_train_loader, cifar10_test_loader)

train_cifar10(n_blocks=[3, 4, 6, 3], n_channels=64, reg='batch_norm', lr=0.001, n_epochs=20)
train_cifar10(n_blocks=[3, 4, 6, 3], n_channels=64, reg='fixup', lr=0.001, n_epochs=20)
