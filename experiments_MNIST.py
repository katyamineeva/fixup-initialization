num_classes = 10
batch_size = 256

mnist_train_dataset = dsets.MNIST(root='./MNIST/', 
                            train=True, 
                            transform=transforms.ToTensor(),
                            download=True)

mnist_test_dataset = dsets.MNIST(root='./MNIST/', 
                           train=False, 
                           transform=transforms.ToTensor())

mnist_train_loader = torch.utils.data.DataLoader(dataset=mnist_train_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False)

mnist_test_loader = torch.utils.data.DataLoader(dataset=mnist_test_dataset, 
                                                batch_size=batch_size,         
                                                shuffle=False)

def train_mnist(n_blocks, n_channels, reg, lr, n_epochs):
    model = ResNet(10, block=BottleneckBlock, n_blocks=n_blocks, in_channels=1, n_channels=n_channels, reg=reg)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = ReduceLROnPlateau(opt, 'min', patience=3)
    train(model, opt, sch, n_epochs, mnist_train_loader, mnist_test_loader)

train_mnist(n_blocks=[3, 4, 6, 3], n_channels=16, reg='batch_norm', lr=0.001, n_epochs=3)
train_mnist(n_blocks=[3, 4, 6, 3], n_channels=16, reg='fixup', lr=0.001, n_epochs=3)
