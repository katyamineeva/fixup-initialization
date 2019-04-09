'''
flattens each element of a given batch
used as connection between 2d-layers and 1d-layers
- input shape [batch_size, d1, d2, ..., dn]
- output shape [batch_size, d1 * d2 * ... * dn]
'''
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)

'''
repeats the channels of each image in a given batch dilation_factor times
- input shape [batch_size, n_channels, height, width]
- output shape [batch_size, dilation_factor * n_channels, height, width]
'''
class Dilate2d(nn.Module):
    def __init__(self, dilation_factor):
        super().__init__()
        self.dilation_factor = dilation_factor
        
    def forward(self, x):
        out = x.repeat((1, self.dilation_factor, 1, 1))
        return out

'''
multiplies by a scalar multiplier elementwise
the multiplier is a parameter of the module and it's value is optimized in training mode
- input shape can be any
- output shape is the same as input shape
'''
class ScalarMultiplier(nn.Module):
    def __init__(self):
        super().__init__()
        self.multiplier = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        out = x * self.multiplier
        return out
    
'''
adds a scalar bias elementwise
the bias is a parameter of the module and it's value is optimized in training mode
- input shape can be any
- output shape is the same as input shape
'''
    
class ScalarBias(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        out = x + self.bias
        return out
    
def conv_fixup_init(layer, scaling_factor=1):
    # here we assume that kernel is squared matrix
    k = layer.kernel_size[0]
    n = layer.out_channels
    sigma = sqrt(2 / (k * k * n)) * scaling_factor
    layer.weight.data.normal_(0, sigma)
    return layer
    
class BasicBlock(nn.Module):
    scaling_factor = 1
    
    def __init__(self, in_channels,  n_channels, stride=1, reg=None):
        super().__init__()
        self.fixup = (reg == 'fixup')
        
        conv1 = nn.Conv2d(in_channels, n_channels, kernel_size=3, padding=1, stride=stride)
        conv2 = nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        if stride == 1:
            x_transform = Dilate2d(n_channels // in_channels)
        else:
            x_transform = nn.Conv2d(in_channels, n_channels, kernel_size=1, stride=stride)

        
        if reg == 'fixup':            
            if stride == 2:
                x_transform = conv_fixup_init(x_transform)
                # self.x_transform = nn.Sequential(ScalarBias(), x_transform)
                self.x_transform = x_transform
            else:
                self.x_transform = x_transform

            conv1 = conv_fixup_init(conv1, self.scaling_factor)
            conv2.weight.data.zero_()
            
            self.layers = nn.Sequential(
                ScalarBias(),
                conv1,
                ScalarBias(),
                nn.ReLU(),
                ScalarBias(),
                conv2,
                ScalarMultiplier(),
                ScalarBias()
            )
        elif reg == 'batch_norm':
            if stride == 2:
                self.x_transform = nn.Sequential(x_transform, nn.BatchNorm2d(n_channels))
            else:
                self.x_transform = x_transform
            
            self.layers = nn.Sequential(
                conv1,
                nn.BatchNorm2d(n_channels),
                nn.ReLU(),
                conv2,
                nn.BatchNorm2d(n_channels)
            )
        else:
            self.x_transform = x_transform
            self.layers = nn.Sequential(
                conv1,
                nn.ReLU(),
                conv2
            )
            
    def forward(self, x):
        output = self.layers(x)
        x = self.x_transform(x)
        output = self.relu(output + x)
        return output

class BottleneckBlock(nn.Module):
    scaling_factor = 1
    
    def __init__(self, in_channels,  n_channels, stride=1, reg=None):
        super().__init__()
        self.fixup = (reg == 'fixup')
        
        conv1 = nn.Conv2d(in_channels, n_channels, kernel_size=1, stride=stride)
        conv2 = nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1)
        conv3 = nn.Conv2d(n_channels, 4 * n_channels, kernel_size=1)
        self.relu = nn.ReLU()
        if stride == 1:
            x_transform = Dilate2d(4 * n_channels // in_channels)
        else:
            x_transform = nn.Conv2d(in_channels, 4 * n_channels, kernel_size=1, stride=stride)
        
        if reg == 'fixup':
            if stride == 2:
                x_transform = conv_fixup_init(x_transform)
                # self.x_transform = nn.Sequential(ScalarBias(), x_transform)
                self.x_transform = x_transform
            else:
                self.x_transform = x_transform
            
            conv_1 = conv_fixup_init(conv1, self.scaling_factor)
            conv_2 = conv_fixup_init(conv2, self.scaling_factor)
            conv3.weight.data.zero_()
            
            self.layers = nn.Sequential(
                ScalarBias(),
                conv1,
                ScalarBias(),
                nn.ReLU(),
                ScalarBias(),
                conv2,
                ScalarBias(),
                conv3,
                ScalarMultiplier(),
                ScalarBias()
            )
        elif reg == 'batch_norm':
            if stride == 2:
                self.x_transform = nn.Sequential(x_transform, nn.BatchNorm2d(4 * n_channels))
            else:
                self.x_transform = x_transform
            
            self.layers = nn.Sequential(
                conv1,
                nn.BatchNorm2d(n_channels),
                nn.ReLU(),
                conv2,
                nn.BatchNorm2d(n_channels),
                nn.ReLU(),
                conv3,
                nn.BatchNorm2d(4 * n_channels)
            )
        else:
            self.x_transform = x_transform
            self.layers = nn.Sequential(
                conv1,
                nn.ReLU(),
                conv2,
                nn.ReLU(),
                conv3
            )
            
    def forward(self, x):
        output = self.layers(x)
        x = self.x_transform(x)
        output = self.relu(output + x)
        return output

    
           
class ResNet(nn.Module):
    def __init__(self, n_classes, in_channels=1, n_channels=16, block=BasicBlock, n_blocks=[2, 2, 2, 2], reg=None):
        super().__init__()
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.block = block
        self.dilation = 1 if block == BasicBlock else 4
        self.n_blocks = n_blocks
        self.reg = reg
        
        m = 3              # a small positive integer constant for fixup initialization
        L = sum(n_blocks)  # number of residual blocks
        BasicBlock.scaling_factor = L ** (-1.0 / (2 * m - 2))
        BottleneckBlock.scaling_factor = L ** (-1.0 / (2 * m - 2))
        
        # layers = [ScalarBias(), nn.Conv2d(in_channels, n_channels, kernel_size=7, stride=2)]
        layers = [nn.Conv2d(in_channels, n_channels, kernel_size=7, stride=2)]
        
        if reg == 'batch_norm':
            layers += [nn.BatchNorm2d(n_channels)]
        # layers += [nn.MaxPool2d(kernel_size=3, stride=2), ScalarBias(), nn.ReLU()]
        layers += [nn.MaxPool2d(kernel_size=3, stride=2), nn.ReLU()]
        
        for i in range(len(n_blocks)):
            layers += self.__conv_block(i)
        
        r = 5
        layers += [nn.AdaptiveAvgPool2d((r, r))]
        
        n_channels = self.dilation * 2 ** (len(n_blocks) - 1) * n_channels
        conv_out_size = r * r * n_channels
        # layers += [Flatten(), ScalarBias(), nn.Linear(conv_out_size, n_classes)]
        layers += [Flatten(), nn.Linear(conv_out_size, n_classes)]
        
        if reg == 'fixup':
            # standard initialization
            # layers[1] = conv_fixup_init(layers[1])
            layers[0] = conv_fixup_init(layers[0])
            
            # linear layer initialized to 0
            layers[-1].weight.data.zero_()
            layers[-1].bias.data.zero_()

        self.layers = nn.Sequential(*layers)
    
    def __conv_block(self, i):
        n_channels = 2 ** i * self.n_channels
        n_blocks = self.n_blocks[i]
        if i == 0:
            layers = [self.block(n_channels, n_channels, reg=self.reg, stride=2)]
        else:
            layers = [self.block(self.dilation * n_channels // 2, n_channels, reg=self.reg, stride=2)]
        layers += [self.block(self.dilation * n_channels, n_channels, reg=self.reg) for j in range(n_blocks - 1)]
        return layers
            
        
    def forward(self, x):
        out = self.layers(x)
        return out
        