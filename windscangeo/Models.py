import torch
import torch.nn as nn

class ConventionalCNN(nn.Module):
    """
    A simple CNN for image regression tasks.
    This model consists of a series of convolutional layers followed by
    fully connected layers. It is designed to process images and output a
    single regression value (e.g., wind speed).
    """
    def __init__(
        self,
        image_height: int,
        image_width: int,
        features_cnn: list[int],
        kernel_size: int,
        in_channels: int,
        activation_cnn: nn.Module = nn.ReLU(),
        activation_final: nn.Module = nn.Identity(),
        stride: int = 1,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.activation_cnn = activation_cnn
        self.activation_final = activation_final
        self.dropout_rate = dropout_rate

        # ------- Convolutional backbone -------
        self.convs = nn.ModuleList()
        for feature in features_cnn:
            self.convs.extend(
                [
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=feature,
                        kernel_size=kernel_size,
                        padding=1,
                        stride=stride,
                    ),
                    self.activation_cnn,
                    nn.MaxPool2d(kernel_size=2),
                    nn.Dropout(self.dropout_rate),
                ]
            )
            in_channels = feature

        # ------- Classifier / regressor head -------
        self.flattened_size = self._get_flattened_size(image_height, image_width)
        self.fc_cnn = nn.Linear(self.flattened_size, 64)
        self.dropout_cnn = nn.Dropout(self.dropout_rate)
        self.head = nn.Sequential(
            nn.Linear(64, 16),
            self.activation_cnn,
            nn.Dropout(self.dropout_rate),
            nn.Linear(16, 1),
            self.activation_final,
        )

    def _get_flattened_size(self, h, w):
        x = torch.zeros(1, self.convs[0].in_channels, h, w)
        for layer in self.convs:
            x = layer(x)
        return x.numel()

    def forward(self, image):
        x = image
        for layer in self.convs:
            x = layer(x)
        x = x.view(x.size(0), -1)              # flatten
        x = self.activation_cnn(self.fc_cnn(x))
        x = self.dropout_cnn(x)
        out = self.head(x)
        return out
    

# from kaggle
# https://www.kaggle.com/code/umongsain/vision-transformer-from-scratch-pytorch

class ViT(nn.Module):
    """
    Vision Transformer (ViT) model for image classification or regression tasks.
    This model consists of an image-to-sequence layer, a transformer encoder,
    and a multi-layer perceptron (MLP) head for classification or regression.

    Taken from # https://www.kaggle.com/code/umongsain/vision-transformer-from-scratch-pytorch
    """
    def __init__(
        self,
        img_size,
        patch_size,
        n_channels,
        d_model,
        nhead,
        dim_feedforward,
        blocks,
        mlp_head_units,
        n_classes,
    ):
        super().__init__()
        """
        Args:
            img_size: Size of the image
            patch_size: Size of the patch
            n_channels: Number of image channels
            d_model: The number of features in the transformer encoder
            nhead: The number of heads in the multiheadattention models
            dim_feedforward: The dimension of the feedforward network model in the encoder
            blocks: The number of sub-encoder-layers in the encoder
            mlp_head_units: The hidden units of mlp_head
            n_classes: The number of output classes
        """
        self.img2seq = Img2Seq(img_size, patch_size, n_channels, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, activation="gelu", batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, blocks
        )
        self.mlp = get_mlp(d_model, mlp_head_units, n_classes)
        
        self.output = nn.Identity() # For regression

    def forward(self, batch):

        batch = self.img2seq(batch)
        batch = self.transformer_encoder(batch)
        batch = batch[:, 0, :]
        batch = self.mlp(batch)
        output = self.output(batch)
        return output
    

class Img2Seq(nn.Module):
    """
    This layers takes a batch of images as input and
    returns a batch of sequences
    
    Shape:
        input: (b, h, w, c)
        output: (b, s, d)
    
    taken from https://www.kaggle.com/code/umongsain/vision-transformer-from-scratch-pytorch
    """
    def __init__(self, img_size, patch_size, n_channels, d_model):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size

        nh, nw = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        n_tokens = nh * nw

        token_dim = patch_size[0] * patch_size[1] * n_channels
        self.linear = nn.Linear(token_dim, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_emb = nn.Parameter(torch.randn(n_tokens, d_model))

    def __call__(self, batch):
        batch = patchify(batch, self.patch_size)

        b, c, nh, nw, ph, pw = batch.shape

        # Flattening the patches
        batch = torch.permute(batch, [0, 2, 3, 4, 5, 1])
        batch = torch.reshape(batch, [b, nh * nw, ph * pw * c])

        batch = self.linear(batch)
        cls = self.cls_token.expand([b, -1, -1])
        emb = batch + self.pos_emb

        return torch.cat([cls, emb], axis=1)
    
def get_mlp(in_features, hidden_units, out_features):
    """
    Returns a MLP head

    taken from https://www.kaggle.com/code/umongsain/vision-transformer-from-scratch-pytorch
    """
    dims = [in_features] + hidden_units + [out_features]
    layers = []
    for dim1, dim2 in zip(dims[:-2], dims[1:-1]):
        layers.append(nn.Linear(dim1, dim2))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)

def patchify(batch, patch_size):
    """
    Patchify the batch of images
        
    Shape:
        batch: (b, h, w, c)
        output: (b, nh, nw, ph, pw, c)

    taken from https://www.kaggle.com/code/umongsain/vision-transformer-from-scratch-pytorch
    """
    b, c, h, w = batch.shape
    ph, pw = patch_size
    nh, nw = h // ph, w // pw

    batch_patches = torch.reshape(batch, (b, c, nh, ph, nw, pw))
    batch_patches = torch.permute(batch_patches, (0, 1, 2, 4, 3, 5))

    return batch_patches


######################
# RESNET from https://github.com/JayPatwardhan/ResNet-PyTorch/blob/master/ResNet/ResNet.py


class Bottleneck(nn.Module):
    """
    taken from https://github.com/JayPatwardhan/ResNet-PyTorch/blob/master/ResNet/ResNet.py
    A bottleneck block for ResNet architecture.
    This block consists of three convolutional layers with batch normalization
    and ReLU activation. The first layer reduces the number of channels,
    the second layer applies a 3x3 convolution, and the third layer expands
    the number of channels back to the original size. The block also supports
    downsampling through an optional identity downsample layer.
    """
    expansion = 4
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels*self.expansion)
        
        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        
        x = self.relu(self.batch_norm2(self.conv2(x)))
        
        x = self.conv3(x)
        x = self.batch_norm3(x)
        
        #downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        #add identity
        x+=identity
        x=self.relu(x)
        
        return x

class Block(nn.Module):
    """
    A basic block for ResNet architecture.
    This block consists of two convolutional layers with batch normalization
    and ReLU activation. The first layer applies a 3x3 convolution, and the
    second layer applies another 3x3 convolution. The block also supports
    downsampling through an optional identity downsample layer. The expansion
    factor is set to 1, meaning the output channels are the same as the input
    channels.

    taken from https://github.com/JayPatwardhan/ResNet-PyTorch/blob/master/ResNet/ResNet.py
    """

    expansion = 1
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()
       

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
      identity = x.clone()

      x = self.relu(self.batch_norm2(self.conv1(x)))
      x = self.batch_norm2(self.conv2(x))

      if self.i_downsample is not None:
          identity = self.i_downsample(identity)
      print(x.shape)
      print(identity.shape)
      x += identity
      x = self.relu(x)
      return x



class ResNet(nn.Module):

    """
    A ResNet model for image classification or regression tasks.
    This model consists of an initial convolutional layer, followed by a series
    of residual blocks, and a fully connected layer for classification or regression.
    The number of residual blocks in each layer is specified by the `layer_list` parameter.
    taken from https://github.com/JayPatwardhan/ResNet-PyTorch/blob/master/ResNet/ResNet.py
    """
    def __init__(self, ResBlock, layer_list, num_classes, num_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*ResBlock.expansion, num_classes)
        
    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        
        return x
        
    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != planes*ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes*ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes*ResBlock.expansion)
            )
            
        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes*ResBlock.expansion
        
        for i in range(blocks-1):
            layers.append(ResBlock(self.in_channels, planes))
            
        return nn.Sequential(*layers)

        
        
def ResNet50(num_classes, channels=3):
    return ResNet(Bottleneck, [3,4,6,3], num_classes, channels)
    
def ResNet101(num_classes, channels=3):
    return ResNet(Bottleneck, [3,4,23,3], num_classes, channels)

def ResNet152(num_classes, channels=3):
    return ResNet(Bottleneck, [3,8,36,3], num_classes, channels)