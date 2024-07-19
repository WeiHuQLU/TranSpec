import torch
from torch import nn
from util import PositionalEncoding
import math
from torch.nn import functional as F

class Linear(nn.Linear):
    """
    Custom Linear layer with specialized initialization.

    Args:
        d_in (int): Input dimension.
        d_out (int): Output dimension.
        bias (bool): Whether to use bias.
        init (str): Initialization method, "default" or "final".
    """
    def __init__(
        self,
        d_in: int,
        d_out: int,
        bias: bool = True,
        init: str = "default",
    ):
        super(Linear, self).__init__(d_in, d_out, bias=bias)

        self.use_bias = bias

        if self.use_bias:
            with torch.no_grad():
                self.bias.fill_(0)

        if init == "default":
            self._trunc_normal_init(1.0)
        elif init == "final":
            self._zero_init(False)
        else:
            raise ValueError("Invalid init method.")

    def _trunc_normal_init(self, scale=1.0):
        """
        Truncated normal distribution initialization.

        Args:
            scale (float): Scaling factor for the standard deviation.
        """
        TRUNCATED_NORMAL_STDDEV_FACTOR = 0.87962566103423978
        _, fan_in = self.weight.shape
        scale = scale / max(1, fan_in)
        std = (scale ** 0.5) / TRUNCATED_NORMAL_STDDEV_FACTOR
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std)

    def _zero_init(self, use_bias=True):
        """
        Zero initialization method.

        Args:
            use_bias (bool): Whether to initialize bias.
        """
        with torch.no_grad():
            self.weight.fill_(0.0)
            if use_bias:
                with torch.no_grad():
                    self.bias.fill_(1.0)


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) with customizable layers and activation.

    Args:
        d_in (int): Input dimension.
        n_layers (int): Number of hidden layers.
        d_hidden (int): Hidden layer dimension.
        d_out (int): Output dimension.
        activation (nn.Module): Activation function.
        bias (bool): Whether to use bias.
    """
    def __init__(
        self,
        d_in,
        n_layers,
        d_hidden,
        d_out,
        activation=nn.ReLU(),
        bias=True,
    ):
        super(MLP, self).__init__()
        layers = [Linear(d_in, d_hidden, bias), activation]
        for _ in range(n_layers):
            layers += [Linear(d_hidden, d_hidden, bias), activation]
        layers.append(Linear(d_hidden, d_out, bias, init="final"))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass for the MLP.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.mlp(x)
        return x


class CNN(nn.Module):
    """
    Convolutional Neural Network (CNN) with multiple layers and activation.

    Args:
        input_channels (int): Number of input channels.
        d_model (int): Dimension of the model.
        activation (nn.Module): Activation function.
        bias (bool): Whether to use bias.
    """
    def __init__(
        self,
        input_channels,
        d_model,
        activation=nn.ReLU(),
        bias=True,
    ):
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, 32, 10, 2),
            activation,
            nn.Conv1d(32, 32, 3, 1),
            activation,
            nn.Conv1d(32, 64, 10, 2),
            activation,
            nn.Conv1d(64, 64, 3, 1),
            activation,
            nn.Conv1d(64, 128, 3, 1),
            activation,
            nn.Conv1d(128, 256, 3, 1),
            activation,
            nn.Conv1d(256, 512, 3, 1),
            activation,
            Linear(735, d_model, bias)
        )

    def forward(self, x):
        """
        Forward pass for the CNN.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.cnn(x)
        return x


class Model(nn.Module):
    """
    Combined model incorporating Transformer, CNN, and MLP components.

    Args:
        d_model (int): Dimension of the model.
        en_layers (int): Number of encoder layers.
        de_layers (int): Number of decoder layers.
        en_head (int): Number of heads in the encoder's multi-head attention.
        de_head (int): Number of heads in the decoder's multi-head attention.
        en_dim_feed (int): Feed-forward dimension in the encoder.
        de_dim_feed (int): Feed-forward dimension in the decoder.
        dropout (float): Dropout rate.
        max_len (int): Maximum length for positional encoding.
        vocab_size (int): Vocabulary size.
        bias (bool): Whether to use bias.
        use_cnn (bool): Whether to use CNN.
        use_mlp (bool): Whether to use MLP.
        input_channels (int): Number of input channels, default 1.
        reshape_size (int): Reshape size for individual spectra, default 10.
    """
    def __init__(
        self,
        d_model,
        en_layers,
        de_layers,
        en_head,
        de_head,
        en_dim_feed,
        de_dim_feed,
        dropout,
        max_len,
        vocab_size,
        bias=True,
        use_cnn=False,
        use_mlp=False,
        input_channels=1,
        reshape_size=10,
    ):
        super(Model, self).__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model, en_head, en_dim_feed, dropout,
                                                        batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, en_layers)

        self.decoder_layer = nn.TransformerDecoderLayer(d_model, de_head, de_dim_feed, dropout,
                                                        batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, de_layers)

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.pe = PositionalEncoding(d_model, dropout, max_len)

        self.pre_type = nn.Sequential(
            Linear(d_model, d_model, bias),
            nn.ReLU(),
            Linear(d_model, vocab_size, bias, init="final"),
        )

        self.use_cnn = use_cnn
        self.use_mlp = use_mlp
        self.input_channels = input_channels  
        self.reshape_size = reshape_size 

        if self.use_cnn:
            self.c = CNN(input_channels, d_model, activation=nn.ReLU())

        if self.use_mlp:
            self.m = MLP(300, 4, 512, d_model, activation=nn.ReLU())

    def forward(self, en, de_1, tgt_mask, tgt_key_padding_mask):
        """
        Forward pass for the combined model.

        Args:
            en (torch.Tensor): Encoder input tensor.
            de_1 (torch.Tensor): Decoder input tensor.
            tgt_mask (torch.Tensor): Target mask tensor.
            tgt_key_padding_mask (torch.Tensor): Target key padding mask tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        if self.use_cnn:
            x = self.c(en)

        if self.use_mlp:
            if self.input_channels == 1:
                x = self.m(en.reshape([en.shape[0], self.reshape_size, 300]))
            elif self.input_channels == 2:
                x = self.m(en.reshape([en.shape[0], self.reshape_size * 2, 300]))

        x = self.encoder(x)

        tgt = self.embedding(de_1)
        tgt = self.pe(tgt)

        tgt = self.decoder(tgt, x, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)

        pre_type = self.pre_type(tgt)
        return pre_type
