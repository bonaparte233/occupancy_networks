import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist


class CResnetBlockConv1d(nn.Module):
    """Conditional ResNet block for 1D convolution."""
    
    def __init__(self, c_dim, size_in, size_out=None, size_h=None, norm_method='batch_norm'):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in
        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.bn_0 = CBatchNorm1d(c_dim, size_in, norm_method=norm_method)
        self.bn_1 = CBatchNorm1d(c_dim, size_h, norm_method=norm_method)

        self.fc_0 = nn.Conv1d(size_in, size_h, 1)
        self.fc_1 = nn.Conv1d(size_h, size_out, 1)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv1d(size_in, size_out, 1, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x, c):
        net = self.fc_0(self.actvn(self.bn_0(x, c)))
        dx = self.fc_1(self.actvn(self.bn_1(net, c)))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class CBatchNorm1d(nn.Module):
    """Conditional batch normalization for 1D."""
    
    def __init__(self, c_dim, f_dim, norm_method='batch_norm'):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.norm_method = norm_method
        # Submodules
        self.conv_gamma = nn.Conv1d(c_dim, f_dim, 1)
        self.conv_beta = nn.Conv1d(c_dim, f_dim, 1)
        if norm_method == 'batch_norm':
            self.bn = nn.BatchNorm1d(f_dim, affine=False)
        elif norm_method == 'instance_norm':
            self.bn = nn.InstanceNorm1d(f_dim, affine=False)
        elif norm_method == 'group_norm':
            self.bn = nn.GroupNorm(1, f_dim, affine=False)
        else:
            raise ValueError('Invalid normalization method!')
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.conv_gamma.weight)
        nn.init.zeros_(self.conv_beta.weight)
        nn.init.ones_(self.conv_gamma.bias)
        nn.init.zeros_(self.conv_beta.bias)

    def forward(self, x, c):
        assert(x.size(0) == c.size(0))
        assert(c.size(1) == self.c_dim)

        # c is assumed to be of size batch_size x c_dim x T
        if len(c.size()) == 2:
            c = c.unsqueeze(2)

        # Affine mapping
        gamma = self.conv_gamma(c)
        beta = self.conv_beta(c)

        # Batchnorm
        net = self.bn(x)
        out = gamma * net + beta

        return out


class ResnetBlockFC(nn.Module):
    """Fully connected ResNet block."""
    
    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class Decoder(nn.Module):
    """Simple decoder network."""
    
    def __init__(self, dim=3, z_dim=0, c_dim=128, hidden_size=256, n_blocks=5, leaky=False):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim

        # Submodules
        if z_dim != 0:
            self.fc_z = nn.Linear(z_dim, hidden_size)

        self.fc_p = nn.Linear(dim, hidden_size)

        if c_dim != 0:
            self.fc_c = nn.Linear(c_dim, hidden_size)

        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)
        ])

        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc_out = nn.Linear(hidden_size, 1)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, p, z, c, **kwargs):
        batch_size, T, D = p.size()
        net = self.fc_p(p)

        if self.z_dim != 0:
            net_z = self.fc_z(z).unsqueeze(1)
            net = net + net_z

        if self.c_dim != 0:
            net_c = self.fc_c(c).unsqueeze(1)
            net = net + net_c

        for block in self.blocks:
            net = block(net)

        out = self.fc_out(self.actvn(self.bn(net.transpose(1, 2)).transpose(1, 2)))
        out = out.squeeze(-1)

        return out


class DecoderCBatchNorm(nn.Module):
    """Decoder with conditional batch normalization."""
    
    def __init__(self, dim=3, z_dim=0, c_dim=128, hidden_size=256, n_blocks=5, leaky=False):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim

        # Submodules
        if z_dim != 0:
            self.fc_z = nn.Linear(z_dim, c_dim)

        self.conv_p = nn.Conv1d(dim, hidden_size, 1)
        self.blocks = nn.ModuleList([
            CResnetBlockConv1d(c_dim, hidden_size) for i in range(n_blocks)
        ])

        self.bn = CBatchNorm1d(c_dim, hidden_size)
        self.conv_out = nn.Conv1d(hidden_size, 1, 1)
        self.actvn = nn.ReLU()

    def forward(self, p, z, c, **kwargs):
        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        net = self.conv_p(p)

        if self.z_dim != 0:
            c = c + self.fc_z(z)

        for block in self.blocks:
            net = block(net, c)

        out = self.conv_out(self.actvn(self.bn(net, c)))
        out = out.squeeze(1)

        return out


# Decoder dictionary
decoder_dict = {
    'simple': Decoder,
    'cbatchnorm': DecoderCBatchNorm,
}
