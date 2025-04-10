import torch
import torch.nn as nn

import dhg
from dhg.nn import MLP
from dhg.nn import GCNConv, GATConv
from dhg.nn import HGNNConv
from kan_layer import NaiveFourierKANLayer as KANLayer
from torch import spmm
from dhg.models import GAT


class MyGCN(nn.Module):
    r"""The GCN model proposed in `Semi-Supervised Classification with Graph Convolutional Networks <https://arxiv.org/pdf/1609.02907>`_ paper (ICLR 2017).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``use_bn`` (``bool``): If set to ``True``, use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``): Dropout ratio. Defaults to ``0.5``.
    """
    def __init__(self, in_channels: int,
                 hid_channels: int,
                 num_classes: int,
                 use_bn: bool = False,
                 drop_rate: float = 0.5) -> None:
        super().__init__()
        self.layers0 = GCNConv(in_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate)
        self.layers1 = GCNConv(hid_channels, num_classes, use_bn=use_bn, is_last=True)

    def forward(self, X: torch.Tensor, g: "dhg.Graph", get_emb=False) -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``g`` (``dhg.Graph``): The graph structure that contains :math:`N` vertices.
        """        
        emb = self.layers0(X, g)
        X = self.layers1(emb, g)
        if get_emb:
            return emb
        else:
            return X


class MyGAT(nn.Module):
    r"""The GAT convolution layer proposed in `Graph Attention Networks <https://arxiv.org/pdf/1710.10903>`_ paper (ICLR 2018).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``out_channels`` (``int``): :math:`C_{out}` is the number of output channels.
        ``bias`` (``bool``): If set to ``False``, the layer will not learn the bias parameter. Defaults to ``True``.
        ``use_bn`` (``bool``): If set to ``True``, the layer will use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``): The dropout probability. If ``dropout <= 0``, the layer will not drop values. Defaults to ``0.5``.
        ``atten_neg_slope`` (``float``): Hyper-parameter of the ``LeakyReLU`` activation of edge attention. Defaults to ``0.2``.
        ``is_last`` (``bool``): If set to ``True``, the layer will not apply the final activation and dropout functions. Defaults to ``False``.
    """
    def __init__(self, in_channels: int,
                 hid_channels: int,
                 num_classes: int,
                 num_heads: int = 1,
                 use_bn: bool = False,
                 drop_rate: float = 0.5) -> None:
        super().__init__()
        self.layers0 = GAT(in_channels, hid_channels, num_heads=num_heads, use_bn=use_bn, drop_rate=drop_rate)
        self.layers1 = GAT(hid_channels, num_classes, num_heads=num_heads, use_bn=use_bn, is_last=True)

    def forward(self, X: torch.Tensor, g: "dhg.Graph", get_emb=False) -> torch.Tensor:
        r""" The forward function.

        Args:
            X (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N_v, C_{in})`.
            g (``dhg.Graph``): The graph structure that contains :math:`N_v` vertices.
        """      
        emb = self.layers0(X, g)
        X = self.layers1(emb, g)
        if get_emb:
            return emb
        else:
            return X
        

class MyHGNN(nn.Module):
    r"""The HGNN model proposed in `Hypergraph Neural Networks <https://arxiv.org/pdf/1809.09401>`_ paper (AAAI 2019).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``use_bn`` (``bool``): If set to ``True``, use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``, optional): Dropout ratio. Defaults to 0.5.
    """

    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        num_classes: int,
        use_bn: bool = False,
        drop_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.layers0 = HGNNConv(in_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate)
        # self.layers1 = HGNNConv(hid_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate)
        self.layers1 = HGNNConv(hid_channels, num_classes, use_bn=use_bn, is_last=True)


    def forward(self, X: torch.Tensor, hg: "dhg.Hypergraph", get_emb=False) -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``hg`` (``dhg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
        """
        emb1 = self.layers0(X, hg)
        # emb1 = self.layers1(emb, hg)
        X = self.layers1(emb1, hg)

        if get_emb:
            return emb1
        else:
            return X
        # return X


class MyMLPs(nn.Module):
    def __init__(self, dim_in, dim_hid, n_classes) -> None:
        super().__init__()
        self.layer0 = MLP([dim_in, dim_hid])
        self.layer1 = nn.Linear(dim_hid, n_classes)
    
    def forward(self, X, get_emb=False):
        emb = self.layer0(X)
        X = self.layer1(emb)
        if get_emb:
            return emb
        else:
            return X


# from kan import MultKAN
from eff_kan import KAN as MultKAN
from eff_kan import KANLinear
class MyKAN(nn.Module):
    def __init__(self, dim_in, dim_hid, n_classes) -> None:
        super().__init__()
        # self.layer0 = FC_KAN([dim_in, dim_hid, n_classes], func_list=['dog','bs'])
        self.layer0 = MultKAN([dim_in, dim_hid, n_classes])
        # self.layer0 = FastKAN([dim_in, dim_hid, n_classes])
        # self.layer1 = KANLayer(dim_hid, n_classes)

    def forward(self, X, get_emb=False):
        # emb = self.layer0(X)
        # X = self.layer1(emb)
        # if get_emb:
        #     return emb
        # else:
        #     return X
        return self.layer0(X)

