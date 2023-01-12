import torch
import logging

from .conv import ConvNet
from .mlp import MLP


class Encoder(torch.nn.Module):
    """Conv layers shared by actor and critic in SAC."""
    def __init__(
        self,
        in_channels,
        img_sz,
        kernel_sz,
        stride,
        padding,
        n_channel,
        mlp_hidden_dim,
        mlp_output_dim,
        use_sm=False,
        use_spec=False,
        use_bn_conv=False,
        use_bn_mlp=False,
        use_ln_mlp=False,
        device='cpu',
        verbose=True,
    ):
        super().__init__()
        self.conv = ConvNet(input_n_channel=in_channels,
                            cnn_kernel_size=kernel_sz,
                            cnn_stride=stride,
                            cnn_padding=padding,
                            output_n_channel=n_channel,
                            img_size=img_sz,
                            use_sm=use_sm,
                            use_spec=use_spec,
                            use_bn=use_bn_conv,
                            verbose=verbose).to(device)

        mlp_dim_list = [self.conv.get_output_dim(), *mlp_hidden_dim, mlp_output_dim]
        self.mlp = MLP(mlp_dim_list,
                       activation_type='relu',
                       out_activation_type='identity',
                       use_spec=False,
                       use_bn=use_bn_mlp,
                       use_ln=use_ln_mlp,
                       verbose=verbose).to(device)


    def forward(self, image, detach=False):
        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        conv_out = self.conv(image)
        out = self.mlp(conv_out)
        if detach:
            out = out.detach()
        return out
