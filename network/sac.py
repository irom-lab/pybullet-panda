import torch
from torch import nn
import numpy as np
import copy
from torch.distributions import Normal

from network.conv import ConvNet
from network.mlp import MLP
from network.gru import GRU
from network.lstm import LSTM
from network.util import tie_weights


class Encoder(torch.nn.Module):
    """Conv layers shared by actor and critic in SAC."""
    def __init__(
        self,
        input_n_channel,
        img_sz,
        kernel_sz,
        stride,
        padding,
        n_channel,
        use_sm=True,
        use_spec=False,
        use_bn=False,
        use_residual=False,
        device='cpu',
        verbose=True,
    ):
        super().__init__()
        if verbose:
            print(
                "The neural network for encoder has the architecture as below:"
            )
        self.conv = ConvNet(input_n_channel=input_n_channel,
                            cnn_kernel_size=kernel_sz,
                            cnn_stride=stride,
                            cnn_padding=padding,
                            output_n_channel=n_channel,
                            img_size=img_sz,
                            use_sm=use_sm,
                            use_spec=use_spec,
                            use_bn=use_bn,
                            use_residual=use_residual,
                            verbose=verbose).to(device)

    def forward(self, image, detach=False):
        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        out = self.conv(image)
        if detach:
            out = out.detach()
        return out

    def copy_conv_weights_from(self, source):
        """
        Tie convolutional layers - assume actor and critic have same conv
        structure.
        """
        for source_module, module in zip(source.conv.moduleList,
                                         self.conv.moduleList):
            for source_layer, layer in zip(
                    source_module.children(), module.children(
                    )):  # children() works for both Sequential and nn.Module
                if isinstance(layer, nn.Conv2d):
                    tie_weights(src=source_layer, trg=layer)

    def get_output_dim(self):
        return self.conv.get_output_dim()


class SACPiNetwork(torch.nn.Module):
    def __init__(
        self,
        input_n_channel,
        mlp_dim,
        action_dim,
        action_mag,
        activation_type,  # for MLP; ReLU default for conv
        img_sz,
        kernel_sz,
        stride,
        padding,
        n_channel,
        latent_dim=0,
        append_dim=0,
        rec_type=None,
        rec_hidden_size=0,
        rec_num_layers=1,
        rec_bidirectional=False,
        rec_dropout=0,
        use_sm=True,
        use_ln=True,
        use_bn=False,
        use_residual=False,
        device='cpu',
        verbose=True,
    ):
        super().__init__()
        self.device = device
        self.rec_hidden_size = rec_hidden_size
        if rec_bidirectional:
            self.rec_hidden_batch_dim = 2 * rec_num_layers
        else:
            self.rec_hidden_batch_dim = rec_num_layers
        self.img_sz = img_sz
        if np.isscalar(img_sz):
            self.img_sz = [img_sz, img_sz]

        # Conv layers shared with critic
        self.encoder = Encoder(input_n_channel=input_n_channel,
                               img_sz=img_sz,
                               kernel_sz=kernel_sz,
                               stride=stride,
                               padding=padding,
                               n_channel=n_channel,
                               use_sm=use_sm,
                               use_spec=False,
                               use_bn=use_bn,
                               use_residual=use_residual,
                               device=device,
                               verbose=False)
        if use_sm:
            dim_conv_out = n_channel[-1] * 2  # assume spatial softmax
        else:
            dim_conv_out = self.encoder.get_output_dim()

        # Add recurrent if specified
        if rec_type == 'GRU':
            self.layernorm = nn.LayerNorm(dim_conv_out).to(self.device)
            self.rec = GRU(dim_conv_out + append_dim,
                           rec_hidden_size,
                           device=device,
                           num_layers=rec_num_layers,
                           bidirectional=rec_bidirectional,
                           dropout=rec_dropout)  # output dim is hidden_size
            rec_output_dim = (int(rec_bidirectional) + 1) * rec_hidden_size
            mlp_dim = [rec_output_dim + latent_dim] + mlp_dim + [action_dim]
        elif rec_type == 'LSTM':
            self.layernorm = nn.LayerNorm(dim_conv_out).to(self.device)
            self.rec = LSTM(dim_conv_out + append_dim,
                            rec_hidden_size,
                            device=device,
                            num_layers=rec_num_layers,
                            bidirectional=rec_bidirectional,
                            dropout=rec_dropout)  # output dim is hidden_size
            rec_output_dim = (int(rec_bidirectional) + 1) * rec_hidden_size
            mlp_dim = [rec_output_dim + latent_dim] + mlp_dim + [action_dim]
        else:
            self.rec = None
            mlp_dim = [dim_conv_out + append_dim + latent_dim
                       ] + mlp_dim + [action_dim]

        # Linear layers
        self.mlp = GaussianPolicy(mlp_dim, action_mag, activation_type, use_ln,
                                  device, verbose)

    def forward(
            self,
            image,  # NCHW or LNCHW
            append=None,  # LN x append_dim
            latent=None,  # LN x z_dim
            detach_encoder=False,
            detach_rec=False,
            init_rnn_state=None,  # N x hidden_dim
    ):
        """
        Assume all arguments have the same number of leading dims (L and N),
        and returns the same number of leading dims. init_rnn_state is always
        L=1.
        """
        # Convert to torch
        np_input = False
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).to(self.device)
            np_input = True
        if isinstance(append, np.ndarray):
            append = torch.from_numpy(append).float().to(self.device)

        # Convert [0, 255] to [0, 1]
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        else:
            raise TypeError

        # Get dimensions
        L = 0
        num_extra_dim = 0
        if image.dim() == 3:  # running policy deterministically at test time
            image = image.unsqueeze(0)
            if append is not None:
                append = append.unsqueeze(0)
            num_extra_dim += 1
            N, C, H, W = image.shape
        elif image.dim() == 4:
            N, C, H, W = image.shape
        else:
            L, N, C, H, W = image.shape
            image = image.view(L * N, C, H, W)
        if self.rec is not None and L == 0:
            # recurrent but input does not have L
            if append is not None:
                append = append.unsqueeze(0)
            num_extra_dim += 1
            L = 1
        restore_seq = L > 0

        # # Append latent to image channels
        # if latent is not None:
        #     latent = latent.unsqueeze(-1).unsqueeze(-1)  # make H, W channels
        #     if image.dim() == 4:  # no seq
        #         latent = latent.repeat(1, 1, H, W)
        #     else:  #! assume same latent for seq
        #         latent = latent.repeat(L, 1, 1, H, W)
        #         latent = latent.view(L * N, -1, H, W)
        #     image = torch.cat((image, latent), dim=-3)  # dim=C

        # Forward thru conv
        conv_out = self.encoder.forward(image, detach=detach_encoder)

        # Put dimension back
        if restore_seq:
            conv_out = conv_out.view(L, N, -1)

        # Append, recurrent, latent
        if self.rec is not None:
            conv_out = self.layernorm(conv_out)
            if append is not None:
                conv_out = torch.cat((conv_out, append), dim=-1)
            conv_out, (hn, cn) = self.rec(conv_out, init_rnn_state, detach_rec)
        elif append is not None:
            conv_out = torch.cat((conv_out, append), dim=-1)
        if latent is not None:
            conv_out = torch.cat((conv_out, latent), dim=-1)

        # MLP
        output = self.mlp(conv_out)

        # Restore dimension
        for _ in range(num_extra_dim):
            output = output.squeeze(0)

        # Convert back to np
        if np_input:
            output = output.detach().cpu().numpy()

        if self.rec is not None:
            return output, (hn, cn)
        else:
            return output

    def sample(self,
               image,
               append=None,
               latent=None,
               detach_encoder=False,
               detach_rec=False,
               init_rnn_state=None):
        """
        Assume all arguments have the same number of leading dims (L and N),
        and returns the same number of leading dims. init_rnn_state is always
        L=1. Not converting back to numpy.
        """

        # Convert to torch
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).to(self.device)
        if isinstance(append, np.ndarray):
            append = torch.from_numpy(append).float().to(self.device)

        # Convert [0, 255] to [0, 1]
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        else:
            raise TypeError

        # Get dimensions
        L = 0
        num_extra_dim = 0
        if image.dim() == 3:  # running policy deterministically at test time
            image = image.unsqueeze(0)
            if append is not None:
                append = append.unsqueeze(0)
            num_extra_dim += 1
            N, C, H, W = image.shape
        elif image.dim() == 4:
            N, C, H, W = image.shape
        else:
            L, N, C, H, W = image.shape
            image = image.view(L * N, C, H, W)
        if self.rec is not None and L == 0:
            # recurrent but input does not have L
            if append is not None:
                append = append.unsqueeze(0)
            num_extra_dim += 1
            L = 1
        restore_seq = L > 0

        # Append latent to image channels
        # if latent is not None:
        #     latent = latent.unsqueeze(-1).unsqueeze(-1)  # make H, W channels
        #     if image.dim() == 4:  # no seq
        #         latent = latent.repeat(1, 1, H, W)
        #     else:  #! assume same latent for seq
        #         latent = latent.repeat(L, 1, 1, H, W)
        #         latent = latent.view(L * N, -1, H, W)
        #     image = torch.cat((image, latent), dim=-3)  # dim=C

        # Get CNN output
        conv_out = self.encoder.forward(image, detach=detach_encoder)

        # Put dimension back
        if restore_seq:
            conv_out = conv_out.view(L, N, -1)

        # Append, recurrent, latent
        if self.rec is not None:
            conv_out = self.layernorm(conv_out)
            if append is not None:
                conv_out = torch.cat((conv_out, append), dim=-1)
            conv_out, (hn, cn) = self.rec(conv_out, init_rnn_state, detach_rec)
        elif append is not None:
            conv_out = torch.cat((conv_out, append), dim=-1)
        if latent is not None:
            conv_out = torch.cat((conv_out, latent), dim=-1)

        # MLP
        action, log_prob = self.mlp.sample(conv_out)

        # Restore dimension
        for _ in range(num_extra_dim):
            action = action.squeeze(0)
            log_prob = log_prob.squeeze(0)

        if self.rec is not None:
            return (action, log_prob), (hn, cn)
        else:
            return (action, log_prob)

    def sample_init_rnn_state(self):
        if self.rec is None:
            raise NotImplementedError
        else:
            return (torch.randn(self.rec_hidden_batch_dim, 1,
                                self.rec_hidden_size).to(self.device),
                    torch.randn(self.rec_hidden_batch_dim, 1,
                                self.rec_hidden_size).to(self.device))


class SACTwinnedQNetwork(torch.nn.Module):
    def __init__(
        self,
        input_n_channel,
        mlp_dim,
        action_dim,
        activation_type,  # for MLP; ReLU default for conv
        img_sz,
        kernel_sz,
        stride,
        padding,
        n_channel,
        latent_dim=0,
        append_dim=0,
        rec_type=None,
        rec_hidden_size=0,
        rec_num_layers=1,
        rec_bidirectional=False,
        rec_dropout=0,
        use_sm=True,
        use_ln=True,
        use_bn=False,
        use_residual=False,
        device='cpu',
        verbose=True,
    ):

        super().__init__()
        self.device = device
        self.img_sz = img_sz
        if np.isscalar(img_sz):
            self.img_sz = [img_sz, img_sz]

        # Conv layers shared with critic
        self.encoder = Encoder(input_n_channel=input_n_channel,
                               img_sz=img_sz,
                               kernel_sz=kernel_sz,
                               stride=stride,
                               padding=padding,
                               n_channel=n_channel,
                               use_sm=use_sm,
                               use_spec=False,
                               use_bn=use_bn,
                               use_residual=use_residual,
                               device=device,
                               verbose=False)
        if use_sm:
            dim_conv_out = n_channel[-1] * 2  # assume spatial softmax
        else:
            dim_conv_out = self.encoder.get_output_dim()

        # Add Recurrent if specified
        if rec_type == 'GRU':
            self.layernorm = nn.LayerNorm(dim_conv_out).to(self.device)
            self.rec = GRU(dim_conv_out + append_dim,
                           rec_hidden_size,
                           device=device,
                           num_layers=rec_num_layers,
                           bidirectional=rec_bidirectional,
                           dropout=rec_dropout)  # output dim is hidden_size
            rec_output_dim = (int(rec_bidirectional) + 1) * rec_hidden_size
            mlp_dim = [rec_output_dim + latent_dim + action_dim
                       ] + mlp_dim + [1]
        elif rec_type == 'LSTM':
            self.layernorm = nn.LayerNorm(dim_conv_out).to(self.device)
            self.rec = LSTM(dim_conv_out + append_dim,
                            rec_hidden_size,
                            device=device,
                            num_layers=rec_num_layers,
                            bidirectional=rec_bidirectional,
                            dropout=rec_dropout)  # output dim is hidden_size
            rec_output_dim = (int(rec_bidirectional) + 1) * rec_hidden_size
            mlp_dim = [rec_output_dim + latent_dim + action_dim
                       ] + mlp_dim + [1]
        else:
            self.rec = None
            mlp_dim = [dim_conv_out + latent_dim + append_dim + action_dim
                       ] + mlp_dim + [1]

        self.Q1 = MLP(mlp_dim,
                      activation_type,
                      out_activation_type='Identity',
                      use_ln=use_ln,
                      verbose=False).to(device)
        self.Q2 = copy.deepcopy(self.Q1)
        if verbose:
            print("The MLP for critic has the architecture as below:")
            print(self.Q1.moduleList)

    def forward(self,
                image,
                actions,
                append=None,
                latent=None,
                detach_encoder=False,
                detach_rec=False,
                init_rnn_state=None):
        """
        Assume all arguments have the same number of leading dims (L and N),
        and returns the same number of leading dims. init_rnn_state is always
        L=1.
        """

        # Convert to torch
        np_input = False
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).to(self.device)
            np_input = True
        if isinstance(append, np.ndarray):
            append = torch.from_numpy(append).float().to(self.device)

        # Convert [0, 255] to [0, 1]
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        else:
            raise TypeError

        # Get dimensions
        L = 0
        num_extra_dim = 0
        if image.dim() == 3:  # running policy deterministically at test time
            image = image.unsqueeze(0)
            actions = actions.unsqueeze(0)
            if append is not None:
                append = append.unsqueeze(0)
            num_extra_dim += 1
            N, C, H, W = image.shape
        elif image.dim() == 4:
            N, C, H, W = image.shape
        else:
            L, N, C, H, W = image.shape
            image = image.view(L * N, C, H, W)
        if self.rec is not None and L == 0:
            # recurrent but input does not have L
            if append is not None:
                append = append.unsqueeze(0)
            actions = actions.unsqueeze(0)
            num_extra_dim += 1
            L = 1
        restore_seq = L > 0

        # Append latent to image channels
        # if latent is not None:
        #     latent = latent.unsqueeze(-1).unsqueeze(-1)  # make H, W channels
        #     if image.dim() == 4:  # no seq
        #         latent = latent.repeat(1, 1, H, W)
        #     else:  #! assume same latent for seq
        #         latent = latent.repeat(L, 1, 1, H, W)
        #         latent = latent.view(L * N, -1, H, W)
        #     image = torch.cat((image, latent), dim=-3)  # dim=C

        # Get CNN output
        conv_out = self.encoder.forward(image, detach=detach_encoder)

        # Put dimension back
        if restore_seq:
            conv_out = conv_out.view(L, N, -1)

        # Append, recurrent, latent
        if self.rec is not None:
            conv_out = self.layernorm(conv_out)
            if append is not None:
                conv_out = torch.cat((conv_out, append), dim=-1)
            conv_out, (hn, cn) = self.rec(conv_out, init_rnn_state, detach_rec)
        elif append is not None:
            conv_out = torch.cat((conv_out, append), dim=-1)
        if latent is not None:
            conv_out = torch.cat((conv_out, latent), dim=-1)

        # Append action to mlp
        conv_out = torch.cat((conv_out, actions), dim=-1)

        # MLP
        q1 = self.Q1(conv_out)
        q2 = self.Q2(conv_out)

        # Restore dimension
        for _ in range(num_extra_dim):
            q1 = q1.squeeze(0)
            q2 = q2.squeeze(0)

        # Convert back to np
        if np_input:
            q1 = q1.detach().cpu().numpy()
            q2 = q2.detach().cpu().numpy()

        if self.rec is not None:
            return q1, q2, (hn, cn)
        else:
            return q1, q2


#== Policy (Actor) Model ==
class GaussianPolicy(nn.Module):
    def __init__(self,
                 dimList,
                 action_mag,
                 activation_type='relu',
                 use_ln=True,
                 device='cpu',
                 verbose=True):
        super(GaussianPolicy, self).__init__()
        self.device = device
        self.mean = MLP(dimList,
                        activation_type,
                        out_activation_type='Tanh',
                        use_ln=use_ln,
                        verbose=False).to(device)
        self.log_std = MLP(dimList,
                           activation_type,
                           out_activation_type='Identity',
                           use_ln=use_ln,
                           verbose=False).to(device)
        if verbose:
            print("The MLP for MEAN has the architecture as below:")
            print(self.mean.moduleList)
            # print("The MLP for LOG_STD has the architecture as below:")
            # print(self.log_std.moduleList)

        self.a_max = action_mag
        self.a_min = -action_mag
        self.scale = (self.a_max - self.a_min) / 2.0  # basically the mag
        self.bias = (self.a_max + self.a_min) / 2.0
        self.LOG_STD_MAX = 1
        self.LOG_STD_MIN = -10
        self.eps = 1e-8

    def forward(self, state):  # mean only
        state_tensor = state.to(self.device)
        mean = self.mean(state_tensor)
        return mean * self.scale + self.bias

    def sample(self, state, get_x=False):
        state_tensor = state.to(self.device)
        mean = self.mean(state_tensor)
        log_std = self.log_std(state_tensor)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)

        # Sample
        # print(mean, std)
        normal_rv = Normal(mean, std)
        x = normal_rv.rsample(
        )  # reparameterization trick (mean + std * N(0,1))
        log_prob = normal_rv.log_prob(x)

        # Get action
        y = torch.tanh(x)  # constrain the output to be within [-1, 1]
        action = y * self.scale + self.bias
        # Get the correct probability: x -> a, a = c * y + b, y = tanh x
        # followed by: p(a) = p(x) x |det(da/dx)|^-1
        # log p(a) = log p(x) - log |det(da/dx)|
        # log |det(da/dx)| = sum log (d a_i / d x_i)
        # d a_i / d x_i = c * ( 1 - y_i^2 )
        log_prob -= torch.log(self.scale * (1 - y.pow(2)) + self.eps)
        if log_prob.dim() > 1:
            log_prob = log_prob.sum(log_prob.dim() - 1, keepdim=True)
        else:
            log_prob = log_prob.sum()
        # mean = torch.tanh(mean) * self.scale + self.bias
        if get_x:
            return action, log_prob, x
        return action, log_prob

    def get_pdf(self, state, x):
        # def get_pdf(self, state, action):
        #! either state is a single vector or action is a single vector
        # y = (action - self.bias) / self.scale
        # x = torch.atanh(y)

        state_tensor = state.to(self.device)
        mean = self.mean(state_tensor)
        log_std = self.log_std(state_tensor)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)
        normal_rv = Normal(mean, std)
        log_prob = normal_rv.log_prob(x)

        y = torch.tanh(x)
        log_prob -= torch.log(self.scale * (1 - y.pow(2)) + self.eps)
        if log_prob.dim() > 1:
            log_prob = log_prob.sum(log_prob.dim() - 1, keepdim=True)
        else:
            log_prob = log_prob.sum()
        return log_prob
