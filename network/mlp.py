from torch import nn
from torch.nn.utils import spectral_norm
from collections import OrderedDict
import logging


activation_dict = nn.ModuleDict({
    "relu": nn.ReLU(),
    "elu": nn.ELU(),
    "tanh": nn.Tanh(),
    "identity": nn.Identity()
})


class MLP(nn.Module):
    """
    Construct a fully-connected neural network with flexible depth, width and
    activation function choices.
    """
    def __init__(self,
                 layer_size,
                 activation_type='relu',
                 out_activation_type='identity',
                 use_ln=False,
                 use_spec=False,
                 use_bn=False,
                 verbose=True):
        """
        __init__: Initalizes.

        Args:
            layer_size (int List): the dimension of each layer.
            activation_type (str, optional): type of activation layer. Support
                'Sin', 'Tanh' and 'ReLU'. Defaults to 'Tanh'.
            verbose (bool, optional): print info or not. Defaults to False.
        """
        super(MLP, self).__init__()

        # Construct module list: if use `Python List`, the modules are not
        # added to computation graph. Instead, we should use `nn.ModuleList()`.
        self.moduleList = nn.ModuleList()
        numLayer = len(layer_size) - 1
        for idx in range(numLayer):
            i_dim = layer_size[idx]
            o_dim = layer_size[idx + 1]

            # self.moduleList.append(
            #     nn.Linear(in_features=i_dim, out_features=o_dim)
            # )
            linear_layer = nn.Linear(i_dim, o_dim)
            if use_spec:
                linear_layer = spectral_norm(linear_layer)
            # final - could also be the first layer
            if idx == numLayer - 1:
                if use_ln:
                    module = nn.Sequential(
                        OrderedDict([
                            ('linear_1', linear_layer),
                            ('norm_1', nn.LayerNorm(o_dim)),
                        ]))
                # elif use_bn:
                #     module = nn.Sequential(
                #         OrderedDict([
                #             ('linear_1', linear_layer),
                #             ('norm_1', nn.BatchNorm1d(o_dim)),
                #             ('act_1', activation_dict[out_activation_type]),
                #         ]))
                else:
                    module = nn.Sequential(
                        OrderedDict([
                            ('linear_1', linear_layer),
                            ('act_1', activation_dict[out_activation_type]),
                        ]))
            # first layer
            elif idx == 0:
                if use_ln:
                    module = nn.Sequential(
                        OrderedDict([
                            ('linear_1', linear_layer),
                            ('norm_1', nn.LayerNorm(o_dim)),
                        ]))
                elif use_bn:
                    module = nn.Sequential(
                        OrderedDict([
                            ('linear_1', linear_layer),
                            ('norm_1', nn.BatchNorm1d(o_dim)),
                            ('act_1', activation_dict[activation_type]),
                        ]))
                else:
                    module = nn.Sequential(
                        OrderedDict([
                            ('linear_1', linear_layer),
                            ('act_1', activation_dict[activation_type]),
                        ]))
            # hidden layers
            else:
                if use_ln:
                    module = nn.Sequential(
                        OrderedDict([
                            ('linear_1', linear_layer),
                            ('norm_1', nn.LayerNorm(o_dim)),
                            ('act_1', activation_dict[activation_type]),
                        ]))
                elif use_bn:
                    module = nn.Sequential(
                        OrderedDict([
                            ('linear_1', linear_layer),
                            ('norm_1', nn.BatchNorm1d(o_dim)),
                            ('act_1', activation_dict[activation_type]),
                        ]))
                else:
                    module = nn.Sequential(
                        OrderedDict([
                            ('linear_1', linear_layer),
                            ('act_1', activation_dict[activation_type]),
                        ]))

            self.moduleList.append(module)
        if verbose:
            logging.info(self.moduleList)


    def forward(self, x):
        for m in self.moduleList:
            x = m(x)
        return x
