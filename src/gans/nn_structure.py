import torch.nn as nn
import numpy as np
from collections import namedtuple, OrderedDict

#

# optimized for 2D tasks right now
# it seems that inside and outside are equivalent.


#Networks and Layers structures


operational_layers = {nn.Conv2d: {'in_channels': 'int',
                                  'out_channels': 'int',
                                  'kernel_size': 'int',
                                  'stride': 'int',
                                  'padding': 'int'},

                      nn.ConvTranspose2d: {
                          'in_channels': 'int',
                          'out_channels': 'int',
                          'kernel_size': 'int',
                          'stride': 'int',
                          'padding': 'int'},
                      # nn.ReflectionPad2d: {},
                      # nn.ReplicationPad2d: {},
                      # nn.ZeroPad2d: {},
                      # nn.ConstantPad2d: {}
}

Normalization_layers = {nn.BatchNorm2d: {'num_features': 'int'},
                        # nn.GroupNorm: {},
                        # nn.InstanceNorm2d: {},
                        # nn.LayerNorm: {},
}

Non_linear_layers = {nn.ReLU: {},
                     nn.LeakyReLU: {},
                     # nn.Tanh: {},
                     # nn.Sigmoid: {},
                     # nn.HardTanh: {},
                     # nn.Threshold: {},
}

# post_processing_layers = {nn.MaxPool2d: {},
#                           nn.MaxUnpool2d: {},
#                           nn.AvgPool2d: {},
# }
#

class Layer(object):

    def __init__(self,
                 in_shape: int,
                 out_shape: int,
                 latent_shape: int,
                 processing_layer=nn.Identity):

        self.in_shape = in_shape
        self.out_shape = out_shape
        self.latent_maps = latent_shape
        self.processing_layer = processing_layer
        self.processing_layer_params = [in_shape, out_shape]
        self.nonlinear_layer = None
        self.nonlinear_layer_params = []
        self.normalization_layer = None
        self.normalization_layer_params = []
        self.post_processing_layer = None
        self.post_processing_layer_params = []

    def compile(self) -> OrderedDict:
        load_dict = [self.processing_layer(*self.processing_layer_params),]

        if self.normalization_layer is not None:
            load_dict += self.normalization_layer(*self.normalization_layer_params)

        if self.nonlinear_layer is not None:
            load_dict += self.nonlinear_layer(*self.normalization_layer_params)

        if self.post_processing_layer is not None:
            load_dict += self.post_processing_layer(*self.post_processing_layer_params)

        return OrderedDict(load_dict)

    def mutate(self):
        pass


class NetworkStructure(object):

    def __init__(self, in_shape: np.array, out_shape: np.array):
        """
        Sets the in_shape and out_shape for the network.

        :param in_shape:
        :param out_shape:
        """
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.structure = []
        pass

    def add(self, index, layer_type, layer_parameters):
        """
        Inserts a layer of layer_type with parameters layer_parameters in the position indicated
        by the index, then performs the adjustment of parameters of the neighboring layers

        :param index:
        :param layer_type:
        :param layer_parameters:
        :return:
        """
        pass

    def set(self, index, layer_meta_type, layer_type=None, layer_parameters=None):
        """
        Modifies a layer at index to the layer_type and layer_parameters or layer parameter
        modification only, then performs the adjustment of parameters of the neighboring layers

        :param index:
        :param layer_type:
        :param layer_parameters:
        :return:
        """
        pass

    def delete(self, index):
        """
        Deletes the layer at the index and corrects the shapes of neighboring layers.

        :param index:
        :return:
        """
        pass

    def compile(self) -> OrderedDict:
        """
        Complies the inner structure to a straight GAN.

        :return:
        """
        base = list(self.structure[0].compile().items())

        for layer in self.structure[1:]:
            base += list(layer.compile().items())

        return OrderedDict(base)

    def calculate_complexity(self):
        """
        returns the complexity of the underlying model:

        :return:
        """
        pass

    def apply_normalization_policy(self, policy="width_linearization"):
        """
        Performs a normalization of the pipeline to ensure consistent width and absence of
        non-orthodox patterns in the image.

        :param policy: normalization policy, by default "width_linearization", performs width
        normalisation by inserting linear layers between the layers with inconsistent widths.
        :return:
        """
        corrected_structure = [self.structure[0]]
        for layer, next_layer in self.structure[:-1], self.structure[1:]:
            if next_layer.in_shape != layer.out_shape:
                corrected_structure.append(nn.Linear(layer.out_shape, next_layer.in_shape))
            corrected_structure.append(next_layer)
        self.structure = corrected_structure


if __name__ == "__main__":
    input_layer = Layer(16, 40, nn.ConvTranspose2d)
    input_layer.nonlinear_layer = nn.ReLU
    input_layer.normalization_layer = nn.BatchNorm2d(160)

    output_layer = Layer(40, 64*64, nn.ConvTranspose2d)
    output_layer.nonlinear_layer = nn.Tanh

    net_abstract_representation = NetworkStructure(np.array([16, ]), np.array([64, 64]))
    net_abstract_representation.structure = [input_layer, output_layer]

    net_abstract_representation.apply_normalization_policy()
    print(net_abstract_representation.compile())
