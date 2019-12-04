import torch.nn as nn


class NetworkStructure(object):
    # parameter array:
    #   ((ProcessingLayer, params),
    #   (NormalizationLayer, params),
    #   (NonLinear_Layer, params))

    def __init__(self, in_shape, out_shape):
        """
        Sets the in_shape and out_shape for the network.

        :param in_shape:
        :param out_shape:
        """
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.structure_list = []
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

    def set(self, index, layer_type=None, layer_parameters=None):
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

    def to_ordered_dict(self):
        """
        Complies the inner structure to a straight GAN.

        :return:
        """
        pass

