import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


class PolicyNetwork(nn.Module):
    def __init__(self, input_shape, num_layers, num_filters):
        super(PolicyNetwork, self).__init__()
        self.input_shape = input_shape
        self.num_layers = num_layers
        self.num_filters = num_filters

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        self.build_module()

    def build_module(self):
        """
        Builds network whilst automatically inferring shapes of layers.
        """
        self.layer_dict = nn.ModuleDict()
        # initialize a module dict, which is effectively a dictionary that can collect layers and integrate them into pytorch
        print(
            "Building basic block of ConvolutionalNetwork using input shape",
            self.input_shape,
        )
        x = torch.zeros(
            (self.input_shape)
        )  # create dummy inputs to be used to infer shapes of layers

        out = x
        self.layer_dict["input_FCC"] = FCCNetwork(
            input_shape=out.shape,
            num_layers=self.num_layers,
            num_filters=self.num_filters,
        )
        out = self.layer_dict["input_FCC"].forward(out)

        self.layer_dict["action"] = nn.Linear(
            in_features=out.shape,
            out_features=5,
            bias=True,
        )
        outact = self.layer_dict["action"].forward(out)

        self.layer_dict["comm"] = nn.Linear(
            in_features=out.shape,
            out_features=10,
            bias=True,
        )
        outcom = self.layer_dict["comm"].forward(out)

        self.layer_dict["critic"] = nn.Linear(
            in_features=out.shape,
            out_features=1,
            bias=True,
        )
        val = self.layer_dict["critic"].forward(out)

        self.softmax = nn.Softmax(dim=1)

        return outact, outcom, val

    def forward(self, x):
        out = x
        out = self.layer_dict["input_FCC"].forward(out)

        move = self.softmax(self.layer_dict["action"].forward(out))
        communicate = self.softmax(self.layer_dict["comm"].forward(out))
        value = self.layer_dict["critic"].forward(out)

        return move, communicate, value


class FCCNetwork(nn.Module):
    def __init__(self, input_shape, num_layers, num_filters):
        super(FCCNetwork, self).__init__()
        # set up class attributes useful in building the network and inference
        self.input_shape = input_shape
        self.num_layers = num_layers
        self.num_filters = num_filters

        # initialize a module dict, which is effectively a dictionary that can collect layers and integrate them into pytorch
        self.layer_dict = nn.ModuleDict()
        # build the network
        self.build_module()

    def build_module(self):
        print("Building basic block of FCCNetwork using input shape", self.input_shape)
        x = torch.zeros((self.input_shape))

        out = x
        out = out.view(out.shape[0], -1)

        for i in range(self.num_layers):
            self.layer_dict["fcc_{}".format(i)] = nn.Linear(
                in_features=out.shape[1],  # initialize a fcc layer
                out_features=self.num_filters,
                bias=True,
            )
            # apply ith fcc layer to the previous layers outputs
            out = self.layer_dict["fcc_{}".format(i)](out)
            out = F.relu(out)  # apply a ReLU on the outputs

        print("Block is built, output volume is", out.shape)
        return out

    def forward(self, x):
        """
        Forward prop data through the network and return the preds
        :param x: Input batch x a batch of shape batch number of samples, each of any dimensionality.
        :return: preds of shape (b, num_classes)
        """
        out = x
        out = out.view(out.shape[0], -1)

        for i in range(self.num_layers):
            out = self.layer_dict["fcc_{}".format(i)](out)
            # apply ith fcc layer to the previous layers outputs
            out = F.relu(out)  # apply a ReLU on the outputs

        return out

    def reset_parameters(self):
        """
        Re-initializes the networks parameters
        """
        for item in self.layer_dict.children():
            item.reset_parameters()

        self.logits_linear_layer.reset_parameters()
