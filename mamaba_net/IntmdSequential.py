import torch.nn as nn


# class IntermediateSequential(nn.Sequential):
#     def __init__(self, *args, return_intermediate=True):
#         super().__init__(*args)
#         self.return_intermediate = return_intermediate
#
#     def forward(self, input):
#         if not self.return_intermediate:
#             return super().forward(input)
#
#         intermediate_outputs = {}
#         output = input#到底是input是啥
#         for name, module in self.named_children():
#             output = intermediate_outputs[name] = module(output)
#
#         return output, intermediate_outputs
# class IntermediateSequential(nn.Sequential):
#     def __init__(self, *args, return_intermediate=True):
#         super().__init__(*args)
#         self.return_intermediate = return_intermediate
#
#     def forward(self, input):
#         intermediate_outputs = {}
#         output = input
#         for name, module in self.named_children():
#             output = module(output)
#             intermediate_outputs[name] = output  # Store output before moving to the next layer
#
#         return intermediate_outputs  # Only return the dictionary of intermediate outputs
# from collections import OrderedDict
#
# class IntermediateSequential(nn.Sequential):
#     def __init__(self, *args, return_intermediate=True):
#         super().__init__(*args)
#         self.return_intermediate = return_intermediate
#
#     def forward(self, input):
#         intermediate_outputs = OrderedDict()  # Use OrderedDict to preserve order
#         output = input
#         for name, module in self.named_children():
#             output = module(output)
#             intermediate_outputs[name] = output
#
#         return OrderedDict(list(intermediate_outputs.items())[-4:])  # Return the last four
class IntermediateSequential(nn.Sequential):
    def __init__(self, *args, return_intermediate=True):
        super().__init__(*args)
        self.return_intermediate = return_intermediate

    def forward(self, input):
        intermediate_outputs = []
        output = input
        for name, module in self.named_children():
            output = module(output)
            intermediate_outputs.append(output)

        return intermediate_outputs[-4:]