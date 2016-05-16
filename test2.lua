require 'torch'

local input_dims = {}
input_dims[0] = 1 
input_dims[1] = 640
input_dims[2] = 480 

print(torch.zeros(1,unpack(args.input_dims)))