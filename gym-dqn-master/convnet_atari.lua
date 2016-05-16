require 'nn'

return function(args)
    local net = nn.Sequential()

    net:add(nn.Reshape(unpack(args.input_dims)))

    net:add(nn.SpatialConvolutionMM(4, 64, 8, 8, 4, 4, 1))
    net:add(nn.ReLU())
    net:add(nn.SpatialConvolutionMM(64, 64, 4, 4, 2, 2, 1))
    net:add(nn.ReLU())
    net:add(nn.SpatialConvolutionMM(64, 64, 3, 3, 1, 1, 1))
    net:add(nn.ReLU())

    local nel
    if args.gpu >= 0 then
        nel = net:cuda():forward(torch.zeros(1,unpack(args.input_dims))
                :cuda()):nElement()
    else
        nel = net:forward(torch.zeros(1,unpack(args.input_dims))):nElement()
    end

    net:add(nn.Reshape(nel))

    net:add(nn.Linear(nel, 512))
    net:add(nn.ReLU())

    net:add(nn.Linear(512, args.n_actions))
    if args.gpu >=0 then
        net:cuda()
    end
    return net
end
