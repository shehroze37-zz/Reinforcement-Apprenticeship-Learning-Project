require 'nn'


return function(args)

    print('Creating convolutional neural network')


    

    args.n_units        = {32, 64, 64}
    args.filter_size    = {8, 4, 3}
    args.filter_stride  = {4, 2, 1}
    args.n_hid          = {512}
    args.nl             = nn.ReLU

    


    local net = nn.Sequential()

    net:add(nn.Reshape(unpack(args.input_dims)))

    --- first convolutional layer
    local convLayer = nn.SpatialConvolution

    if args.gpu >= 0 then
        net:add(nn.Transpose({1,2},{2,3},{3,4}))
        convLayer = nn.SpatialConvolutionCUDA
    end

    net:add(convLayer(args.hist_len*args.ncols, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],1))
    net:add(args.nl())
    net:add(nn.SpatialMaxPooling(2,2,2,2))

    -- Add convolutional layers
    for i=1,(#args.n_units-1) do
        -- second convolutional layer
        net:add(convLayer(args.n_units[i], args.n_units[i+1],
                            args.filter_size[i+1], args.filter_size[i+1],
                            args.filter_stride[i+1], args.filter_stride[i+1]))
        net:add(args.nl())
        net:add(nn.SpatialMaxPooling(2,2,2,2))
    end



    local input_channels = args.ncols * args.hist_len

    local nel
    if args.gpu >= 0 then
        net:add(nn.Transpose({4,3},{3,2},{2,1}))
        nel = net:cuda():forward(torch.zeros(input_channels ,unpack(args.input_dims))
                :cuda()):nElement()
    else
        nel = net:forward(torch.zeros(input_channels,unpack(args.input_dims))):nElement()
    end




    -- reshape all feature planes into a vector per example
    net:add(nn.Reshape(nel))

    -- fully connected layer
    net:add(nn.Linear(nel, args.n_hid[1]))
    net:add(args.nl())
    local last_layer_size = args.n_hid[1]

    for i=1,(#args.n_hid-1) do
        -- add Linear layer
        last_layer_size = args.n_hid[i+1]
        net:add(nn.Linear(args.n_hid[i], last_layer_size))
        net:add(args.nl())
    end

    -- add the last fully connected layer (to actions)
    net:add(nn.Linear(last_layer_size, args.n_actions))

    if args.gpu >=0 then
        net:cuda()
    end
    if args.verbose >= 2 then
        print(net)
        print('Convolutional layers flattened output size:', nel)
    end
    return net


    --[[net:add(nn.Reshape(unpack(args.input_dims)))


    print(torch.zeros(1,unpack(args.input_dims))   )


    net:add(nn.SpatialConvolutionMM(4, 64, 8, 8, 4, 4, 1))
    net:add(nn.ReLU())
    net:add(nn.SpatialConvolutionMM(64, 64, 4, 4, 2, 2, 1))
    net:add(nn.ReLU())
    net:add(nn.SpatialConvolutionMM(64, 64, 3, 3, 1, 1, 1))
    net:add(nn.ReLU())

    local nel


    print(net)


    nel = net:forward(torch.zeros(1,unpack(args.input_dims))):nElement()

    if args.gpu >= 0 then
        nel = net:cuda():forward(torch.zeros(1,unpack(args.input_dims)):cuda()):nElement()
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
    return net]]
end
