dofile("dqn/Scale.lua")

return function(args)
    return nn.Scale(84, 84, true)
end
