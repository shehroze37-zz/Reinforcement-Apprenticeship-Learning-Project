


local class = require 'class'

-- define some dummy A class
local A = class('A')

function A:__init(stuff)

  

  self.stuff = stuff

  

end

function A:run()
  
	
	
	
	input_dims = lua.toTable({ })
	input_dims[0] = 1 
	input_dims[1] = 640
	input_dims[2] = 480 

	print(torch.zeros(1,unpack(args.input_dims)))




end

-- create some instances of both classes
local a = A('hello world from A')

-- run stuff
a:run()
