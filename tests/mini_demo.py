import torch
from pad2d_op import pad2d,pad2dT
from pad2d_op import Pad2d,Pad2dT

inp = torch.arange(25).reshape(5,5).cuda().float()
padded = pad2d(inp, [2,2,2,2], "symmetric")
print(padded)

pad_mod = Pad2d([2,2,2,2],"symmetric")
padded = pad_mod(inp)
print(padded)

out = torch.ones(81).reshape(9,9).cuda().float()
padT_mod = Pad2dT([2,2,2,2],"symmetric")
inp = padT_mod(out)
print(inp)