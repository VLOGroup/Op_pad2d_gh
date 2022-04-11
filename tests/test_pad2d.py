import os,sys
from os.path import join
import tempfile
from typing import Tuple
import torch
import numpy as np

from parameterized import parameterized, parameterized_class
import unittest

from typing import List, Tuple

import pad2d_op

class DummyJitableModel(torch.nn.Module):
    ''' Dummy Model for testing jit script export'''
    def __init__(self,  padding:List[int], 
                        mode:str="symmetric",):
        super(DummyJitableModel,self).__init__()
        self.pad2d  = pad2d_op.Pad2d(padding, mode)
        self.pad2dT = pad2d_op.Pad2dT(padding, mode)
    def forward(self, x):
        y1 = self.pad2d(x)
        y1 = y1 + 1
        z1 = self.pad2dT(y1)
        z1 = z1 +1
        return z1


def get_fun_or_object(fun_type, padding, mode):
    if fun_type == "pad2d":
        return lambda x : pad2d_op.pad2d(x, padding, mode)
    elif fun_type == "Pad2d":
        return pad2d_op.Pad2d(padding, mode)

    elif fun_type == "pad2dT":
        return lambda x : pad2d_op.pad2dT(x, padding, mode)
    elif fun_type == "Pad2dT":
        return pad2d_op.Pad2dT(padding, mode)
    else:
        raise RuntimeError("Unkonwn funciton tpye: {fun_type}")


class Testpad2DTranspose_Op(unittest.TestCase):
    @parameterized.expand([
    #  ( # Op,                  pad    padT        padding   ,  mode    ),
    #                                             x0 x1 y1 y2 
       ("FunPadOp_symmetric", "pad2d", "pad2dT", (2, 3, 5, 7), 'symmetric', ),
       ("FunPadOp_replicate", "pad2d", "pad2dT", (2, 3, 5, 7), 'replicate', ),
       ("FunPadOp_reflect",   "pad2d", "pad2dT", (2, 3, 5, 7), 'reflect',   ),
       ("FunPadOp_symmetric", "pad2d", "pad2dT", (1, 1, 1, 1), 'symmetric', ),
       ("FunPadOp_replicate", "pad2d", "pad2dT", (1, 1, 1, 1), 'replicate', ),
       ("FunPadOp_reflect",   "pad2d", "pad2dT", (1, 1, 1, 1), 'reflect',   ),
       # Testing Modules Interfaces:
       ("ModPadOp_symmetric", "Pad2d", "Pad2dT", (2, 3, 5, 1), 'symmetric', ),
       ("ModPadOp_replicate", "Pad2d", "Pad2dT", (2, 3, 5, 1), 'replicate', ),
       ("ModPadOp_reflect",   "Pad2d", "Pad2dT", (2, 3, 5, 1), 'reflect',   ),
       ("ModPadOp_symmetric", "Pad2d", "Pad2dT", (1, 1, 1 ,1), 'symmetric', ),
       ("ModPadOp_replicate", "Pad2d", "Pad2dT", (1, 1, 1 ,1), 'replicate', ),
       ("ModPadOp_reflect",   "Pad2d", "Pad2dT", (1, 1, 1 ,1), 'reflect',   ),
    ])    
    def test_padding_Adjointness(self, name, pad2d_str, pad2dT_str, padding, mode,):
        print(f"Adjointness Testing on: {name} ->", end="")
        if len(padding) != 4: raise ValueError(f"padding should be 4 integer values! {padding}")
        N,C,H,W = 3, 2, 128, 512   # Use realistic dimension so that errors can actuall sum up
        # Computing pseudo input
        Wp = W + padding[0] + padding[1]
        Hp = H + padding[2] + padding[3]
        x = torch.rand( [N*C,H, W ]).double().cuda()
        y = torch.rand( [N*C,Hp,Wp]).double().cuda()

        # Short hands for Operator and its Transpose
        # pad  = lambda x: pad2d_op.pad2d (x, padding, mode)
        # padT = lambda y: pad2d_op.pad2dT(y, padding, mode)
        pad  = get_fun_or_object(pad2d_str,  padding, mode)
        padT = get_fun_or_object(pad2dT_str, padding, mode)

        # Performing Adjointess Test
        # < Op(x), y > == < x, OpT(y) > 
        sp1 = torch.sum(pad(x)*      y )
        sp2 = torch.sum(    x * padT(y))
        is_adjoint = torch.allclose(sp1, sp2)

        self.assertTrue(is_adjoint, msg=f"pad2DTranspose_Op did not pass adjointment test {sp1}!={sp2} ")

        print( " ok" if is_adjoint else "fail")



    @parameterized.expand([
    #  ( # Op,                  pad    padT        padding   ,  mode    ),
    #                                             x0 x1 y1 y2 
       # Testing Function Interfaces:
       ("FunPadOp_symmetric", "pad2d", "pad2dT", (2, 3, 5, 7), 'symmetric', ),
       ("FunPadOp_replicate", "pad2d", "pad2dT", (2, 3, 5, 7), 'replicate', ),
       ("FunPadOp_reflect",   "pad2d", "pad2dT", (2, 3, 5, 7), 'reflect',   ),
       ("FunPadOp_symmetric", "pad2d", "pad2dT", (1, 1, 1, 1), 'symmetric', ),
       ("FunPadOp_replicate", "pad2d", "pad2dT", (1, 1, 1, 1), 'replicate', ),
       ("FunPadOp_reflect",   "pad2d", "pad2dT", (1, 1, 1, 1), 'reflect',   ),
       # Testing Modules Interfaces:
       ("ModPadOp_symmetric", "Pad2d", "Pad2dT", (2, 3, 5, 1), 'symmetric', ),
       ("ModPadOp_replicate", "Pad2d", "Pad2dT", (2, 3, 5, 1), 'replicate', ),
       ("ModPadOp_reflect",   "Pad2d", "Pad2dT", (2, 3, 5, 1), 'reflect',   ),
       ("ModPadOp_symmetric", "Pad2d", "Pad2dT", (1, 1, 1 ,1), 'symmetric', ),
       ("ModPadOp_replicate", "Pad2d", "Pad2dT", (1, 1, 1 ,1), 'replicate', ),
       ("ModPadOp_reflect",   "Pad2d", "Pad2dT", (1, 1, 1 ,1), 'reflect',   ),
    ])    
    def test_gradient(self, name, pad2d_str, pad2dT_str, padding, mode,):
        print(f"Grad Testing on: {name} ->", end="")
        if len(padding) != 4: raise ValueError(f"padding should be 4 integer values! {padding}")
        N,C,H,W = 3, 2, 128, 512   # Use realistic dimension so that errors can actuall sum up
        # Computing pseudo input
        Wp = W + padding[0] + padding[1]
        Hp = H + padding[2] + padding[3]
        x = torch.rand( [N,C,H, W ]).double().cuda()
        y = torch.ones( [N,C,Hp,Wp]).double().cuda()

        # Short hands for Operator and its Transpose
        fun  = get_fun_or_object(pad2d_str,  padding, mode)
        def compute_loss(scale, y=y):
            return torch.sum(y*fun(x*scale))

        scale = 1.

        # compute the gradient using the custom gradient implementation
        # y = 1 @ K ( a * x + b)
        #  => dy/da = x @ K^T @ 1
        x = torch.autograd.Variable(x, requires_grad=True)
        loss = compute_loss(scale)
        loss.backward()
        self.assertTrue(x.grad is not None, msg=f"Gradient Test for '{name}' Failed! No gradient passed through operator")
        grad_scale = torch.sum(x * x.grad)
        x.grad.zero_()
        

        # check it numerically
        epsilon = 1e-4
        with torch.no_grad():
            l_p = compute_loss(scale+epsilon)
            l_n = compute_loss(scale-epsilon)
            grad_scale_num = (l_p - l_n) / (2 * epsilon)
        is_gradient_test_pass = torch.allclose(grad_scale, grad_scale_num)
        self.assertTrue(is_gradient_test_pass, msg=f"pad2DT did not pass adjointment test {grad_scale}!={grad_scale_num} ")
        print( " ok" if is_gradient_test_pass else "fail")


    @parameterized.expand([
    #  ( # Op,                padding   ,  mode     ),
    #                       x0 x1 y1 y2 
    ("FunPadOp_symmetric", (2, 3, 5, 7), 'symmetric', ),
    ("FunPadOp_replicate", (2, 3, 5, 7), 'replicate', ),
    ("FunPadOp_reflect",   (2, 3, 5, 7), 'reflect',   ),
    ])    
    def test_torchscript_trace_export(self, name, padding, mode):
        print("Saving Model via Jit Trace ->", end=""
        )
        data_in = torch.randn([2,3,16,32]).cuda()
        mod = DummyJitableModel( padding, mode)
            
        try:
            jit_trace = torch.jit.trace(mod, data_in) # Jit Trace is more restrictive
            # print("Traced TorchScript Graph")
            # print(jit_trace.graph)
            with tempfile.TemporaryDirectory() as tmp:
                tmp_file = join(tmp,"pytorch_test_module_saving.pt")
                jit_trace.save(tmp_file)
                self.assertTrue(os.path.isfile(tmp_file), f"Test failed because binary model was not saved!")
                os.remove(tmp_file)
        except:
            self.assertTrue(False, f"Test Failed because an assertion occured")
            raise 
        print(" ok")

    @parameterized.expand([
    #  ( # Op,                padding   ,  mode     , value ),
    #                       x0 x1 y1 y2 
    ("FunPadOp_symmetric", (2, 3, 5, 7), 'symmetric', ),
    ("FunPadOp_replicate", (2, 3, 5, 7), 'replicate', ),
    ("FunPadOp_reflect",   (2, 3, 5, 7), 'reflect',   ),
    ])    
    def test_torchscript_script_export(self, name, padding, mode):
        print("Saving Model via Jit Script ->", end="")
        data_in = torch.randn([2,3,16,32]).cuda()
        mod = DummyJitableModel( padding, mode)
            
        try:
            jit_module = torch.jit.script(mod)
            with tempfile.TemporaryDirectory() as tmp:
                tmp_file = join(tmp,"pytorch_test_module_saving.pt")
                jit_module.save(tmp_file)
                self.assertTrue(os.path.isfile(tmp_file), f"Test failed because binary model was not saved!")
                os.remove(tmp_file)
        except:
            self.assertTrue(False, f"Test Failed because an assertion occured")
            raise
        print(" ok")

if __name__ == '__main__':
    unittest.main()
    print("All Tests for Module pad2d_op done")