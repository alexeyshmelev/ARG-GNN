"""Tiny CPU equivalence checks; no real data, GPU contexts or scientific launches."""
import copy
import unittest
from unittest.mock import Mock
import torch
from torch.utils.checkpoint import checkpoint
from .activation_checkpoint_v1 import install_layers


class TinyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm=torch.nn.LayerNorm(8,dtype=torch.float64)
        self.linear=torch.nn.Linear(8,8,dtype=torch.float64)
        self.dropout=torch.nn.Dropout(.05)
    def forward(self,value,*,scale=1.):
        return value+self.dropout(torch.nn.functional.gelu(self.linear(self.norm(value))))*scale


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=torch.nn.ModuleList([TinyLayer() for _ in range(3)])
    def forward(self,value):
        for layer in self.layers:value=layer(value,scale=.7)
        return value


def full_loss(value):
    z=torch.nn.functional.normalize(value,dim=-1)
    similarity=z@z.T/.3
    labels=torch.arange(len(z))%3
    eye=torch.eye(len(z),dtype=torch.bool)
    mask=labels[:,None].eq(labels[None,:])&~eye
    log_probability=similarity-torch.logsumexp(similarity.masked_fill(eye,float('-inf')),dim=1,keepdim=True)
    return (-(log_probability*mask).sum(1)/mask.sum(1)).mean()


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.assertFalse(torch.cuda.is_initialized())
        torch.set_num_threads(1)
        torch.manual_seed(17)

    def test_complete_loss_gradients_AdamW_update_and_RNG_are_identical(self):
        plain=TinyEncoder();recomputed=copy.deepcopy(plain)
        install_layers(recomputed.layers,torch,checkpoint)
        x=torch.randn(32,8,dtype=torch.float64)
        results=[]
        for model in (plain,recomputed):
            optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,betas=(.9,.999),eps=1e-6,weight_decay=1e-6)
            torch.manual_seed(36)
            output=model(x)
            loss=full_loss(output)
            after_forward=torch.get_rng_state().clone()
            loss.backward()
            gradients=[parameter.grad.clone() for parameter in model.parameters()]
            after_backward=torch.get_rng_state().clone()
            optimizer.step()
            results.append((output.detach(),loss.detach(),gradients,after_forward,after_backward,
                            [parameter.detach().clone() for parameter in model.parameters()]))
        for index in (0,1,3,4):self.assertTrue(torch.equal(results[0][index],results[1][index]),index)
        for index in (2,5):
            for before,after in zip(results[0][index],results[1][index]):self.assertTrue(torch.equal(before,after),index)
        self.assertFalse(torch.cuda.is_initialized())

    def test_parameter_identity_schema_and_eval_forward_preserved(self):
        model=TinyEncoder()
        identities=[id(parameter) for parameter in model.parameters()]
        keys=list(model.state_dict())
        fake=Mock(side_effect=checkpoint)
        install_layers(model.layers,torch,fake)
        self.assertEqual(identities,[id(parameter) for parameter in model.parameters()])
        self.assertEqual(keys,list(model.state_dict()))
        model.eval()
        with torch.no_grad():model(torch.randn(4,8,dtype=torch.float64))
        fake.assert_not_called()

    def test_non_reentrant_explicit_RNG_policy_and_no_double_install(self):
        layer=TinyLayer();fake=Mock(side_effect=checkpoint)
        rows=install_layers([layer],torch,fake)
        layer(torch.randn(4,8,dtype=torch.float64)).sum().backward()
        self.assertEqual(len(rows),1)
        self.assertFalse(fake.call_args.kwargs['use_reentrant'])
        self.assertTrue(fake.call_args.kwargs['preserve_rng_state'])
        self.assertIsNotNone(layer.linear.weight.grad)
        with self.assertRaisesRegex(ValueError,'exactly once'):install_layers([layer],torch,checkpoint)


if __name__=='__main__':unittest.main()
