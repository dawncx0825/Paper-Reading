import ast
import copy
import math

import pytest
import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel

from dpo_repro.core import ROOT, batch, dpo_loss, logps


def tiny():
    torch.manual_seed(123)
    return GPT2LMHeadModel(GPT2Config(vocab_size=17, n_positions=16, n_embd=16,
        n_layer=1, n_head=2, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)).eval()


def test_completion_mask_keeps_eos_but_not_padding():
    rows = [dict(ids=[2, 3, 4, 0], prompt_len=2), dict(ids=[2, 5, 0], prompt_len=1)]
    b = batch(rows, 0, device='cpu')
    assert b['labels'].tolist() == [[-100, -100, 4, 0], [-100, 5, 0, -100]]
    assert b['attention_mask'].tolist() == [[1, 1, 1, 1], [1, 1, 1, 0]]
    logits = torch.zeros(2, 4, 7, requires_grad=True)
    lp, count = logps(logits, b['labels'])
    torch.testing.assert_close(lp, torch.full((2,), -2 * math.log(7)))
    assert count.tolist() == [2, 2]
    lp.sum().backward()
    assert torch.count_nonzero(logits.grad[0, 0]) == 0  # prompt prediction masked
    assert torch.count_nonzero(logits.grad[0, 2]) > 0  # real EOS prediction retained


def test_dpo_matches_paper_and_gradient_direction():
    c = torch.tensor([-2., -3.], requires_grad=True)
    r = torch.tensor([-4., -1.], requires_grad=True)
    rc, rr = c.detach().clone(), r.detach().clone()
    loss, _ = dpo_loss(c, r, rc, rr, .1)
    torch.testing.assert_close(loss, torch.full((2,), math.log(2)))
    loss.mean().backward()
    assert (c.grad < 0).all() and (r.grad > 0).all()
    swapped, _ = dpo_loss(r.detach(), c.detach(), rr, rc, .1)
    torch.testing.assert_close(swapped, loss.detach())


def test_official_loss_and_logps_parity():
    source = ROOT / 'vendor/official_dpo/trainers.py'
    assert source.exists(), 'Download the pinned official source before validation'
    tree = ast.parse(source.read_text())
    names = {'preference_loss', '_get_batch_logps'}
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    for f in funcs:
        f.returns = None
        for arg in f.args.args:
            arg.annotation = None
    ns = dict(torch=torch, F=F)
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(source), 'exec'), ns)
    torch.manual_seed(9)
    x = [torch.randn(5, requires_grad=True) for _ in range(4)]
    ours, _ = dpo_loss(*x, .5)
    official = ns['preference_loss'](*x, .5)[0]
    torch.testing.assert_close(ours, official)
    ours_grad = torch.autograd.grad(ours.sum(), x, retain_graph=True)
    ref_grad = torch.autograd.grad(official.sum(), x)
    for a, b in zip(ours_grad, ref_grad):
        torch.testing.assert_close(a, b)
    logits = torch.randn(2, 5, 17)
    labels = torch.tensor([[-100, -100, 2, 3, 0], [-100, 5, 0, -100, -100]])
    torch.testing.assert_close(logps(logits, labels)[0], ns['_get_batch_logps'](logits, labels))


def test_accumulation_with_short_tail_matches_full_batch():
    m = tiny()
    other = copy.deepcopy(m)
    rows = [dict(ids=[1, 2, 3, 4], prompt_len=2), dict(ids=[1, 7, 6], prompt_len=1),
            dict(ids=[2, 3, 4, 5, 6], prompt_len=2)]
    b = batch(rows, 0, 'cpu')
    loss = -logps(m(b['input_ids'], attention_mask=b['attention_mask']).logits, b['labels'])[0].mean()
    loss.backward()
    for i in range(0, len(rows), 2):
        b = batch(rows[i:i+2], 0, 'cpu')
        lp = logps(other(b['input_ids'], attention_mask=b['attention_mask']).logits, b['labels'])[0]
        (-lp.sum() / len(rows)).backward()
    for p, q in zip(m.parameters(), other.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=2e-5)


def test_checkpoint_reload_and_optimizer_resume(tmp_path):
    m = tiny()
    opt = torch.optim.RMSprop(m.parameters(), lr=1e-4)
    ids = torch.tensor([[1, 2, 3, 4]])
    def step(m, opt):
        opt.zero_grad()
        loss = m(ids, labels=ids).loss
        loss.backward()
        opt.step()
    step(m, opt)
    m.save_pretrained(tmp_path)
    torch.save(opt.state_dict(), tmp_path / 'optimizer.pt')
    restored = GPT2LMHeadModel.from_pretrained(tmp_path).eval()
    torch.testing.assert_close(m(ids).logits, restored(ids).logits)
    opt2 = torch.optim.RMSprop(restored.parameters(), lr=1e-4)
    opt2.load_state_dict(torch.load(tmp_path / 'optimizer.pt', weights_only=True))
    step(m, opt)
    step(restored, opt2)
    for p, q in zip(m.parameters(), restored.parameters()):
        torch.testing.assert_close(p, q)


def test_online_and_cached_reference_reproduce_loss(tmp_path):
    m = tiny()
    b = batch([dict(ids=[1, 2, 3, 4], prompt_len=2),
               dict(ids=[1, 2, 6], prompt_len=2)], 0, 'cpu')
    with torch.no_grad():
        lp = logps(m(b['input_ids'], attention_mask=b['attention_mask']).logits, b['labels'])[0]
    torch.save(lp, tmp_path / 'ref.pt')
    loaded = torch.load(tmp_path / 'ref.pt', weights_only=True)
    loss, _ = dpo_loss(lp[:1], lp[1:], loaded[:1], loaded[1:], .1)
    torch.testing.assert_close(loss, torch.tensor([math.log(2)]))
