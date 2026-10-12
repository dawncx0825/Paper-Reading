"""Real-checkpoint reload verification after the smoke training run."""
import torch

from dpo_repro.core import ROOT, batch, forward_logps, model, read_rows, sft_path, tokenizer, write_json

tok = tokenizer()
rows = read_rows(ROOT / 'data/smoke/sft.jsonl')[:2]
b = batch(rows, tok.pad_token_id)
path = sft_path('smoke')
def inspect():
    m = model(path)
    with torch.inference_mode():
        lp, _ = forward_logps(m, b)
        ids = m.generate(b['input_ids'][:1, :8], attention_mask=b['attention_mask'][:1, :8],
                         max_new_tokens=8, do_sample=False, pad_token_id=tok.pad_token_id)
    result = (lp.cpu(), ids.cpu())
    del m
    torch.cuda.empty_cache()
    return result
first, second = inspect(), inspect()
torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)
assert torch.equal(first[1], second[1])
state = torch.load(path / 'training_state.pt', map_location='cpu', weights_only=False)
assert state['update'] == 2
write_json(ROOT / 'logs/reload_checks.json', dict(profile='smoke', repeated_load_logp_max_error=0,
    greedy_generation_identical=True, checkpoint_updates=state['update'],
    full_interrupted_trajectory_bitwise_equivalence='not claimed'))
print('Real checkpoint reload: PASS', flush=True)
