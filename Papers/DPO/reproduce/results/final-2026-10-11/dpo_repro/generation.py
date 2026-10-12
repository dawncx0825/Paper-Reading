import torch

from .core import batch, forward_logps


@torch.inference_mode()
def generate(m, tok, prompts, count, max_new, batch_size, seed):
    expanded = [(p, k) for p in prompts for k in range(count)]
    was_training = m.training
    checkpointing = m.is_gradient_checkpointing
    if checkpointing:
        m.gradient_checkpointing_disable()
    m.eval()
    try:
        for start in range(0, len(expanded), batch_size):
            subset = expanded[start:start + batch_size]
            torch.manual_seed(seed + start)
            torch.cuda.manual_seed_all(seed + start)
            width = max(len(p['prompt_ids']) for p, _ in subset)
            ids = torch.full((len(subset), width), tok.pad_token_id, dtype=torch.long, device='cuda')
            mask = torch.zeros_like(ids)
            for i, (p, _) in enumerate(subset):
                n = len(p['prompt_ids'])
                ids[i, -n:] = torch.tensor(p['prompt_ids'], device='cuda')
                mask[i, -n:] = 1
            output = m.generate(input_ids=ids, attention_mask=mask, do_sample=True,
                temperature=1.0, top_k=0, top_p=1.0, max_new_tokens=max_new,
                pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                use_cache=True, repetition_penalty=1.0)
            rows = []
            for (p, k), y in zip(subset, output[:, width:].tolist()):
                if tok.eos_token_id in y:
                    y = y[:y.index(tok.eos_token_id) + 1]
                rows.append(dict(id=f'{p["id"]}:{k}', prompt_id=p['id'], review_id=p['review_id'],
                    prompt_ids=p['prompt_ids'], completion_ids=y, prompt=p['prompt'],
                    completion=tok.decode(y, skip_special_tokens=True),
                    text=tok.decode(p['prompt_ids'] + y, skip_special_tokens=True),
                    stopped_eos=y[-1] == tok.eos_token_id))
            # Score exact generated tokens with the same FP32 compute policy.
            b = batch([dict(ids=r['prompt_ids'] + r['completion_ids'], prompt_len=len(r['prompt_ids']))
                       for r in rows], tok.pad_token_id)
            lp, _ = forward_logps(m, b)
            for r, x in zip(rows, lp.tolist()):
                r['policy_logp'] = x
            yield rows
    finally:
        m.train(was_training)
        if checkpointing:
            m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
