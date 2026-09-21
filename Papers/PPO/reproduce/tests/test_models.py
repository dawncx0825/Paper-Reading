from types import SimpleNamespace

import torch

from ppo_repro.models import align_model_and_tokenizer, apply_reward_center


class DummyReward(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.score = torch.nn.Linear(2, 1, bias=False)
        torch.nn.init.ones_(self.score.weight)


def test_reward_center_hook_preserves_state_keys_and_gradient():
    model = DummyReward()
    keys_before = set(model.state_dict())
    handle = apply_reward_center(model, 1.5)
    value = model.score(torch.tensor([[1.0, 2.0]], requires_grad=True))
    assert torch.allclose(value, torch.tensor([[1.5]]))
    value.sum().backward()
    assert model.score.weight.grad is not None
    assert set(model.state_dict()) == keys_before
    handle.remove()


def test_align_model_without_generation_config():
    class Tokenizer:
        pad_token_id = 3
        eos_token_id = 2

        def __len__(self):
            return 4

    class Embeddings:
        num_embeddings = 4

    class Model:
        generation_config = None
        config = SimpleNamespace(pad_token_id=None)

        def get_input_embeddings(self):
            return Embeddings()

        def resize_token_embeddings(self, *_args, **_kwargs):
            raise AssertionError("matching vocabulary must not be resized")

    model = Model()
    align_model_and_tokenizer(model, Tokenizer())
    assert model.config.pad_token_id == 3
