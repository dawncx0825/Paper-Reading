import torch

from vit_repro.model import ViTSpec, VisionTransformer
from vit_repro.weights import interpolate_position_embedding


def tiny_model(image_size: int = 32) -> VisionTransformer:
    return VisionTransformer(
        ViTSpec(
            image_size=image_size,
            patch_size=8,
            num_classes=7,
            hidden_size=32,
            depth=2,
            num_heads=4,
            mlp_dim=64,
        )
    )


def test_model_shape_and_zero_head():
    model = tiny_model()
    output = model(torch.randn(3, 3, 32, 32))
    assert output.shape == (3, 7)
    assert model.position_embedding.shape == (1, 17, 32)
    assert torch.count_nonzero(model.head.weight) == 0
    assert torch.count_nonzero(output) == 0


def test_position_interpolation_keeps_class_position():
    source = torch.randn(1, 197, 32)
    target = interpolate_position_embedding(source, (24, 24))
    assert target.shape == (1, 577, 32)
    torch.testing.assert_close(target[:, 0], source[:, 0])


def test_wrong_image_size_is_rejected():
    model = tiny_model()
    try:
        model(torch.randn(1, 3, 24, 24))
    except ValueError as error:
        assert "Expected 32x32" in str(error)
    else:
        raise AssertionError("Expected image-size validation failure")

