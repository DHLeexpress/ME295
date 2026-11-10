"""Map the librarian environment's observations to pi0/pi0.5 inputs and the model's outputs back to robot actions.

Same layout as openpi's UR5 example (examples/ur5/README.md): state = 6 UR joint angles + gripper (0 open .. 1
closed), actions = absolute joint targets + gripper, cameras = one third-person (base) + one wrist camera.
"""
import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def make_librarian_example() -> dict:
    return {
        "observation/ext_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/state": np.random.rand(7).astype(np.float32),
        "prompt": "pick up the blue book",
    }


@dataclasses.dataclass(frozen=True)
class LibrarianInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base = _parse_image(data["observation/ext_image"])
        wrist = _parse_image(data["observation/wrist_image"])
        if self.model_type == _model.ModelType.PI0_FAST:
            names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
            images = (base, np.zeros_like(base), wrist)
            masks = (np.True_, np.True_, np.True_)
        else:
            names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
            images = (base, wrist, np.zeros_like(base))
            masks = (np.True_, np.True_, np.False_)
        inputs = {
            "state": np.asarray(data["observation/state"], np.float32),
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, masks, strict=True)),
        }
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], np.float32)
        if "prompt" in data:
            p = data["prompt"]
            inputs["prompt"] = p.decode("utf-8") if isinstance(p, bytes) else p
        return inputs


@dataclasses.dataclass(frozen=True)
class LibrarianOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :7])}
