"""Train / inference configs for the librarian baseline, registered into openpi's config registry by register()."""
import dataclasses
import pathlib

from typing_extensions import override

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import weight_loaders
import openpi.training.optimizer as _optimizer
import openpi.transforms as _transforms

from . import librarian_policy

ROOT = pathlib.Path(__file__).resolve().parents[2]
REPO_ID = "me295/librarian_pick"          # LeRobot dataset id (stored under $HF_LEROBOT_HOME/<REPO_ID>)


@dataclasses.dataclass(frozen=True)
class LeRobotLibrarianDataConfig(_config.DataConfigFactory):
    """LeRobot dataset written by baseline/openpi_ext/convert_to_lerobot.py. Joint targets are absolute in the data
    and converted to deltas w.r.t. the current joints (gripper stays absolute), as in openpi's UR5 recipe."""

    @override
    def create(self, assets_dirs: pathlib.Path, model_config) -> _config.DataConfig:
        repack = _transforms.Group(inputs=[_transforms.RepackTransform({
            "observation/ext_image": "ext_image",
            "observation/wrist_image": "wrist_image",
            "observation/state": "state",
            "actions": "actions",
            "prompt": "prompt",
        })])
        data_transforms = _transforms.Group(
            inputs=[librarian_policy.LibrarianInputs(model_type=model_config.model_type)],
            outputs=[librarian_policy.LibrarianOutputs()],
        )
        mask = _transforms.make_bool_mask(6, -1)
        data_transforms = data_transforms.push(inputs=[_transforms.DeltaActions(mask)],
                                               outputs=[_transforms.AbsoluteActions(mask)])
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack,
            data_transforms=data_transforms,
            model_transforms=_config.ModelTransformFactory()(model_config),
        )


_LORA = pi0_config.Pi0Config(pi05=True, action_horizon=10, paligemma_variant="gemma_2b_lora",
                             action_expert_variant="gemma_300m_lora")

CONFIGS = [
    # Main baseline: pi0.5 base fine-tuned (LoRA) on scripted-expert demos of the pick-up task.
    _config.TrainConfig(
        name="pi05_librarian_lora",
        model=_LORA,
        data=LeRobotLibrarianDataConfig(repo_id=REPO_ID, base_config=_config.DataConfig(prompt_from_task=True)),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        freeze_filter=_LORA.get_freeze_filter(),
        ema_decay=None,
        batch_size=32,
        num_train_steps=10_000,
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=500, peak_lr=5e-5, decay_steps=10_000,
                                                   decay_lr=5e-6),
        save_interval=2_500,
        keep_period=None,
        log_interval=50,
        wandb_enabled=False,
        assets_base_dir=str(ROOT / "baseline" / "assets"),
        checkpoint_base_dir=str(ROOT / "baseline" / "checkpoints"),
    ),
]


def register() -> None:
    for c in CONFIGS:
        if c.name not in _config._CONFIGS_DICT:
            _config._CONFIGS.append(c)
            _config._CONFIGS_DICT[c.name] = c
