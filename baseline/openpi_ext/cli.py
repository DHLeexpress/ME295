"""Run openpi's scripts with the librarian configs registered (openpi's venv, from the ME295 root).

    PY=baseline/third_party/openpi/.venv/bin/python
    $PY -m baseline.openpi_ext.cli norm-stats pi05_librarian_lora
    $PY -m baseline.openpi_ext.cli train pi05_librarian_lora --exp-name v1 --overwrite
    $PY -m baseline.openpi_ext.cli serve pi05_librarian_lora --dir baseline/checkpoints/pi05_librarian_lora/v1/9999
    $PY -m baseline.openpi_ext.cli serve pi05_droid --dir gs://openpi-assets/checkpoints/pi05_droid --port 8001
"""
import argparse
import importlib
import logging
import pathlib
import sys

from . import configs

OPENPI = pathlib.Path(__file__).resolve().parents[1] / "third_party" / "openpi"


def _script(name: str):
    """Import openpi/scripts/<name>.py as a top-level module (spawned data-loader workers re-import it by name)."""
    if str(OPENPI / "scripts") not in sys.path:
        sys.path.insert(0, str(OPENPI / "scripts"))
    return importlib.import_module(name)


def main():
    configs.register()
    logging.basicConfig(level=logging.INFO, force=True)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "-h"
    if cmd == "norm-stats":
        ap = argparse.ArgumentParser()
        ap.add_argument("config")
        ap.add_argument("--max-frames", type=int, default=None)
        a = ap.parse_args(sys.argv[2:])
        _script("compute_norm_stats").main(a.config, a.max_frames)
    elif cmd == "train":
        from openpi.training import config as _config
        sys.argv = [sys.argv[0], *sys.argv[2:]]     # tyro: <config name> [--overrides]
        _script("train").main(_config.cli())
    elif cmd == "serve":
        from openpi.policies import policy_config
        from openpi.serving import websocket_policy_server
        from openpi.training import config as _config
        ap = argparse.ArgumentParser()
        ap.add_argument("config")
        ap.add_argument("--dir", required=True, help="checkpoint dir (local path or gs://...)")
        ap.add_argument("--port", type=int, default=8000)
        ap.add_argument("--default-prompt", default=None)
        a = ap.parse_args(sys.argv[2:])
        policy = policy_config.create_trained_policy(_config.get_config(a.config), a.dir,
                                                     default_prompt=a.default_prompt)
        logging.info("serving %s from %s on port %d", a.config, a.dir, a.port)
        websocket_policy_server.WebsocketPolicyServer(policy=policy, host="0.0.0.0", port=a.port,
                                                      metadata=policy.metadata).serve_forever()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
