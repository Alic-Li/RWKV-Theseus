"""Small optional W&B logger; only the student leader initializes the SDK."""
import json
from uuid import uuid4
from pathlib import Path


class Tracker:
    def __init__(self):
        self.run = None
        self.cfg = None

    def start(self, cfg, topo, resume=False):
        if topo.rank != topo.leader or cfg["wandb_mode"] == "disabled":
            return
        import wandb
        self.cfg = cfg
        root = Path(cfg["output"])
        root.mkdir(parents=True, exist_ok=True)
        identity_path = root / "wandb_run.json"
        identity = {"project": cfg["wandb_project"], "entity": cfg["wandb_entity"]}
        previous = json.loads(identity_path.read_text()) if resume and identity_path.exists() else None
        reuse = (cfg["wandb_mode"] == "online" and previous is not None
                 and previous.get("mode") == "online"
                 and all(previous.get(k) == v for k, v in identity.items()))
        run_id = previous["id"] if reuse else uuid4().hex[:8]
        self.run = wandb.init(
            project=cfg["wandb_project"], entity=cfg["wandb_entity"],
            name=cfg["wandb_name"], id=run_id, mode=cfg["wandb_mode"],
            resume="allow" if reuse else None, dir=str(root),
            config={k: v for k, v in cfg.items() if not k.startswith("wandb_")},
            save_code=False, settings=wandb.Settings(init_timeout=60, console="off"),
        )
        temporary = identity_path.with_suffix(".tmp")
        temporary.write_text(json.dumps({**identity, "id": run_id, "mode": cfg["wandb_mode"]}))
        temporary.replace(identity_path)
        self.run.define_metric("progress/global_step")
        for prefix in ("sum_loss", "mean_cos", "layer_*", "val/*"):
            self.run.define_metric(prefix, step_metric="progress/global_step")

    def log(self, event, fields):
        if self.run is None or event not in {"train", "validation"}:
            return
        values = {"progress/global_step": fields["step"]}
        prefix = "" if event == "train" else "val/"
        values.update({prefix + key: fields[key] for key in ("sum_loss", "mean_cos")})
        for layer, details in fields["layers"].items():
            values.update({f"{prefix}{layer}/{key}": value for key, value in details.items()
                           if key in {"nmse", "cosine", "rrms", "grad_norm", "lr"}})
        self.run.log(values)

    def finish(self, exit_code=0):
        if self.run is not None:
            self.run.finish(exit_code=exit_code)
            self.run = None
