"""Small optional W&B logger; only the student leader initializes the SDK."""
import json
from uuid import uuid4
from pathlib import Path
from .config import steps_for


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
        for prefix in ("train/*", "val/*", "progress/*"):
            self.run.define_metric(prefix, step_metric="progress/global_step")

    def log(self, event, fields):
        if self.run is None or event not in {"train", "validation"}:
            return
        stage, step = fields["stage"], fields["step"]
        values = {"progress/global_step": ((stage - 1) * self.epoch_steps if self.cfg.get("training_mode") == "epoch"
                                                  else sum(steps_for(self.cfg, s) for s in range(stage - 1))) + step,
                  "progress/stage": stage, "progress/stage_step": step}
        if event == "train":
            values.update({f"train/{k}": fields[k] for k in
                           ("nmse", "cosine", "lr", "grad_norm", "seconds_per_step", "tokens_per_second")})
            values["train/loss"] = fields["nmse"] + self.cfg["cosine_weight"] * (1 - fields["cosine"])
        else:
            # Actual student forward; teacher-forced details remain in metrics.jsonl.
            values.update({f"val/{k}": fields["student_forward"][k] for k in ("nmse", "rrms", "cosine")})
            if "original_teacher_kl" in fields:
                values["val/original_teacher_kl"] = fields["original_teacher_kl"]
        # W&B owns its internal history step; the explicit optimizer-step axis
        # permits train + validation at the same step and checkpoint replay.
        self.run.log(values)

    def finish(self, exit_code=0):
        if self.run is not None:
            self.run.finish(exit_code=exit_code)
            self.run = None
