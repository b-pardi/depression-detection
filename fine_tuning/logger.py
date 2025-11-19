from transformers import TrainerCallback
import json
import os

class JSONMetricsLoggerCallback(TrainerCallback):
    """
    Saves all metrics emitted by SFTTrainer into a JSONL file.
    One line per log event.
    """
    def __init__(self, out_path="training_metrics.jsonl"):
        self.out_path = out_path

        # ensure directory exists
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        entry = {
            "step": state.global_step,
            "epoch": state.epoch,
            **logs
        }

        with open(self.out_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
