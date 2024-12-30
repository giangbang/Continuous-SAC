import sys
import os
import torch
from datetime import datetime
import numpy as np
import time
from collections import deque
import pandas as pd

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore[misc, assignment]


class Logger:
    """
    Logging class, support printing monitoring information to `std_out` and tf.event files
    """

    def __init__(
        self,
        run_name=datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        folder="runs",
        algo="sac",
        env="Env",
    ):
        self.run_name = run_name
        self.dir_name = f"{folder}/{env}/{algo}/{run_name}"
        self.writer = SummaryWriter(self.dir_name)
        self.name_to_values = dict()
        self.current_env_step = 0
        self.start_time = time.time()
        self.last2file = -float("inf")
        self._data = dict()
        self.save_every = 10 * 60

    def add_hyperparams(self, hyperparams: dict):
        """
        Save hyperparameters into tensorboard
        """
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % ("\n".join([f"|{key}|{value}|" for key, value in hyperparams.items()])),
        )

    def add_run_command(self):
        """
        Automatically save terminal command to tensorboard
        """
        cmd = " ".join(sys.argv)
        self.writer.add_text("terminal", cmd)
        with open(os.path.join(self.dir_name, "cmd.txt"), "w") as file:
            file.write(cmd)

    def add_scalar(self, key, val, step, smoothing=True):
        self.writer.add_scalar(key, val, step)
        if key not in self.name_to_values:
            self.name_to_values[key] = deque(maxlen=5 if smoothing else 1)
        self.name_to_values[key].extend([val])
        self.current_env_step = max(self.current_env_step, step)
        step = self.current_env_step

        if (
            len(getattr(self._data, "global_step", [])) > 0
            and self._data["global_step"][-1] == step
        ):
            data = getattr(self._data, key, [])
            if data:
                data[-1] = val
            else:
                self._data[key] = [val]
        else:
            self._data[key] = getattr(self._data, key, []) + [val]
            self._data["global_step"] = getattr(self._data, "global_step", []) + [step]
            for k, v in self._data.items():
                if k != key and k != "global_step":
                    v.append(None)

        if time.time() - self.last2file > self.save_every:
            self.save2txt()
            self.last2file = time.time()

    def to_df(self):
        max_len = len(self._data["global_step"])
        for k, v in self._data.items():
            if k != "global_step" and len(v) < max_len:
                self._data[k] = [None] * (max_len - len(v)) + v
        df = pd.DataFrame(self._data)
        df.set_index("global_step", inplace=True)
        return df

    def save2txt(self, file_name: str = None):
        if file_name is None:
            file_name = os.path.join(self.dir_name, "progress.csv")
        self.to_df().to_csv(file_name)

    def close(self):
        self.writer.close()

    def log_stdout(self):
        """
        Print results to terminal
        """
        results = {}
        for name, vals in self.name_to_values.items():
            results[name] = np.mean(vals)
        results["step"] = self.current_env_step
        pprint(results)

    def __getitem__(self, key):
        if key not in self.name_to_values:
            self.name_to_values[key] = self._default_values()
        return self.name_to_values.get(key)

    def __setitem__(self, key, val):
        self[key].extend([val])

    def _default_values(self, deque_len=10):
        return deque(maxlen=deque_len)

    def fps(self):
        """
        Measuring the fps
        """
        time_pass = time.time() - self.start_time  # in second
        return int(self.current_env_step / time_pass)


def pprint(dict_data):
    """Pretty print Hyper-parameters"""
    hyper_param_space, value_space = 40, 40
    format_str = "| {:<" + f"{hyper_param_space}" + "} | {:<" + f"{value_space}" + "}|"
    hbar = "-" * (hyper_param_space + value_space + 6)

    print(hbar)

    for k, v in dict_data.items():
        print(format_str.format(truncate_str(str(k), 40), truncate_str(str(v), 40)))

    print(hbar)


def truncate_str(input_str, max_length):
    """Truncate the string if it exceeds `max_length`"""
    if len(input_str) > max_length - 3:
        return input_str[: max_length - 3] + "..."
    return input_str
