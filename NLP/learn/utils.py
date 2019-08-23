
from collections import OrderedDict
import json
import logging
from pathlib import Path
import os
import re

import requests


logger = logging.getLogger(__name__)


""" Notification """


def get_session_name():
    session_name = "local"
    return session_name


def send_message_to_slack(webhook_url, title=None, message=None):  # pragma: no cover
    if message is None:
        data = {"text": f"{get_session_name()} session is exited."}
    else:
        data = {"attachments": [{"title": title, "text": message, "color": "#438C56"}]}

    try:
        if webhook_url == "":
            print(data["text"])
        else:
            requests.post(webhook_url, data=json.dumps(data))
    except Exception as e:
        print(str(e))


def load_optimizer_checkpoint(optimizer, checkpoint):
    optimizer.load_state_dict(checkpoint["optimizer"])

    logger.info(f"Load optimizer checkpoints...!")
    return optimizer


def get_sorted_path(checkpoint_dir, both_exist=False):
    paths = []
    for root, dirs, files in os.walk(checkpoint_dir):
        for f_name in files:
            if "model" in f_name or "optimizer" in f_name:
                paths.append(Path(root) / f_name)

    path_with_train_count = {}
    for path in paths:
        train_count = re.findall("\d+", path.name)[0]
        train_count = int(train_count)
        if train_count not in path_with_train_count:
            path_with_train_count[train_count] = {}

        if "model" in path.name:
            path_with_train_count[train_count]["model"] = path
        if "optimizer" in path.name:
            path_with_train_count[train_count]["optimizer"] = path

    if both_exist:
        remove_keys = []
        for key, checkpoint in path_with_train_count.items():
            if not ("model" in checkpoint and "optimizer" in checkpoint):
                remove_keys.append(key)

        for key in remove_keys:
            del path_with_train_count[key]

    return OrderedDict(sorted(path_with_train_count.items()))
