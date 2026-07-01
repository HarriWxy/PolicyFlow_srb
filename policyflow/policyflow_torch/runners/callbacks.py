import os
import torch


def make_save_model_cb(directory):
    def cb(runner, stat):
        path = os.path.join(directory, "model_{}.pt".format(stat["current_iteration"]))
        runner.save(path)
    return cb


def make_save_model_onnx_cb(directory):
    def cb(runner, stat):
        path = os.path.join(
            directory, "model_{}.onnx".format(stat["current_iteration"])
        )
        runner.export_onnx(path)

    return cb


def make_interval_cb(callback, interval):
    def cb(runner, stat):
        if stat["current_iteration"] % interval != 0:
            return
        callback(runner, stat)

    return cb


def make_tensorboard_cb(directory):
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=directory, flush_secs=10)

    def cb(runner, stat):
        it = stat["current_iteration"]
        
        if "training_info" in stat:
            training_info = stat["training_info"]
            for key, value in training_info.items():
                writer.add_scalar(key, value, it)

        mean_reward = (
            sum(stat["returns"]) / len(stat["returns"])
            if len(stat["returns"]) > 0
            else 0.0
        )
        mean_steps = (
            sum(stat["lengths"]) / len(stat["lengths"])
            if len(stat["lengths"]) > 0
            else 0.0
        )
        writer.add_scalar("rollout/ep_rew_mean", mean_reward, it)
        writer.add_scalar("rollout/ep_len_mean", mean_steps, it)

        # Log reward_terms (matching SB3 RewardTermsTensorboardCallback)
        reward_terms_list = stat.get("reward_terms", [])
        if reward_terms_list:
            reward_terms_by_name = {}
            for step_reward_terms in reward_terms_list:
                if not isinstance(step_reward_terms, dict):
                    continue
                for name, value in step_reward_terms.items():
                    if isinstance(value, torch.Tensor):
                        vals = value.detach().cpu().float().flatten().tolist()
                    elif hasattr(value, '__iter__'):
                        vals = [float(v) for v in value]
                    else:
                        vals = [float(value)]
                    reward_terms_by_name.setdefault(name, []).extend(vals)
            for name, values in sorted(reward_terms_by_name.items()):
                if values:
                    import numpy
                    mean_val = float(numpy.asarray(values, dtype=numpy.float32).mean())
                    writer.add_scalar(f"rollout/reward_terms/{name}", mean_val, it)

        info = stat.get("info", [])

        if info:
            for key in info[0]:
                info_tensor = torch.tensor([], device=runner._device)
                for ep_info in info:
                    # handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    info_tensor = torch.cat((info_tensor, ep_info[key].to(runner._device)))
                value = torch.mean(info_tensor)
                # log to logger and terminal
                if "/" in key:
                    writer.add_scalar(key, value, it)
                else:
                    writer.add_scalar("Episode/" + key, value, it)

    return cb
