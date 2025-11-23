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
                
            # writer.add_scalar("Loss/policy", training_info["policy_loss"], it)
            # writer.add_scalar("Loss/value", training_info["value_loss"], it)
            # writer.add_scalar("Loss/entropy", training_info["entropy_loss"], it)
            # writer.add_scalar("Train/policy_std", training_info["policy_std"], it)
            # writer.add_scalar("Train/kl", training_info["kl"], it)
            # writer.add_scalar("Train/learning_rate", training_info["learning_rate"], it)

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
        writer.add_scalar("Train/mean_reward", mean_reward, it)
        writer.add_scalar("Train/mean_episode_length", mean_steps, it)

        info: dict = stat["info"]

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
