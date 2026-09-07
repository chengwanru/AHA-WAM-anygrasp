import sys
import os
import subprocess

sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./description/utils")
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

import numpy as np
from pathlib import Path
from collections import deque
import traceback

import yaml
from datetime import datetime
import importlib
import argparse
import pdb
import json

from generate_episode_instructions import *

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def _resolve_ffmpeg_exe() -> str:
    """Absolute ffmpeg path — do not rely on PATH (cluster images often lack it)."""
    from envs.utils.images_to_video import resolve_ffmpeg_exe

    return resolve_ffmpeg_exe()


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance


def eval_function_decorator(policy_name, model_name):
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e

def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "../task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def get_eval_video_size(args):
    head_camera_cfg = get_camera_config(args["camera"]["head_camera_type"])
    video_w = int(head_camera_cfg["w"])
    video_h = int(head_camera_cfg["h"])

    if args["camera"].get("collect_wrist_camera", False):
        wrist_camera_cfg = get_camera_config(args["camera"]["wrist_camera_type"])
        wrist_w = int(wrist_camera_cfg["w"])
        wrist_h = int(wrist_camera_cfg["h"])
        video_w = max(video_w, wrist_w * 2)
        video_h = video_h + wrist_h

    return f"{video_w}x{video_h}"


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    return bool(value)


def _result_suffix_from_task_config(task_config):
    if task_config == "demo_clean":
        return "clean"
    if task_config == "demo_randomized":
        return "random"
    raise ValueError(
        f"Unsupported `task_config` for fixed result naming: {task_config}. "
        "Expected one of: ['demo_clean', 'demo_randomized']."
    )


def main(usr_args):
    eval_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    # checkpoint_num = usr_args['checkpoint_num']
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args["instruction_type"]
    skip_get_obs_within_replan = parse_bool(usr_args.get("skip_get_obs_within_replan", False))
    eval_num_episodes = int(usr_args.get("eval_num_episodes", 100))
    if eval_num_episodes <= 0:
        raise ValueError(f"`eval_num_episodes` must be > 0, got: {eval_num_episodes}")
    eval_output_dir = usr_args.get("eval_output_dir")
    save_dir = None
    video_save_dir = None
    video_size = None

    get_model = eval_function_decorator(policy_name, "get_model")

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    if eval_output_dir is not None and str(eval_output_dir).strip() != "":
        save_dir = Path(str(eval_output_dir))
    else:
        save_dir = Path(f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{eval_ts}")
    save_dir.mkdir(parents=True, exist_ok=True)

    if args["eval_video_log"]:
        video_save_dir = save_dir
        video_size = get_eval_video_size(args)
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir

    # output camera config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    seed = usr_args["seed"]

    st_seed = 100000 * (1 + seed)
    suc_nums = []
    test_num = eval_num_episodes
    topk = 1

    model = get_model(usr_args)
    st_seed, suc_num = eval_policy(task_name,
                                   TASK_ENV,
                                   args,
                                   model,
                                   st_seed,
                                   test_num=test_num,
                                   video_size=video_size,
                                   instruction_type=instruction_type,
                                   skip_get_obs_within_replan=skip_get_obs_within_replan,
                                   usr_args=usr_args)
    suc_nums.append(suc_num)

    topk_success_rate = sorted(suc_nums, reverse=True)[:topk]

    result_suffix = _result_suffix_from_task_config(task_config)
    file_path = os.path.join(save_dir, f"_result_{result_suffix}.txt")
    with open(file_path, "w") as file:
        file.write(f"Timestamp: {eval_ts}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        # file.write(str(task_reward) + '\n')
        file.write("\n".join(map(str, np.array(suc_nums) / test_num)))

    print(f"Data has been saved to {file_path}")
    # return task_reward


def eval_policy(task_name,
                TASK_ENV,
                args,
                model,
                st_seed,
                test_num=100,
                video_size=None,
                instruction_type=None,
                skip_get_obs_within_replan=False,
                usr_args=None):
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")
    usr_args = usr_args or {}

    expert_check = os.environ.get("SKIP_EXPERT_CHECK", "0") not in ("1", "true", "True", "TRUE")
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0

    now_id = 0
    succ_seed = 0
    suc_test_seed_list = []

    policy_name = args["policy_name"]
    eval_func = eval_function_decorator(policy_name, "eval")
    reset_func = eval_function_decorator(policy_name, "reset_model")

    now_seed = st_seed
    task_total_reward = 0
    clear_cache_freq = args["clear_cache_freq"]

    args["eval_mode"] = True

    while succ_seed < test_num:
        render_freq = args["render_freq"]
        args["render_freq"] = 0

        if expert_check:
            try:
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except UnStableError as e:
                # print(" -------------")
                # print("Error: ", e)
                # print(" -------------")
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                continue
            except Exception as e:
                print(" -------------")
                print("Error: ", e)
                print("Stack Trace: ", traceback.format_exc())
                print(" -------------")
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                print("error occurs !")
                continue

        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            succ_seed += 1
            suc_test_seed_list.append(now_seed)
        else:
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq

        try:
            TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        except UnStableError as e:
            # This seed passed expert_check but failed during rollout env init.
            # Roll back the accepted-seed counter and skip to next seed.
            succ_seed -= 1
            if len(suc_test_seed_list) > 0 and suc_test_seed_list[-1] == now_seed:
                suc_test_seed_list.pop()
            TASK_ENV.close_env()
            now_seed += 1
            continue
        except Exception as e:
            succ_seed -= 1
            if len(suc_test_seed_list) > 0 and suc_test_seed_list[-1] == now_seed:
                suc_test_seed_list.pop()
            print(" -------------")
            print("Error: ", e)
            print("Stack Trace: ", traceback.format_exc())
            print(" -------------")
            TASK_ENV.close_env()
            now_seed += 1
            print("error occurs !")
            continue
        if expert_check:
            episode_info_list = [episode_info["info"]]
            results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)
            instruction = np.random.choice(results[0][instruction_type])
        else:
            instruction = f"{args['task_name'].replace('_', ' ')}"
        TASK_ENV.set_instruction(instruction=instruction)  # set language instruction

        current_video_path = None
        if TASK_ENV.eval_video_path is not None:
            episode_idx = TASK_ENV.test_num
            # Include pid so parallel jobs never collide on episodeN.mp4
            current_video_path = (
                Path(TASK_ENV.eval_video_path) / f"episode{episode_idx}.pid{os.getpid()}.mp4"
            )
            ffmpeg = subprocess.Popen(
                [
                    _resolve_ffmpeg_exe(),
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    video_size,
                    "-framerate",
                    "10",
                    "-i",
                    "-",
                    "-pix_fmt",
                    "yuv420p",
                    "-vcodec",
                    "libx264",
                    "-crf",
                    "23",
                    str(current_video_path),
                ],
                stdin=subprocess.PIPE,
            )
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

        succ = False
        reset_func(model)
        task_state_history = []

        # Simple open_microwave fallback: if the door stops moving while the
        # gripper is closed, force the gripper open for a few steps while the
        # policy continues to move the arm.  This lets the policy re-approach
        # the handle and re-grasp.  Gated by OPEN_MICROWAVE_FALLBACK.
        fallback_enabled = (
            task_name == "open_microwave"
            and os.environ.get("OPEN_MICROWAVE_FALLBACK", "0") in ("1", "true", "True", "TRUE")
        )
        fallback_window = 20
        fallback_qpos_threshold = 0.01
        fallback_min_step = 100
        fallback_open_steps = 10
        fallback_cooldown = 0

        def _pose_to_lists(pose):
            """Handle both sapien Pose objects and plain [p,q] lists."""
            if hasattr(pose, "p") and hasattr(pose, "q"):
                return pose.p.tolist(), pose.q.tolist()
            if isinstance(pose, (list, tuple, np.ndarray)) and len(pose) >= 7:
                return pose[:3].tolist() if hasattr(pose, "tolist") else list(pose[:3]), pose[3:7].tolist() if hasattr(pose, "tolist") else list(pose[3:7])
            return None, None

        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            need_obs = True
            if skip_get_obs_within_replan and hasattr(model, "should_request_observation"):
                need_obs = bool(model.should_request_observation())

            observation = None
            if need_obs:
                observation = TASK_ENV.get_obs()
            eval_func(TASK_ENV, model, observation)

            # Collect task-specific state for failure analysis.
            if task_name == "turn_switch" and hasattr(TASK_ENV, "robot") and hasattr(TASK_ENV, "switch"):
                try:
                    left_tcp = TASK_ENV.robot.get_left_tcp_pose()
                    right_tcp = TASK_ENV.robot.get_right_tcp_pose()
                    switch_pose = TASK_ENV.switch.get_pose()
                    left_tcp_p, left_tcp_q = _pose_to_lists(left_tcp)
                    right_tcp_p, right_tcp_q = _pose_to_lists(right_tcp)
                    task_state_history.append({
                        "step": int(TASK_ENV.take_action_cnt),
                        "left_tcp_p": left_tcp_p,
                        "left_tcp_q": left_tcp_q,
                        "right_tcp_p": right_tcp_p,
                        "right_tcp_q": right_tcp_q,
                        "switch_p": switch_pose.p.tolist() if hasattr(switch_pose, "p") else None,
                        "switch_q": switch_pose.q.tolist() if hasattr(switch_pose, "q") else None,
                        "switch_qpos": TASK_ENV.switch.get_qpos().tolist(),
                        "left_gripper_val": TASK_ENV.robot.get_left_gripper_val(),
                        "right_gripper_val": TASK_ENV.robot.get_right_gripper_val(),
                    })
                except Exception:
                    pass
            elif task_name == "open_microwave" and hasattr(TASK_ENV, "robot") and hasattr(TASK_ENV, "microwave"):
                try:
                    left_tcp = TASK_ENV.robot.get_left_tcp_pose()
                    right_tcp = TASK_ENV.robot.get_right_tcp_pose()
                    microwave_pose = TASK_ENV.microwave.get_pose()
                    microwave_qpos = TASK_ENV.microwave.get_qpos().tolist()
                    microwave_qlimits = TASK_ENV.microwave.get_qlimits().tolist()
                    left_tcp_p, left_tcp_q = _pose_to_lists(left_tcp)
                    right_tcp_p, right_tcp_q = _pose_to_lists(right_tcp)
                    task_state_history.append({
                        "step": int(TASK_ENV.take_action_cnt),
                        "left_tcp_p": left_tcp_p,
                        "left_tcp_q": left_tcp_q,
                        "right_tcp_p": right_tcp_p,
                        "right_tcp_q": right_tcp_q,
                        "microwave_p": microwave_pose.p.tolist() if hasattr(microwave_pose, "p") else None,
                        "microwave_q": microwave_pose.q.tolist() if hasattr(microwave_pose, "q") else None,
                        "microwave_qpos": microwave_qpos,
                        "microwave_qlimits": microwave_qlimits,
                        "left_gripper_val": TASK_ENV.robot.get_left_gripper_val(),
                        "right_gripper_val": TASK_ENV.robot.get_right_gripper_val(),
                    })
                except Exception:
                    pass

            # Detect stuck door and trigger gripper-open fallback.
            if fallback_enabled and fallback_cooldown == 0 and TASK_ENV.take_action_cnt >= fallback_min_step and len(task_state_history) >= fallback_window:
                recent = task_state_history[-fallback_window:]
                qpos_values = [s["microwave_qpos"][0] for s in recent]
                gripper_values = [s["left_gripper_val"] for s in recent]
                qpos_stuck = max(qpos_values) - min(qpos_values) < fallback_qpos_threshold
                gripper_closed = all(g < 0.5 for g in gripper_values)
                if qpos_stuck and gripper_closed:
                    print(
                        f"[open_microwave fallback] detected stuck door at step "
                        f"{TASK_ENV.take_action_cnt} (qpos range {max(qpos_values)-min(qpos_values):.4f}); "
                        f"forcing gripper open for {fallback_open_steps} steps"
                    )
                    if hasattr(model, "pending_actions"):
                        model.pending_actions.clear()
                    model.gripper_override_steps = fallback_open_steps
                    model.gripper_override_value = 1.0
                    fallback_cooldown = fallback_open_steps + fallback_window

            if fallback_cooldown > 0:
                fallback_cooldown -= 1

            if TASK_ENV.eval_success:
                succ = True
                break
        # task_total_reward += TASK_ENV.episode_score
        if TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()
            if current_video_path is None or not current_video_path.exists():
                # Video is diagnostic only — do not abort a successful/failed episode
                # when ffmpeg output is missing (encoder issues or stale races).
                print(
                    f"\033[93m[WARN] eval video missing (skipped): {current_video_path}\033[0m"
                )
            else:
                is_randomized = "randomized" in str(args["task_config"]).lower()
                renamed_video_path = (
                    Path(TASK_ENV.eval_video_path)
                    / f"episode{episode_idx}_randomized-{str(is_randomized).lower()}_success-{str(succ).lower()}.pid{os.getpid()}.mp4"
                )
                try:
                    current_video_path.rename(renamed_video_path)
                except FileNotFoundError:
                    print(
                        f"\033[93m[WARN] eval video rename race (skipped): {current_video_path}\033[0m"
                    )

        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        timing_getter = getattr(model, "get_timing_rollout", None)
        step_log_getter = getattr(model, "get_step_log", None)
        if callable(timing_getter):
            timing = timing_getter()
            take_action_cnt = max(int(getattr(TASK_ENV, "take_action_cnt", 0)), 1)
            infer_s = float(timing.get("infer_s", 0.0))
            sim_s = float(timing.get("sim_s", 0.0))
            infer_calls = max(int(round(float(timing.get("infer_calls", 0.0)))), 1)
            prefill_s = float(timing.get("prefill_s", 0.0))
            prefill_calls = max(int(round(float(timing.get("prefill_calls", 0.0)))), 0)
            action_chunk_s = float(timing.get("action_chunk_s", 0.0))
            action_chunk_calls = max(int(round(float(timing.get("action_chunk_calls", 0.0)))), 0)
            skipped_prefills = max(int(round(float(timing.get("skipped_prefills", 0.0)))), 0)
            prefill_decisions = max(int(round(float(timing.get("prefill_decisions", 0.0)))), 1)
            skipped_prefill_ratio = skipped_prefills / prefill_decisions
            skipped_steps = max(int(round(float(timing.get("skipped_steps", 0.0)))), 0)
            chunk_timing_parts = []
            for chunk_idx in range(1, 5):
                chunk_s = float(timing.get(f"chunk_{chunk_idx}_s", 0.0))
                chunk_calls = max(int(round(float(timing.get(f"chunk_{chunk_idx}_calls", 0.0)))), 0)
                chunk_avg_s = (chunk_s / chunk_calls) if chunk_calls > 0 else 0.0
                chunk_timing_parts.append(
                    f"chunk{chunk_idx}_s={chunk_s:.6f} | "
                    f"chunk{chunk_idx}_avg_s={chunk_avg_s:.6f} | "
                    f"chunk{chunk_idx}_calls={chunk_calls}"
                )

            prefill_sub_calls = max(int(round(float(timing.get("prefill_sub_calls", 0.0)))), 1)
            chunk_sub_calls = max(int(round(float(timing.get("chunk_sub_calls", 0.0)))), 1)
            sub_timing_parts = [
                f"prefill_prepare_s={float(timing.get('prefill_prepare_s', 0.0)):.6f} | "
                f"prefill_prepare_avg_s={float(timing.get('prefill_prepare_s', 0.0)) / prefill_sub_calls:.6f}",
                f"prefill_vae_encode_s={float(timing.get('prefill_vae_encode_s', 0.0)):.6f} | "
                f"prefill_vae_encode_avg_s={float(timing.get('prefill_vae_encode_s', 0.0)) / prefill_sub_calls:.6f}",
                f"prefill_video_pre_dit_s={float(timing.get('prefill_video_pre_dit_s', 0.0)):.6f} | "
                f"prefill_video_pre_dit_avg_s={float(timing.get('prefill_video_pre_dit_s', 0.0)) / prefill_sub_calls:.6f}",
                f"prefill_video_dit_forward_s={float(timing.get('prefill_video_dit_forward_s', 0.0)):.6f} | "
                f"prefill_video_dit_forward_avg_s={float(timing.get('prefill_video_dit_forward_s', 0.0)) / prefill_sub_calls:.6f}",
                f"chunk_conditioning_s={float(timing.get('chunk_conditioning_s', 0.0)):.6f} | "
                f"chunk_conditioning_avg_s={float(timing.get('chunk_conditioning_s', 0.0)) / chunk_sub_calls:.6f}",
                f"chunk_action_denoise_s={float(timing.get('chunk_action_denoise_s', 0.0)):.6f} | "
                f"chunk_action_denoise_avg_s={float(timing.get('chunk_action_denoise_s', 0.0)) / chunk_sub_calls:.6f} | "
                f"chunk_action_denoise_step_avg_s={float(timing.get('chunk_action_denoise_step_avg_s', 0.0)):.6f}",
                f"chunk_cleanup_s={float(timing.get('chunk_cleanup_s', 0.0)):.6f} | "
                f"chunk_cleanup_avg_s={float(timing.get('chunk_cleanup_s', 0.0)) / chunk_sub_calls:.6f}",
            ]
            print(
                f"Timing | infer_s={infer_s:.6f} | sim_s={sim_s:.6f} | "
                f"infer_chunk_s={infer_s / infer_calls:.6f} | "
                f"exec_chunk_s={sim_s / infer_calls:.6f} | "
                f"infer_per_action_s={infer_s / take_action_cnt:.6f} | "
                f"sim_per_action_s={sim_s / take_action_cnt:.6f} | "
                f"prefill_s={prefill_s:.6f} | "
                f"prefill_avg_s={((prefill_s / prefill_calls) if prefill_calls > 0 else 0.0):.6f} | "
                f"action_chunk_s={action_chunk_s:.6f} | "
                f"action_chunk_avg_s={((action_chunk_s / action_chunk_calls) if action_chunk_calls > 0 else 0.0):.6f} | "
                f"take_action_cnt={take_action_cnt} | infer_calls={infer_calls} | "
                f"prefill_calls={prefill_calls} | action_chunk_calls={action_chunk_calls} | "
                f"skipped_prefills={skipped_prefills} | skipped_prefill_ratio={skipped_prefill_ratio:.2%} | "
                f"skipped_steps={skipped_steps} | "
                + " | ".join(chunk_timing_parts) + " | "
                + " | ".join(sub_timing_parts)
            )

            # Save detailed per-episode analysis log (latency + skip/phase extras).
            infer_per_action = (infer_s / take_action_cnt) if take_action_cnt > 0 else 0.0
            sim_per_action = (sim_s / take_action_cnt) if take_action_cnt > 0 else 0.0
            infer_per_call = (infer_s / infer_calls) if infer_calls > 0 else 0.0
            latency = {
                "infer_s": infer_s,
                "sim_s": sim_s,
                "prefill_s": prefill_s,
                "action_chunk_s": action_chunk_s,
                "infer_calls": infer_calls,
                "prefill_calls": prefill_calls,
                "action_chunk_calls": action_chunk_calls,
                "take_action_cnt": take_action_cnt,
                "infer_per_action_s": infer_per_action,
                "sim_per_action_s": sim_per_action,
                "infer_s_per_call": infer_per_call,
                "prefill_s_per_call": (prefill_s / prefill_calls) if prefill_calls > 0 else 0.0,
                "action_chunk_s_per_call": (
                    (action_chunk_s / action_chunk_calls) if action_chunk_calls > 0 else 0.0
                ),
                # Frequencies derived from closed-loop timing (Hz).
                "action_hz_policy_only": (take_action_cnt / infer_s) if infer_s > 0 else 0.0,
                "action_hz_infer_plus_sim": (
                    (take_action_cnt / (infer_s + sim_s)) if (infer_s + sim_s) > 0 else 0.0
                ),
                "chunk_hz": (1.0 / infer_per_call) if infer_per_call > 0 else 0.0,
                "skipped_prefills": skipped_prefills,
                "prefill_decisions": prefill_decisions,
                "skipped_prefill_ratio": skipped_prefill_ratio,
            }
            episode_log = {
                "task_name": task_name,
                "task_config": args.get("task_config"),
                "policy_name": args.get("policy_name"),
                "ckpt_setting": str(args.get("ckpt_setting")),
                "instruction_type": instruction_type,
                "action_horizon": usr_args.get("action_horizon"),
                "chunks_per_video_prefill": usr_args.get("chunks_per_video_prefill"),
                "num_inference_steps": usr_args.get("num_inference_steps"),
                "eval_output_dir": str(usr_args.get("eval_output_dir", "")),
                "episode_idx": int(episode_idx),
                "seed": int(now_seed),
                "success": bool(succ),
                "num_steps": int(take_action_cnt),
                "step_lim": int(getattr(TASK_ENV, "step_lim", -1) or -1),
                "video_dit_mode": os.environ.get("VIDEO_DIT_MODE", "baseline"),
                "ovcr_diag_mode": os.environ.get("OVCR_DIAG_MODE", "baseline"),
                "timing": timing,
                "latency": latency,
                "task_state_history": task_state_history,
            }
            step_log = step_log_getter() if callable(step_log_getter) else None
            if step_log is not None:
                episode_log["step_log"] = step_log
                # Aggregate skip_phase decisions embedded in step_log (if any).
                phase_counts = {}
                skip_true = 0
                skip_n = 0
                for s in step_log:
                    sp = s.get("skip_phase") if isinstance(s, dict) else None
                    if not isinstance(sp, dict):
                        continue
                    ph = str(sp.get("phase", "UNKNOWN"))
                    phase_counts[ph] = phase_counts.get(ph, 0) + 1
                    if "skip" in sp:
                        skip_n += 1
                        if sp.get("skip"):
                            skip_true += 1
                episode_log["skip_phase_summary"] = {
                    "phase_step_counts": phase_counts,
                    "skip_decisions": skip_n,
                    "skip_true": skip_true,
                    "skip_ratio_among_decisions": (skip_true / skip_n) if skip_n else 0.0,
                }
            analysis_dir = Path(str(usr_args.get("eval_output_dir", "."))) / "analysis"
            analysis_dir.mkdir(parents=True, exist_ok=True)
            log_path = analysis_dir / f"episode{episode_idx}_analysis.json"
            try:
                with open(log_path, "w", encoding="utf-8") as f:
                    json.dump(episode_log, f, indent=2, default=str)
                print(f"[analysis] wrote {log_path}")
            except Exception as e:
                print(f"Warning: failed to write episode analysis log: {e}")
        else:
            print(f"Timing unavailable | model_type={type(model).__name__}")

        now_id += 1
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))

        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.test_num += 1

        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | \033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => \033[95m{round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%\033[0m, current seed: \033[90m{now_seed}\033[0m\n"
        )
        # TASK_ENV._take_picture()
        now_seed += 1

    return now_seed, TASK_ENV.suc


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except:
                pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    return config


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    usr_args = parse_args_and_config()

    main(usr_args)
