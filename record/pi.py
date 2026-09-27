"""Physical Intelligence (pi-sdk) integration: contract, episodes, training profile, fine-tune.

Everything here follows the Partner API guide. pi_sdk is imported lazily so recording and
the tests work without it. Credentials: PI_API_KEY (organization key) from the environment;
GOOGLE_APPLICATION_CREDENTIALS is only needed at *install* time for PI's package registry.

Robot -> contract mapping (one camera, no separate state sensor):

    observation/camera/rgb/image      arm-mounted camera, H.264 in its own shard
    observation/robot/controls  (N,)  last-commanded lines, float32, 0.0 low / 1.0 high
    action/robot/controls       (N,)  commanded lines this step, float32
    system/step                       DEFAULT_FPS Hz

N and the line order come from record.controls.CONTROL_NAMES (W, O, L, A today: drive, arm up,
camera tilt, steer). This is contract v2. The earlier rc_excavator_v1 (10 values: 4 sticks + 6
buttons) still exists in PI and can't be changed or deleted; v2 is a separate contract with its
own names and its own ids file, so the two never mix (a fine-tune needs every episode on one
contract). Nothing has been created in PI under v2 yet, so its shape can still change here freely;
once `setup()` is actually run against PI, it is permanent.
"""
from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import cv2

from .controls import CONTROL_NAMES, N_CONTROLS
from .episode import DEFAULT_FPS, Step, load_episode

CONTRACT_NAME = "rc_excavator_v2"
PROFILE_NAME = "rc-excavator-training-v2"
FILTER_NAME = "rc-excavator-orbit-demos-v2"
SOURCE_DATASET_PATH = "rc-excavator/orbit-v2"
ROBOT_PLATFORM = CONTRACT_NAME
CAMERA_TOPIC = "observation/camera/rgb/image"
STATE_TOPIC = "observation/robot/controls"
ACTION_TOPIC = "action/robot/controls"
IDS_FILE = Path(".pi/ids.v2.json")  # v1's ids stay in .pi/ids.json, untouched

# Relative tolerance on the system/step rate in the contract. Our choice (the PI guide's example uses
# the same value), not a PI limit, and it can't be changed once the contract exists. Loosen it here
# BEFORE running pi-setup if a wireless camera can't hold the rate.
STEP_TOLERANCE = 0.1

DEFAULT_TASK = "orbit around the object"
BASE_MODEL = "pi07_cosmic"


def _sdk():
    try:
        import pi_sdk
    except ImportError as e:
        raise SystemExit(
            "pi-sdk is not installed. Follow the guide's install step:\n"
            "  pip install keyring keyrings.google-artifactregistry-auth\n"
            "  export GOOGLE_APPLICATION_CREDENTIALS=/path/to/pi-sa-key.json   # the rotated key\n"
            '  pip install "pi-sdk[video,inference]>=0.3.2" \\\n'
            "    --index-url https://us-east5-python.pkg.dev/pi-external-partners/pi-sdk/simple/ \\\n"
            "    --extra-index-url https://pypi.org/simple") from e
    return pi_sdk


def _api_key() -> str:
    key = os.environ.get("PI_API_KEY")
    if not key:
        raise SystemExit("PI_API_KEY is not set. `export PI_API_KEY=...` in your shell (never in a file "
                         "that gets committed, never in chat).")
    return key


def clients():
    pi_sdk = _sdk()
    key = _api_key()
    return pi_sdk.data.Client(api_key=key), pi_sdk.finetuning.Client(api_key=key)


# ---------------------------------------------------------------- saved ids
def load_ids() -> dict:
    return json.loads(IDS_FILE.read_text()) if IDS_FILE.exists() else {}


def save_ids(**new) -> dict:
    ids = {**load_ids(), **new}
    IDS_FILE.parent.mkdir(exist_ok=True)
    IDS_FILE.write_text(json.dumps(ids, indent=2))
    return ids


# --------------------------------------------------------------- contract
def build_contract(fps: int = DEFAULT_FPS):
    from pi_sdk.data import McapContract, NdArrayContract, TopicContract

    def array_topic(topic: str, role: str) -> TopicContract:
        return TopicContract(
            topic=topic, role=role, schema_encoding="protobuf", schema_name="pi.data.v1.NdArray",
            expected_frequency_hz=0, frequency_tolerance=0,
            ndarray=NdArrayContract(shape=(N_CONTROLS,), dtype="float32"),
            required_in_snapshots=("system/step",))

    return McapContract(topics=(
        TopicContract(topic="system/step", role="system", schema_encoding="protobuf",
                      schema_name="pi.data.v1.SnapshotRef", expected_frequency_hz=fps,
                      frequency_tolerance=STEP_TOLERANCE),
        TopicContract(topic=CAMERA_TOPIC, role="observation", schema_encoding="protobuf",
                      schema_name="pi.data.v1.VideoFrame", expected_frequency_hz=0,
                      frequency_tolerance=0, required_in_snapshots=("system/step",)),
        array_topic(STATE_TOPIC, "observation"),
        array_topic(ACTION_TOPIC, "action"),
    ))


def setup(fps: int = DEFAULT_FPS) -> dict:
    """Idempotently create the platform contract and training profile; remember their ids."""
    _sdk()
    from pi_sdk.finetuning import ArraySlice, TrainingProfileConfig

    data, ft = clients()
    res = data.ensure_platform_contract(
        CONTRACT_NAME, build_contract(fps),
        description=f"RC excavator, arm-mounted camera, {N_CONTROLS} digital lines "
                    f"({', '.join(CONTROL_NAMES)}: drive, arm up, camera tilt, steer), {fps} Hz")
    contract_id = res.platform_contract.platform_contract_id
    profile = ft.create_training_profile(
        PROFILE_NAME, contract_id,
        TrainingProfileConfig(
            # The camera sits on the excavator arm (wrist-like) but it is the only camera; treated as
            # the base camera. Revisit with PI if the base model expects a separate wrist view.
            base_camera_topics=(CAMERA_TOPIC,),
            wrist_camera_topics=(),
            state_slices=(ArraySlice(STATE_TOPIC),),
            action_topics=(ACTION_TOPIC,),
            action_space="absolute",
            # All lines low means "doing nothing", and the model has to see examples of that. The default
            # (False) lets PI skip idle stretches. Trim long idle head/tail from episodes instead.
            keep_idle_actions=True))
    return save_ids(platform_contract_id=contract_id, training_profile_id=profile.training_profile_id,
                    fps=fps)


# ---------------------------------------------------------------- episodes
def to_utc(ns: int) -> datetime:
    """Exact epoch-nanoseconds -> UTC datetime, floored to microseconds."""
    return datetime(1970, 1, 1, tzinfo=UTC) + timedelta(microseconds=ns // 1000)


def recording_window(steps: list[Step], fps: int) -> tuple[datetime, datetime]:
    start = to_utc(steps[0].t_ns)
    end = to_utc(steps[-1].t_ns) + timedelta(microseconds=1_000_000 // fps)  # end is exclusive
    return start, end


def write_shards(steps: list[Step], contract, out_dir: Path, fps: int) -> list[Path]:
    """One light shard (arrays + system/step) and one camera shard, per the guide."""
    from pi_sdk.mcap import v1

    out_dir.mkdir(parents=True, exist_ok=True)
    light_path, cam_path = out_dir / "light.mcap", out_dir / "camera.mcap"
    light, cam = v1.Writer(light_path, contract=contract), v1.Writer(cam_path, contract=contract)
    encoder = v1.VideoEncoder(fps=fps)

    for seq, step in enumerate(steps):
        bgr = cv2.imread(str(step.frame_path))
        if bgr is None:
            raise SystemExit(f"Cannot read {step.frame_path}")
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr[: h - h % 2, : w - w % 2], cv2.COLOR_BGR2RGB)  # H.264 needs even sizes
        for frame in encoder.encode(rgb, capture_time_ns=step.t_ns, sequence_id=seq):
            cam.write_video_frame(CAMERA_TOPIC, frame)
        fields = [v1.FieldRef(topic=CAMERA_TOPIC, sequence_id=seq, capture_time_ns=step.t_ns),
                  light.write_ndarray(STATE_TOPIC, step.state, capture_time_ns=step.t_ns, sequence_id=seq),
                  light.write_ndarray(ACTION_TOPIC, step.action, capture_time_ns=step.t_ns, sequence_id=seq)]
        light.write_step(fields, capture_time_ns=step.t_ns, sequence_id=seq)
    for frame in encoder.finalize():
        cam.write_video_frame(CAMERA_TOPIC, frame)
    light.close()
    cam.close()
    return [light_path, cam_path]


def upload_episode(episode_dir: Path, *, robot_id: str = "rc-excavator-1", shard_root: Path = Path("shards"),
                   success: bool = True) -> str:
    _sdk()
    from pi_sdk.data import (
        EpisodeDeclaration, EpisodeMetadata, EpisodeShard, control_source, language_segment,
    )

    ids = load_ids()
    if "platform_contract_id" not in ids:
        raise SystemExit("No platform contract yet. Run `python -m record pi-setup` first.")
    meta, steps = load_episode(episode_dir)
    fps = meta["fps"]
    if fps != ids.get("fps"):
        raise SystemExit(f"Episode was recorded at {fps} Hz but the contract is {ids.get('fps')} Hz.")
    data, _ = clients()
    episode_id, task = meta["episode_id"], meta["task"]
    start, end = recording_window(steps, fps)

    data.create_episode(
        EpisodeDeclaration(episode_id=episode_id, robot_id=robot_id, robot_platform=ROBOT_PLATFORM,
                           recording_start_time=start, recording_end_time=end),
        EpisodeMetadata(dataset="rc-excavator", task=task, is_eval_episode=False, success=success,
                        source_dataset_path=SOURCE_DATASET_PATH),
        platform_contract_id=ids["platform_contract_id"],
        spans=[control_source(start=start, end=end, value="human"),
               language_segment(start=start, end=end, task=task, subtask=task)])
    contract = data.get_platform_contract(ids["platform_contract_id"])
    for path in write_shards(steps, contract, shard_root / episode_id, fps):
        data.upload_shard(episode_id, EpisodeShard.create(
            path, episode_id=episode_id, recording_start_time=start, recording_end_time=end))
    data.finalize_episode_upload(episode_id)
    if data.get_episode(episode_id).upload_completed_at is None:
        raise SystemExit(f"episode {episode_id} did not finalize")
    return episode_id


# --------------------------------------------------------------- fine-tune
def start_finetune(name: str, *, num_steps: int | None, base_model: str = BASE_MODEL,
                   batch_size: int | None = None, episode_ids: tuple[str, ...] | None = None) -> str:
    """episode_ids, if given, scopes training to exactly those episodes instead of every
    successful episode on the contract. This needs its own dataset filter, not a reuse of
    FILTER_NAME: filters are immutable, and PI rejects reusing a name with a different
    query. The filter is named after this run, so each distinct episode selection gets
    its own filter rather than colliding with the broad one or with each other."""
    _sdk()
    from pi_sdk.finetuning import CheckpointPolicy, DatasetFilterInput, EpisodeFilter, StringSelection

    ids = load_ids()
    if "training_profile_id" not in ids:
        raise SystemExit("No training profile yet. Run `python -m record pi-setup` first.")
    _, ft = clients()
    if episode_ids:
        filter_name, episode_filter = f"{FILTER_NAME}-{name}", EpisodeFilter(
            platform_contract_id=ids["platform_contract_id"],
            episode_ids=StringSelection(include=tuple(episode_ids)))
    else:
        filter_name, episode_filter = FILTER_NAME, EpisodeFilter(
            platform_contract_id=ids["platform_contract_id"], source_dataset_path=SOURCE_DATASET_PATH,
            success=True)
    flt = ft.create_dataset_filter(filter_name, episode_filter)
    kwargs = {k: v for k, v in (("num_steps", num_steps), ("batch_size", batch_size)) if v is not None}
    run = ft.create_finetune(name, [DatasetFilterInput(flt.dataset_filter_id)], base_model=base_model,
                             checkpoint_policy=CheckpointPolicy("final_only"),
                             training_profile_id=ids["training_profile_id"], **kwargs)
    save_ids(last_run_id=run.run_id)
    return run.run_id


def wait_for_finetune(run_id: str, poll_s: float = 60) -> None:
    _, ft = clients()
    run = ft.get_finetune(run_id)
    while run.state in {"pending", "running"}:
        print(f"{run.state}" + (f": {run.progress:.0%}" if run.progress is not None else ""))
        time.sleep(poll_s)
        run = ft.get_finetune(run_id)
    if run.state != "succeeded":
        raise SystemExit(f"fine-tune {run.state}: {run.terminal_reason or ''}")
    for m in ft.list_finetune_evaluations(run_id):
        print(f"step {m.step}: train_mae={m.train_mae} validation_mae={m.validation_mae}")
