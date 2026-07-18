import pybullet as p
import pybullet_data
import math
import numpy as np
import copy
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import time
import os
from tqdm import tqdm

PATH = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 0. MODES TO OPTIMIZE
# ============================================================
# All or any subset of: "straight", "sideway", "diagonal", "spin".
MODES = ["straight"]

# ============================================================
# 1. SETTINGS & GLOBALS
# ============================================================
# Tweak these manually depending on the mode(s) you're running.
NUM_ROBOTS = 10		# = Number of robots
DURATION = 10		# = Lenght of each trial in seconds
SPACING = 1.5		#Distance between each robot
GAIT_TRIALS = 30		# = number of generations              
NUM_EXPERIMENTS = 1

# --- Renderer: "windowed" opens the pybullet GUI, "headless" runs in the background, suitable for parallel processing multiple experiments at the same time ---
RENDER_MODE = "windowed"
_RENDER_MAP = {"windowed": p.GUI, "headless": p.DIRECT}


# --- GA hyperparameters ---
ELITE_COUNT = 2
RANDOM_OUTSIDER = 1
TOURNAMENT_SIZE = 3
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.25
MUTATION_SIGMA = 0.15

# --- Settle phase durations ---
INITIAL_SETTLE_SEC = 1.0
GEN_SETTLE_SEC     = 0.5


TARGET_SPEEDS     = [0.15]	#Target Speed in (m/s) used for straight/sideway/diagonal
TARGET_ROT_SPEEDS = [0.6]	#Target rotational speed in (rad/s) used; TARGET_SPEEDS is ignored.
BODY_HEIGHTS      = [0.20]	#Robots Body height from the ground in meters
STEP_HEIGHTS      = [0.015]	#Step height in meters

# Directions for straight / sideway / spin: +1 or -1.
DIRECTIONS          = [1]	# Leave empty ([]) to skip these modes entirely.

# Directions for diagonal: angles in degrees (e.g. 45, 135, 225, 315).
DIAGONAL_DIRECTIONS = []	# Leave empty ([]) to skip diagonal.

# --- Kinematics / dynamics ---
L1 = 0.10
L2 = 0.10
URDF_X_OFFSET = 0.005
BODY_OFFSET = 0.0544

GLOBAL_MAX_TORQUE = 1.8
GLOBAL_MAX_VELOCITY = 6.0
LATERAL_FRICTION = 1.0

# --- Diagonal-only: forward/side amplitude split ---
FORWARD_SHARE = 0.625

# --- Gene search space: (low, high) ---
GENE_BOUNDS = {
    "step_amplitude": (0.005, 0.045),
    "frequency":      (0.6,   1.8),
}

GAIT_LIBRARY = {"tripod": [0.0, 0.5, 0.0, 0.5, 0.0, 0.5]}
PHYSICS_FREQ = 240
SERVO_FREQ = 100
SPAWN_ORI = p.getQuaternionFromEuler([0, 0, 0])

# Spin-mode leg roots
LEG_ROOTS = {
    "FL": [0.12, 0.10], "ML": [0.0, 0.14], "RL": [-0.12, 0.10],
    "FR": [0.12, -0.10], "MR": [0.0, -0.14], "RR": [-0.12, -0.10],
}
LEG_PREFIXES = ["FL", "ML", "RL", "FR", "MR", "RR"]

# ============================================================
# 2. IK
# ============================================================

def solve_leg_ik_3dof(target_x, target_y, target_z):
    x = target_x + URDF_X_OFFSET
    y = target_y
    z_from_thigh = -(target_z - BODY_OFFSET)
    hip_angle = math.atan2(y, -z_from_thigh)
    z_sagittal = -math.sqrt(y**2 + z_from_thigh**2)

    dist_sq = x**2 + z_sagittal**2
    dist = math.sqrt(dist_sq)

    if dist > (L1 + L2) * 0.99 or dist < abs(L1 - L2):
        return None, None, None

    cos_phi = (L1**2 + L2**2 - dist_sq) / (2 * L1 * L2)
    phi = math.acos(np.clip(cos_phi, -1.0, 1.0))
    knee_angle = math.pi - phi

    alpha = math.atan2(z_sagittal, x)
    cos_beta = (L1**2 + dist_sq - L2**2) / (2 * L1 * dist)
    beta = math.acos(np.clip(cos_beta, -1.0, 1.0))
    thigh_angle = alpha - beta + (math.pi / 2)

    return hip_angle, thigh_angle, knee_angle

# ============================================================
# 3. TRAJECTORY DISPATCH (per-mode foot target)
# ============================================================

def foot_target(mode, prefix, phase_multiplier, tz, dna, direction):
    if mode == "straight":
        tx = direction * dna["step_amplitude"] * phase_multiplier
        ty = 0.0
        return tx, ty, tz

    if mode == "sideway":
        tx = 0.0
        ty = direction * dna["step_amplitude"] * phase_multiplier
        return tx, ty, tz

    if mode == "diagonal":
        rad = math.radians(direction)  # direction is degrees here
        amp_x = dna["step_amplitude"] * FORWARD_SHARE
        amp_y = dna["step_amplitude"] * (1.0 - FORWARD_SHARE)
        sign_x = np.sign(math.cos(rad))
        sign_y = np.sign(math.sin(rad))
        tx = amp_x * phase_multiplier * abs(math.cos(rad)) * sign_x
        ty = amp_y * phase_multiplier * abs(math.sin(rad)) * sign_y
        return tx, ty, tz

    if mode == "spin":
        rx, ry = LEG_ROOTS[prefix]
        angle = dna["step_amplitude"] * phase_multiplier
        tx = -ry * angle * direction
        ty = rx * angle * direction
        return tx, ty, tz

    raise ValueError(f"Unknown mode {mode}")


def leg_phase_offset(mode, l_idx, cycle_phase, gait_offsets):
    if mode == "spin":
        prefix = LEG_PREFIXES[l_idx]
        is_group_a = prefix in ["FL", "RR", "ML"]
        return cycle_phase if is_group_a else (cycle_phase + 0.5) % 1.0
    return (cycle_phase + gait_offsets[l_idx]) % 1.0

# ============================================================
# 4. SCORING DISPATCH
# ============================================================

def compute_translational_score(mode, start_pos, current_pos, roll, pitch,
                                target_vel, current_orn, dna, direction):
    move_vec = np.array(current_pos[:2]) - np.array(start_pos[:2])

    if mode == "straight":
        fwd = np.array([1.0, 0.0])
        dist_along_dir = np.dot(move_vec, fwd) * direction
        lateral_deviation = abs(move_vec[1])
        drift_sigma_sq, speed_sigma_sq, stab_sigma_sq, yaw_sigma_sq = 0.01, 0.01, 0.01, 0.05
    elif mode == "sideway":
        fwd = np.array([0.0, 1.0])
        dist_along_dir = np.dot(move_vec, fwd) * direction
        lateral_deviation = abs(move_vec[0])
        drift_sigma_sq, speed_sigma_sq, stab_sigma_sq, yaw_sigma_sq = 0.01, 0.001, 0.03, 0.01
    else:  # diagonal
        rad = math.radians(direction)
        target_dir_vec = np.array([math.cos(rad), math.sin(rad)])
        dist_along_dir = np.dot(move_vec, target_dir_vec)
        proj = dist_along_dir * target_dir_vec
        lateral_deviation = float(np.linalg.norm(move_vec - proj))
        drift_sigma_sq, speed_sigma_sq, stab_sigma_sq, yaw_sigma_sq = 0.05, 0.001, 0.01, 0.005

    drift_penalty = math.exp(-(lateral_deviation ** 2) / drift_sigma_sq)
    actual_vel = dist_along_dir / DURATION
    speed_reward = math.exp(-((target_vel - actual_vel) ** 2) / speed_sigma_sq)

    target_height = start_pos[2]
    vertical_deviation = current_pos[2] - target_height
    stability_error = (roll ** 2 + pitch ** 2 + vertical_deviation ** 2)
    stability_reward = math.exp(-(stability_error) / stab_sigma_sq)

    _, _, yaw = p.getEulerFromQuaternion(current_orn)
    yaw_reward = math.exp(-(yaw ** 2) / yaw_sigma_sq)

    score = max(0.0, dist_along_dir)
    score *= (40.0 * speed_reward * stability_reward * yaw_reward * drift_penalty)

    num_steps = dna["frequency"] * DURATION
    stride_length = dist_along_dir / (num_steps + 1e-6)

    metrics = {
        "distance_traveled":   dist_along_dir,
        "avg_speed":           actual_vel,
        "speed_match_quality": speed_reward,
        "stability_quality":   stability_reward,
        "yaw_quality":         yaw_reward,
        "drift_quality":       drift_penalty,
        "lateral_deviation":   lateral_deviation,
        "stride_length":       stride_length,
        "vertical_deviation":  abs(vertical_deviation),
    }
    return score, metrics


def compute_spin_score(total_yaw, current_orn, current_pos, start_xy,
                       target_speed, spin_dir, dna):
    roll, pitch, _ = p.getEulerFromQuaternion(current_orn)

    actual_rot_speed = (total_yaw * spin_dir) / DURATION
    speed_error = abs(target_speed - actual_rot_speed)

    speed_reward = math.exp(-(speed_error ** 2) / 0.002)
    stability_reward = math.exp(-(abs(roll) + abs(pitch)) / 0.15)

    drift = float(np.linalg.norm(np.array(current_pos[:2]) - start_xy))
    drift_reward = math.exp(-drift / 0.05)

    score = max(0.0, total_yaw * spin_dir)
    score *= 40.0 * speed_reward * stability_reward * drift_reward

    num_steps = dna["frequency"] * DURATION
    yaw_per_cycle = (total_yaw * spin_dir) / (num_steps + 1e-6)
    efficiency_reward = math.tanh(yaw_per_cycle / 0.5)
    score *= (1.0 + 0.5 * efficiency_reward)

    metrics = {
        "actual_rot_speed":    actual_rot_speed,
        "speed_match_quality": speed_reward,
        "stability_quality":   stability_reward,
        "drift_quality":       drift_reward,
        "yaw_per_cycle":       yaw_per_cycle,
        "efficiency_reward":   efficiency_reward,
    }
    return score, metrics

# ============================================================
# 5. GENETIC ALGORITHM HELPERS
# ============================================================

def random_individual(rng):
    return {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in GENE_BOUNDS.items()}

def clip_gene(name, value):
    lo, hi = GENE_BOUNDS[name]
    return float(np.clip(value, lo, hi))

def tournament_select(pop, fitnesses, rng, k=TOURNAMENT_SIZE):
    idxs = rng.integers(0, len(pop), size=k)
    best = idxs[0]
    for i in idxs[1:]:
        if fitnesses[i] > fitnesses[best]:
            best = i
    return pop[best]

def crossover(parent_a, parent_b, rng):
    child = {}
    for gene in GENE_BOUNDS:
        if rng.random() < 0.5:
            alpha = rng.random()
            val = alpha * parent_a[gene] + (1.0 - alpha) * parent_b[gene]
        else:
            val = parent_a[gene] if rng.random() < 0.5 else parent_b[gene]
        child[gene] = clip_gene(gene, val)
    return child

def mutate(individual, rng):
    for gene, (lo, hi) in GENE_BOUNDS.items():
        if rng.random() < MUTATION_RATE:
            span = hi - lo
            individual[gene] = clip_gene(
                gene, individual[gene] + rng.normal(0.0, MUTATION_SIGMA * span)
            )
    return individual

def next_generation(pop, fitnesses, rng):
    N = len(pop)
    order = np.argsort(fitnesses)[::-1]
    new_pop = []

    n_elite = min(ELITE_COUNT, N)
    for i in range(n_elite):
        new_pop.append(copy.deepcopy(pop[order[i]]))

    n_outsiders = min(RANDOM_OUTSIDER, max(0, N - n_elite))
    n_offspring = N - n_elite - n_outsiders

    for _ in range(n_offspring):
        parent_a = tournament_select(pop, fitnesses, rng)
        parent_b = tournament_select(pop, fitnesses, rng)
        if rng.random() < CROSSOVER_RATE:
            child = crossover(parent_a, parent_b, rng)
        else:
            child = copy.deepcopy(parent_a)
        child = mutate(child, rng)
        new_pop.append(child)

    for _ in range(n_outsiders):
        new_pop.append(random_individual(rng))

    return new_pop

# ============================================================
# 6. RESET / STANCE HELPERS
# ============================================================

assert RENDER_MODE in _RENDER_MAP, f"RENDER_MODE must be one of {list(_RENDER_MAP)}"
PYBULLET_CONNECT_MODE = _RENDER_MAP[RENDER_MODE]

def hard_reset_robots(robots, start_positions, client_id):
    for i, r_id in enumerate(robots):
        p.resetBasePositionAndOrientation(
            r_id, start_positions[i], SPAWN_ORI, physicsClientId=client_id
        )
        p.resetBaseVelocity(r_id, [0, 0, 0], [0, 0, 0], physicsClientId=client_id)
        for j in range(p.getNumJoints(r_id, physicsClientId=client_id)):
            p.resetJointState(
                r_id, j,
                targetValue=0.0,
                targetVelocity=0.0,
                physicsClientId=client_id
            )

def hold_stance(robots, joint_map, leg_prefixes, body_height, seconds, client_id):
    h_set, th_set, kn_set = solve_leg_ik_3dof(0, 0, body_height, body_height)
    if h_set is None:
        return

    steps = int(seconds * PHYSICS_FREQ)
    for _ in range(steps):
        for r_id in robots:
            for prefix in leg_prefixes:
                is_left = prefix.endswith("L")
                is_right = prefix.endswith("R")

                actual_th = -th_set if is_left else th_set
                actual_kn = -kn_set if is_left else kn_set
                if is_right:
                    actual_th = th_set

                p.setJointMotorControl2(
                    r_id, joint_map[f"{prefix}3"],
                    p.POSITION_CONTROL, h_set,
                    force=GLOBAL_MAX_TORQUE
                )
                p.setJointMotorControl2(
                    r_id, joint_map[f"{prefix}2"],
                    p.POSITION_CONTROL, actual_th,
                    force=GLOBAL_MAX_TORQUE
                )
                p.setJointMotorControl2(
                    r_id, joint_map[f"{prefix}1"],
                    p.POSITION_CONTROL, actual_kn,
                    force=GLOBAL_MAX_TORQUE
                )
        p.stepSimulation(physicsClientId=client_id)
        if PYBULLET_CONNECT_MODE == p.GUI:
            time.sleep(1.0 / PHYSICS_FREQ)

# ============================================================
# 7. WORKER (GA INTEGRATED)
# ============================================================

def run_experiment_worker(args):
    mode, gait_name, gait_offsets, target_vel, body_height, step_height, direction, exp_idx = args
    rng = np.random.default_rng()

    if PYBULLET_CONNECT_MODE == p.GUI:
        client_id = p.connect(p.GUI, options="--width=4558 --height=1908")
    else:
        client_id = p.connect(p.DIRECT)

    p.resetSimulation(physicsClientId=client_id)
    p.setTimeStep(1.0 / PHYSICS_FREQ, physicsClientId=client_id)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)
    p.setGravity(0, 0, -9.81, physicsClientId=client_id)
    p.setRealTimeSimulation(0, physicsClientId=client_id)

    plane_id = p.loadURDF("plane.urdf", physicsClientId=client_id)
    p.changeDynamics(plane_id, -1, lateralFriction=LATERAL_FRICTION, physicsClientId=client_id)

    urdf_path = os.path.join(PATH, "HexaDog_ZBD.urdf")
    robots = [p.loadURDF(urdf_path,
                         [0, i * SPACING, 0.05], SPAWN_ORI,
                         physicsClientId=client_id,
                         flags=p.URDF_USE_INERTIA_FROM_FILE | p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
              for i in range(NUM_ROBOTS)]

    for r_id in robots:
        p.changeDynamics(r_id, -1, lateralFriction=LATERAL_FRICTION, physicsClientId=client_id)
        for j in range(p.getNumJoints(r_id, physicsClientId=client_id)):
            p.changeDynamics(r_id, j, lateralFriction=LATERAL_FRICTION, physicsClientId=client_id)

    start_positions = [[0, i * SPACING, body_height] for i in range(NUM_ROBOTS)]

    joint_map = {p.getJointInfo(robots[0], j, physicsClientId=client_id)[1].decode(): j
                 for j in range(p.getNumJoints(robots[0], physicsClientId=client_id))}

    hold_stance(robots, joint_map, LEG_PREFIXES, body_height,
                INITIAL_SETTLE_SEC, client_id)

    best_overall_score, best_overall_dna = -1e9, None
    best_overall_metrics = None
    servo_interval = int(PHYSICS_FREQ / SERVO_FREQ)

    population = [random_individual(rng) for _ in range(NUM_ROBOTS)]

    for gen in range(GAIT_TRIALS):

        hard_reset_robots(robots, start_positions, client_id)
        hold_stance(robots, joint_map, LEG_PREFIXES, body_height,
                    GEN_SETTLE_SEC, client_id)

        gen_start_positions = []
        gen_start_xy = []
        prev_yaws = [0.0] * NUM_ROBOTS
        accumulated_yaws = [0.0] * NUM_ROBOTS
        for i, r_id in enumerate(robots):
            pos, orn = p.getBasePositionAndOrientation(r_id, physicsClientId=client_id)
            gen_start_positions.append(list(pos))
            gen_start_xy.append(np.array(pos[:2]))
            _, _, yaw = p.getEulerFromQuaternion(orn)
            prev_yaws[i] = yaw

        steps = int(DURATION * PHYSICS_FREQ)
        for s in range(steps):
            t = s / PHYSICS_FREQ

            if s % servo_interval == 0:
                for i, r_id in enumerate(robots):
                    dna = population[i]
                    cycle_phase = (t * dna["frequency"]) % 1.0

                    for l_idx, prefix in enumerate(LEG_PREFIXES):
                        phase = leg_phase_offset(mode, l_idx, cycle_phase, gait_offsets)

                        s_phase = (phase if phase < 0.5 else phase - 0.5) * 2.0
                        cycloid_factor = s_phase - (math.sin(2.0 * math.pi * s_phase) / (2.0 * math.pi))

                        if phase < 0.5:  # Swing
                            phase_multiplier = (1.0 - 2.0 * cycloid_factor)
                            tz = body_height + step_height * 0.5 * (1.0 - math.cos(2.0 * math.pi * s_phase))
                        else:            # Stance
                            phase_multiplier = (-1.0 + 2.0 * cycloid_factor)
                            tz = body_height

                        tx, ty, tz = foot_target(
                            mode, prefix, phase_multiplier, tz, dna, direction
                        )

                        h, th, kn = solve_leg_ik_3dof(tx, ty, tz, body_height)

                        if h is not None:
                            is_left = prefix.endswith("L")
                            is_right = prefix.endswith("R")

                            actual_th = -th if is_left else th
                            actual_kn = -kn if is_left else kn
                            if is_right:
                                actual_th = th

                            p.setJointMotorControl2(
                                r_id, joint_map[f"{prefix}3"],
                                p.POSITION_CONTROL, h,
                                force=GLOBAL_MAX_TORQUE,
                                maxVelocity=GLOBAL_MAX_VELOCITY
                            )
                            p.setJointMotorControl2(
                                r_id, joint_map[f"{prefix}2"],
                                p.POSITION_CONTROL, actual_th,
                                force=GLOBAL_MAX_TORQUE,
                                maxVelocity=GLOBAL_MAX_VELOCITY
                            )
                            p.setJointMotorControl2(
                                r_id, joint_map[f"{prefix}1"],
                                p.POSITION_CONTROL, actual_kn,
                                force=GLOBAL_MAX_TORQUE,
                                maxVelocity=GLOBAL_MAX_VELOCITY
                            )

            p.stepSimulation(physicsClientId=client_id)
            if PYBULLET_CONNECT_MODE == p.GUI:
                time.sleep(1.0 / PHYSICS_FREQ)

            if mode == "spin":
                for i, r_id in enumerate(robots):
                    _, orn = p.getBasePositionAndOrientation(r_id, physicsClientId=client_id)
                    _, _, yaw = p.getEulerFromQuaternion(orn)
                    diff = yaw - prev_yaws[i]
                    while diff > math.pi:  diff -= 2.0 * math.pi
                    while diff < -math.pi: diff += 2.0 * math.pi
                    accumulated_yaws[i] += diff
                    prev_yaws[i] = yaw

        fitnesses = []
        for i, r_id in enumerate(robots):
            pos, orn = p.getBasePositionAndOrientation(r_id, physicsClientId=client_id)
            roll, pitch, _ = p.getEulerFromQuaternion(orn)

            if mode == "spin":
                score, metrics = compute_spin_score(
                    accumulated_yaws[i], orn, pos, gen_start_xy[i],
                    target_vel, direction, population[i]
                )
            else:
                score, metrics = compute_translational_score(
                    mode, gen_start_positions[i], pos, roll, pitch,
                    target_vel, orn, population[i], direction
                )

            if pos[2] < (body_height * 0.5) or abs(roll) > 1.2:
                score *= 0.001

            fitnesses.append(score)

            if score > best_overall_score:
                best_overall_score = score
                best_overall_dna = copy.deepcopy(population[i])
                best_overall_metrics = copy.deepcopy(metrics)

        print(f"[{mode} exp {exp_idx}] Gen {gen+1}/{GAIT_TRIALS}  "
              f"best={max(fitnesses):.3f}  mean={np.mean(fitnesses):.3f}  "
              f"all-time-best={best_overall_score:.3f}")

        if gen < GAIT_TRIALS - 1:
            population = next_generation(population, fitnesses, rng)

    p.disconnect(client_id)

    return {
        "mode":          mode,
        "exp_id":        exp_idx,
        "gait":          gait_name,
        "target_vel":    target_vel,
        "target_height": body_height,
        "step_height":   step_height,
        "direction":     direction,
        "score":         best_overall_score,
        **best_overall_dna,
        **best_overall_metrics,
    }

# ============================================================
# 8. EXECUTE
# ============================================================

if __name__ == "__main__":
    MAX_WORKERS = multiprocessing.cpu_count() - 1

    print(f"\n=== Running GA optimization for modes: {MODES}  (render: {RENDER_MODE}) ===")

    # Build the full task list up-front across ALL modes,
    # then run everything in one parallel pool.
    #   - Spin uses TARGET_ROT_SPEEDS (rad/s); the others use TARGET_SPEEDS (m/s).
    #   - Diagonal uses DIAGONAL_DIRECTIONS (degrees); the others use DIRECTIONS (+1/-1).
    #   - An empty direction list skips its associated modes.
    tasks = []
    skipped = []
    for mode in MODES:
        speeds_for_mode = TARGET_ROT_SPEEDS if mode == "spin" else TARGET_SPEEDS
        dirs_for_mode   = DIAGONAL_DIRECTIONS if mode == "diagonal" else DIRECTIONS

        if not dirs_for_mode:
            skipped.append(mode)
            continue

        for gait_name, offsets in GAIT_LIBRARY.items():
            for t_vel in speeds_for_mode:
                for b_height in BODY_HEIGHTS:
                    for step_height in STEP_HEIGHTS:
                        for direction in dirs_for_mode:
                            for i in range(NUM_EXPERIMENTS):
                                tasks.append((
                                    mode, gait_name, offsets, t_vel,
                                    b_height, step_height, direction, i
                                ))

    if skipped:
        print(f">>> Skipping (empty direction list): {skipped}")
    print(f">>> Total experiments: {len(tasks)}  |  workers: {MAX_WORKERS}")

    if not tasks:
        raise SystemExit("No experiments to run — check DIRECTIONS / DIAGONAL_DIRECTIONS / MODES.")

    all_results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        all_results = list(
            tqdm(
                executor.map(run_experiment_worker, tasks),
                total=len(tasks),
                desc="GA Runs"
            )
        )

    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv("gait_results.csv", index=False)
        print("\nAll experiments complete. Results saved to gait_results.csv.")
