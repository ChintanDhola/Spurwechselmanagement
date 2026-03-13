import numpy as np


def calculate_idm_accel(v_ego, v_lead, d_lead, veh_params):

    # 1. Free road term
    if veh_params.IDM_v0 > 0:
        acc_free = veh_params.IDM_a * (1 - (v_ego / veh_params.IDM_v0) ** veh_params.IDM_delta)
    else:
        acc_free = 0

    # 2. Interaction term
    # If d_lead is huge (no car in front), interaction is 0
    if d_lead > 200 or d_lead <= 0:
        # If d_lead <= 0, it means a crash or overlap, we clamp to max braking
        if d_lead <= 0:
            return -veh_params.MOBIL_b_safe
        return acc_free

    delta_v = v_ego - v_lead
    s_star = veh_params.IDM_s0 + v_ego * veh_params.IDM_T + \
             (v_ego * delta_v) / (2 * np.sqrt(veh_params.IDM_a * veh_params.IDM_b))

    acc_interaction = -veh_params.IDM_a * (s_star / d_lead) ** 2

    return acc_free + acc_interaction


def check_safety(ego_veh, cutting_in_veh, speed_log, speed_lat, freq, i):
    # Calculate distance to the "leader" (the cutting-in vehicle)
    d_lead = (cutting_in_veh.pos_profile_long[i] - ego_veh.pos_profile_long[i]) \
             - ego_veh.length / 2 - cutting_in_veh.length / 2

    # Calculate relative speeds
    v_ego = ego_veh.speed_profile_long[i]
    v_lead = cutting_in_veh.speed_profile_long[i]

    # Calculate what acceleration IDM would command in this situation
    accel_induced = calculate_idm_accel(v_ego, v_lead, d_lead, ego_veh)

    # MOBIL Safety condition: Is the required braking harder than b_safe?
    # If accel < -b_safe, it is UNSAFE.
    if accel_induced < -ego_veh.MOBIL_b_safe:
        return False  # Unsafe

    return True  # Safe


def react(ego_veh, speed_log, freq):
    """
    Fallback reaction function required by the framework interface.
    Since IDM is continuous, the main logic happens in movement.py.
    This serves as a backup hard brake.
    """
    return max(speed_log - ego_veh.MOBIL_b_safe / freq, 0), 0


def get_target_distance(veh, v_ego):
    """
    Calculates equilibrium distance for a given speed (s = s0 + v*T).
    Used for setting up initial positions in scenarios.
    """
    return veh.IDM_s0 + v_ego * veh.IDM_T