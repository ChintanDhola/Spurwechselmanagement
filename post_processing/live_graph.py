#  Copyright (c) 2021 European Union
#  *
#  Licensed under the EUPL, Version 1.2 or – as soon they will be approved by the
#  European Commission – subsequent versions of the EUPL (the "Licence");
#  You may not use this work except in compliance with the Licence.
#  You may obtain a copy of the Licence at:
#  *
#  https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12
#  *
#  Unless required by applicable law or agreed to in writing,
#  software distributed under the Licence is distributed on an "AS IS" basis,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the Licence for the specific language governing permissions and limitations
#  under the Licence.
#
#

import matplotlib.pyplot as plt

def draw_lanes(ax):
    """
    Draws 3 Lanes from x=-15 to x=300.
    Centered at y=0 (Middle Lane).
    Lane Width = 3.5m
    """
    # X-Range covers the entire plot limit
    x = [-15, 300]

    # Lane 1 (Bottom): y center -3.5 | Bounds: -5.25 to -1.75
    # Lane 2 (Middle): y center  0.0 | Bounds: -1.75 to +1.75
    # Lane 3 (Top):    y center +3.5 | Bounds: +1.75 to +5.25

    # 1. Bottom Road Edge (Solid)
    ax.plot(x, [-5.25, -5.25], 'k-', linewidth=2)

    # 2. Divider 1 (Dashed)
    ax.plot(x, [-1.75, -1.75], 'k--', linewidth=1)

    # 3. Divider 2 (Dashed)
    ax.plot(x, [1.75, 1.75], 'k--', linewidth=1)

    # 4. Top Road Edge (Solid)
    ax.plot(x, [5.25, 5.25], 'k-', linewidth=2)

def plot_map(vehs, ax1, time):
    # --- ADD LANES HERE ---
    draw_lanes(ax1)
    # ----------------------

    if vehs[0].safe:
        colors = ['b', 'g']
    else:
        colors = ['r', 'g']
    # colors = cm.rainbow(np.linspace(0, 1, len(vehs)))

    # ax1.set_xlim(-15, 100)
    ax1.set_xlim(-15, 300)
    ax1.set_ylim(-15, 10)

    for veh, c in zip(vehs, colors):
        xs = veh.pos_profile_long[time]
        ys = veh.pos_profile_lat[time]

        ax1.plot(xs, ys, 'o', c=c)

        shadow_x = [xs - veh.length / 2, xs - veh.length / 2, xs + veh.length / 2, xs + veh.length / 2,
                    xs - veh.length / 2]
        shadow_y = [ys - veh.width / 2, ys + veh.width / 2, ys + veh.width / 2, ys - veh.width / 2, ys - veh.width / 2]

        # ax1.plot(xs, ys, '-', c =c , alpha = 0.5)
        ax1.plot(shadow_x, shadow_y, '-', c=c, alpha=0.5)
    ax1.set_title('Step ' + str(time))
    plt.axis('off')

    plt.pause(0.0000000000000000000000001)
    # plt.pause(0.1)
    ax1.clear()

def plot_map_cout_out(vehs, ax1, time):
    # --- ADD LANES HERE ---
    draw_lanes(ax1)
    # ----------------------

    if vehs[0].safe:
        colors = ['b', 'g', 'k']
    else:
        colors = ['r', 'g', 'k']

    if vehs[1].crash:
        colors[1] = 'r'
        colors[2] = 'r'
    # colors = cm.rainbow(np.linspace(0, 1, len(vehs)))

    # ax1.set_xlim(-15, 100)
    ax1.set_xlim(-15, 300)
    ax1.set_ylim(-15, 10)

    for veh, c in zip(vehs, colors):
        xs = veh.pos_profile_long[time]
        ys = veh.pos_profile_lat[time]

        ax1.plot(xs, ys, 'o', c=c)

        shadow_x = [xs - veh.length / 2, xs - veh.length / 2, xs + veh.length / 2, xs + veh.length / 2,
                    xs - veh.length / 2]
        shadow_y = [ys - veh.width / 2, ys + veh.width / 2, ys + veh.width / 2, ys - veh.width / 2, ys - veh.width / 2]

        # ax1.plot(xs, ys, '-', c =c , alpha = 0.5)
        ax1.plot(shadow_x, shadow_y, '-', c=c, alpha=0.5)
    ax1.set_title('Step ' + str(time))
    plt.axis('off')

    plt.pause(0.0000000000000000000000001)
    # plt.pause(0.1)
    ax1.clear()