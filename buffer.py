import numpy as np
import matplotlib.pyplot as plt

def plot_polar_point(r, theta, save_path="plot.png"):
    """
    Plot a single polar coordinate (r, theta) where theta is in radians.
    Labels for r and theta remain readable even at small angles.
    """
    r, theta = float(r), float(theta)
    theta_deg = np.degrees(theta)

    fig, ax = plt.subplots(figsize=(6, 4), subplot_kw=dict(projection="polar"))
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_thetalim(0, np.pi)

    r_max = max(r * 1.35, 1.0)
    ax.set_ylim(0, r_max)
    ax.set_thetagrids([0, 45, 90, 135, 180])
    ax.set_rlabel_position(90)

    # Vector line + endpoint dot
    ax.plot([0, theta], [0, r], color="navy", lw=2.5, zorder=3)
    ax.scatter([theta], [r], color="navy", s=25, zorder=4)

    # Angle arc
    r_arc = r * 0.25
    theta_arc = np.linspace(0, theta, 100)
    ax.plot(theta_arc, np.full_like(theta_arc, r_arc), color="black", lw=1.2)

    # --- Theta label ---
    # For small angles, push label radially outward to avoid overlap with the line
    label_r = r_arc * (2.5 if theta_deg < 20 else 1.6)
    ax.text(
        theta / 2, label_r, "θ",
        ha="center", va="center", fontsize=11, fontweight="bold"
    )

    # --- R label ---
    # Offset perpendicular to the ray so it doesn't sit on top of the line
    # For small angles, shift label above the line instead of to the side
    r_label_offset = r_max * (0.12 if theta_deg > 20 else 0.18)
    ax.text(
        theta, r * 0.55, "r",
        ha="center", va="bottom", fontsize=11, color="navy", fontweight="bold",
        rotation=theta_deg, rotation_mode="anchor"
    )

    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    # plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# Examples
plot_polar_point(r=2.5, theta=np.radians(10), save_path="small_angle.png")
# plot_polar_point(r=2.5, theta=np.radians(60), save_path="normal_angle.png")