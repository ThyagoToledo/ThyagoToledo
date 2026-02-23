"""
Gargantua animated GIF generator.
Produces a physically inspired black hole accretion-disk animation
(simplified ray-casting with Doppler beaming) using only Pillow + NumPy.
Output: gargantua_preview.gif   (~800x450, 48 frames, looping)
"""

import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# ── canvas ────────────────────────────────────────────────────────────────────
W, H       = 800, 450
CX, CY     = W // 2, H // 2
BH_R       = 78          # black hole shadow radius (px)
N_FRAMES   = 48
FRAME_MS   = 55          # ms per frame  ≈ ~18 fps

# ── pre-compute static grids ──────────────────────────────────────────────────
yg, xg  = np.mgrid[0:H, 0:W]
dx      = (xg - CX).astype(np.float32)
dy      = (yg - CY).astype(np.float32)
dist    = np.sqrt(dx**2 + dy**2)
angle   = np.arctan2(dy, dx)

# normalised signed angle from horizontal equator (vertical separation / dist)
# used to distinguish "above" vs "below" the BH mid-plane
with np.errstate(divide='ignore', invalid='ignore'):
    sin_lat = np.where(dist > 0, dy / dist, 0.0)

# ── star background ───────────────────────────────────────────────────────────
rng         = np.random.default_rng(7)
star_prob   = rng.random((H, W))
star_bright = rng.uniform(60, 255, (H, W)).astype(np.float32)
star_r      = np.where(star_prob > 0.9965, star_bright,        0.0)
star_g      = np.where(star_prob > 0.9965, star_bright * 0.95, 0.0)
star_b      = np.where(star_prob > 0.9965, star_bright,        0.0)

# ── helper: soft clamp ────────────────────────────────────────────────────────
def clamp(arr, lo=0.0, hi=255.0):
    return np.clip(arr, lo, hi).astype(np.uint8)


def disk_color(phase: float):
    """
    Returns (R, G, B) float32 arrays for the accretion-disk layer at the
    given rotation phase (0 .. 2π).

    Visual model:
      - Primary disk image  : wide horizontal band, bright, Doppler-shifted
      - Secondary (lensed)  : thin bright arc that traces the top & bottom
                               of the photon sphere
      - Photon ring         : hot white-gold ring just outside the shadow
    """
    # ── Doppler beaming ───────────────────────────────────────────────────────
    # Left side of disk approaches → brighter; right recedes → dimmer
    rot_angle   = angle + phase                         # co-rotate frame
    doppler     = np.cos(rot_angle)                     # [-1 … +1]
    # intensity multiplier: approaching × 3, receding × 0.4
    doppler_I   = np.where(doppler > 0,
                           1.0 + 2.0 * doppler,
                           1.0 + 0.6 * doppler)

    # ── radial profile (Gaussian centred on r≈150 px) ────────────────────────
    r_peak      = 148.0
    r_sigma     = 68.0
    radial      = np.exp(-0.5 * ((dist - r_peak) / r_sigma) ** 2)

    # ── primary disk : thin horizontal slab ──────────────────────────────────
    prim_lat_sigma = 18.0
    prim_lat       = np.exp(-0.5 * (dy / prim_lat_sigma) ** 2)
    prim_mask      = (dist > BH_R * 0.90) & (dist < 260)

    prim_I         = radial * prim_lat * doppler_I * prim_mask

    # ── secondary (lensed) image : arc above & below ─────────────────────────
    # Appears near the photon ring radius = 1.5 × BH_R, latitude band ≈ 20-50 px
    r_sec      = BH_R * 1.45
    sec_r_sig  = 28.0
    sec_lat_lo = 14.0
    sec_lat_hi = 55.0
    sec_radial = np.exp(-0.5 * ((dist - r_sec) / sec_r_sig) ** 2)
    sec_lat    = ((np.abs(dy) > sec_lat_lo) & (np.abs(dy) < sec_lat_hi)).astype(np.float32)
    # secondary appears only outside the shadow
    sec_mask   = dist > BH_R + 5
    sec_I      = sec_radial * sec_lat * doppler_I * 0.55 * sec_mask

    # ── photon ring ───────────────────────────────────────────────────────────
    r_ph       = BH_R + 7.0
    ph_sigma   = 5.0
    phot_I     = np.exp(-0.5 * ((dist - r_ph) / ph_sigma) ** 2) * 1.3

    # ── combine ───────────────────────────────────────────────────────────────
    total_I    = prim_I + sec_I + phot_I

    # Colour: warm orange → white core → deep red rim  (temperature-like)
    rr = total_I * 255.0
    gg = total_I * 145.0
    bb = total_I *  30.0

    # Extra blue tint on the approaching (beamed) side
    bb = bb + np.where(doppler > 0, total_I * doppler * 55.0, 0.0)

    return rr, gg, bb


# ── render frames ─────────────────────────────────────────────────────────────
frames = []
print(f"Rendering {N_FRAMES} frames …")

for fi in range(N_FRAMES):
    if fi % 8 == 0:
        print(f"  frame {fi+1}/{N_FRAMES}")

    phase = 2.0 * math.pi * fi / N_FRAMES

    rr, gg, bb = disk_color(phase)

    # Composite over star background
    R = star_r + rr
    G = star_g + gg
    B = star_b + bb

    img_arr = np.stack([clamp(R), clamp(G), clamp(B)], axis=2)
    img     = Image.fromarray(img_arr, 'RGB')

    # ── glow pass ─────────────────────────────────────────────────────────────
    glow = img.filter(ImageFilter.GaussianBlur(radius=5))
    # blend: 55 % sharp + 45 % blurred
    img  = Image.blend(img, glow, alpha=0.45)

    # ── black hole shadow ─────────────────────────────────────────────────────
    draw = ImageDraw.Draw(img)
    x0, y0  = CX - BH_R, CY - BH_R
    x1, y1  = CX + BH_R, CY + BH_R
    draw.ellipse([x0, y0, x1, y1], fill=(0, 0, 0))

    # inner shadow gradient (soft edge via concentric circles)
    for δ in range(6):
        alpha_val = int(255 * (1 - δ/6))
        r_i = BH_R + δ * 2
        draw.ellipse(
            [CX - r_i, CY - r_i, CX + r_i, CY + r_i],
            outline=(0, 0, 0, alpha_val),
            width=2
        )

    # ── UI label ──────────────────────────────────────────────────────────────
    draw.text((28, H - 62), "G A R G A N T U A",
              fill=(255, 245, 220, 200))
    draw.text((28, H - 38), "Renderização Granular • Lente Gravitacional Física",
              fill=(200, 170, 110, 155))

    frames.append(img.convert('P', palette=Image.ADAPTIVE, colors=256))

# ── save GIF ──────────────────────────────────────────────────────────────────
output = "gargantua_preview.gif"
frames[0].save(
    output,
    save_all=True,
    append_images=frames[1:],
    optimize=False,
    duration=FRAME_MS,
    loop=0,
)
print(f"\n✔ Saved → {output}")
