"""Appendix B probe: does a stick figure read at 128x128 with the halo trick?
Throwaway. Renders a T-pose and a floss-ish crossing pose, with/without halo."""
import math
from PIL import Image, ImageDraw

SS = 4          # supersample
S = 128
W = S * SS


def fig(pose, halo=True, stroke=4, bg=(255, 255, 255, 255)):
    im = Image.new("RGBA", (W, W), bg)
    d = ImageDraw.Draw(im)
    j = {k: (x * SS, y * SS) for k, (x, y) in pose.items()}
    bones = [  # back-to-front draw order
        ("hip_R", "knee_R"), ("knee_R", "ankle_R"),
        ("shoulder_R", "elbow_R"), ("elbow_R", "wrist_R"),
        ("pelvis", "neck"),
        ("hip_L", "knee_L"), ("knee_L", "ankle_L"),
        ("shoulder_L", "elbow_L"), ("elbow_L", "wrist_L"),
    ]
    sw = stroke * SS
    for a, b in bones:
        if halo:
            d.line([j[a], j[b]], fill=bg, width=sw + 3 * SS)
        d.line([j[a], j[b]], fill=(20, 20, 20, 255), width=sw)
        for p in (j[a], j[b]):
            d.ellipse([p[0]-sw/2, p[1]-sw/2, p[0]+sw/2, p[1]+sw/2], fill=(20, 20, 20, 255))
    hx, hy = j["head"]; r = 9 * SS
    if halo:
        d.ellipse([hx-r-2*SS, hy-r-2*SS, hx+r+2*SS, hy+r+2*SS], fill=bg)
    d.ellipse([hx-r, hy-r, hx+r, hy+r], fill=(20, 20, 20, 255))
    return im.resize((S, S), Image.BOX)


def tpose():
    return dict(head=(64, 22), neck=(64, 36), pelvis=(64, 72),
                shoulder_L=(64, 40), elbow_L=(44, 40), wrist_L=(24, 40),
                shoulder_R=(64, 40), elbow_R=(84, 40), wrist_R=(104, 40),
                hip_L=(60, 72), knee_L=(56, 92), ankle_L=(54, 112),
                hip_R=(68, 72), knee_R=(72, 92), ankle_R=(74, 112))


def floss():
    # arms crossing in front of torso, hips swung
    return dict(head=(66, 22), neck=(64, 36), pelvis=(58, 72),
                shoulder_L=(64, 40), elbow_L=(76, 52), wrist_L=(80, 66),
                shoulder_R=(64, 40), elbow_R=(50, 54), wrist_R=(76, 60),
                hip_L=(54, 72), knee_L=(50, 92), ankle_L=(50, 112),
                hip_R=(62, 72), knee_R=(70, 92), ankle_R=(76, 112))


sheet = Image.new("RGBA", (S * 4, S), (255, 255, 255, 255))
for i, (p, h) in enumerate([(tpose(), False), (tpose(), True), (floss(), False), (floss(), True)]):
    sheet.paste(fig(p, halo=h), (i * S, 0))
sheet.save("scratch/probe_128.png")
sheet.resize((S * 4 * 3, S * 3), Image.NEAREST).save("scratch/probe_128_x3.png")
print("ok")
