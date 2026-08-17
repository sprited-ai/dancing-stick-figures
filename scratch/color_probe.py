"""Color-coded probe. Chain-halo (not per-bone). 128px."""
from PIL import Image, ImageDraw
SS, S = 4, 128; W = S*SS
BG = (255,255,255,255); INK=(20,20,20,255)
# OpenPose-ish but higher-contrast, distinct at 128px, L=warm R=cool
C = dict(torso=(40,40,40), head=(40,40,40),
         arm_L=(232,64,48), fore_L=(255,150,40),     # red / orange
         arm_R=(40,110,230), fore_R=(80,200,240),    # blue / cyan
         leg_L=(200,50,160), shin_L=(255,120,200),   # magenta / pink
         leg_R=(30,150,90), shin_R=(120,220,90))     # green / lime
def rgba(c): return (*c,255)

def draw(pose, halo=True, sw=4, colored=True):
    im = Image.new("RGBA",(W,W),BG); d = ImageDraw.Draw(im)
    j = {k:(x*SS,y*SS) for k,(x,y) in pose.items()}
    # chains back-to-front; each chain = list of (bone_key, a, b)
    chains = [
        [("leg_R","hip_R","knee_R"),("shin_R","knee_R","ankle_R")],
        [("arm_R","shoulder_R","elbow_R"),("fore_R","elbow_R","wrist_R")],
        [("torso","pelvis","neck")],
        [("leg_L","hip_L","knee_L"),("shin_L","knee_L","ankle_L")],
        [("arm_L","shoulder_L","elbow_L"),("fore_L","elbow_L","wrist_L")],
    ]
    w = sw*SS
    for ch in chains:
        if halo:  # whole chain first, so shared joints don't get nicked
            for _,a,b in ch:
                d.line([j[a],j[b]],fill=BG,width=w+3*SS)
                for p in (j[a],j[b]): d.ellipse([p[0]-w/2-1.5*SS,p[1]-w/2-1.5*SS,p[0]+w/2+1.5*SS,p[1]+w/2+1.5*SS],fill=BG)
        for k,a,b in ch:
            col = rgba(C[k]) if colored else INK
            d.line([j[a],j[b]],fill=col,width=w)
            for p in (j[a],j[b]): d.ellipse([p[0]-w/2,p[1]-w/2,p[0]+w/2,p[1]+w/2],fill=col)
    hx,hy=j["head"]; r=9*SS
    if halo: d.ellipse([hx-r-2*SS,hy-r-2*SS,hx+r+2*SS,hy+r+2*SS],fill=BG)
    d.ellipse([hx-r,hy-r,hx+r,hy+r],fill=rgba(C["head"]) if colored else INK)
    return im.resize((S,S),Image.BOX)

tpose=dict(head=(64,22),neck=(64,36),pelvis=(64,72),shoulder_L=(64,40),elbow_L=(44,40),wrist_L=(24,40),
 shoulder_R=(64,40),elbow_R=(84,40),wrist_R=(104,40),hip_L=(60,72),knee_L=(56,92),ankle_L=(54,112),
 hip_R=(68,72),knee_R=(72,92),ankle_R=(74,112))
floss=dict(head=(66,22),neck=(64,36),pelvis=(58,72),shoulder_L=(64,40),elbow_L=(76,52),wrist_L=(80,66),
 shoulder_R=(64,40),elbow_R=(50,54),wrist_R=(76,60),hip_L=(54,72),knee_L=(50,92),ankle_L=(50,112),
 hip_R=(62,72),knee_R=(70,92),ankle_R=(76,112))
walk=dict(head=(64,20),neck=(64,34),pelvis=(62,70),shoulder_L=(64,38),elbow_L=(76,52),wrist_L=(84,64),
 shoulder_R=(64,38),elbow_R=(54,52),wrist_R=(46,64),hip_L=(62,70),knee_L=(72,90),ankle_L=(80,110),
 hip_R=(62,70),knee_R=(54,90),ankle_R=(46,110))
sheet=Image.new("RGBA",(S*4,S),BG)
for i,(p,c) in enumerate([(tpose,False),(tpose,True),(floss,True),(walk,True)]):
    sheet.paste(draw(p,colored=c),(i*S,0))
sheet.save("scratch/color_probe_128.png")
sheet.resize((S*12,S*3),Image.NEAREST).save("scratch/color_probe_128_x3.png")
