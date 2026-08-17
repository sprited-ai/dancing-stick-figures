import sys; sys.path.insert(0,'scratch')
from PIL import Image, ImageDraw
import math
SS,S=4,128; W=S*SS; BG=(255,255,255,255)
C=dict(torso=(40,40,40),arm_L=(232,64,48),fore_L=(255,150,40),arm_R=(40,110,230),fore_R=(80,200,240),
       leg_L=(200,50,160),shin_L=(255,120,200),leg_R=(30,150,90),shin_R=(120,220,90))
def rgba(c): return (*c,255)
def circ(d,p,r,col): d.ellipse([p[0]-r,p[1]-r,p[0]+r,p[1]+r],fill=col)
def draw(pose,sw=4,hands="blob"):
    im=Image.new("RGBA",(W,W),BG); d=ImageDraw.Draw(im)
    j={k:(x*SS,y*SS) for k,(x,y) in pose.items()}
    chains=[[("leg_R","hip_R","knee_R"),("shin_R","knee_R","ankle_R")],
            [("arm_R","shoulder_R","elbow_R"),("fore_R","elbow_R","wrist_R")],
            [("torso","pelvis","neck")],
            [("leg_L","hip_L","knee_L"),("shin_L","knee_L","ankle_L")],
            [("arm_L","shoulder_L","elbow_L"),("fore_L","elbow_L","wrist_L")]]
    w=sw*SS
    def extremity(k,a,b,col,halo):
        ax,ay=j[a]; bx,by=j[b]; dx,dy=bx-ax,by-ay; L=math.hypot(dx,dy) or 1; ux,uy=dx/L,dy/L
        if k.startswith("shin"):  # foot: short segment forward (+x for L, -x for R... just +x here)
            fx=bx+7*SS*(1 if k.endswith("L") else -1); fy=by
            d.line([(bx,by),(fx,fy)],fill=BG if halo else col,width=w+(3*SS if halo else 0))
            circ(d,(fx,fy),w/2+(1.5*SS if halo else 0),BG if halo else col)
        elif k.startswith("fore"):
            hx,hy=bx+ux*3*SS,by+uy*3*SS
            if hands=="blob":
                circ(d,(hx,hy),3.2*SS+(1.5*SS if halo else 0),BG if halo else col)
            elif hands=="mitten":
                # palm blob + 3 fingers fanning along direction
                circ(d,(hx,hy),2.5*SS+(1.5*SS if halo else 0),BG if halo else col)
                px,py=-uy,ux
                for f in (-1,0,1):
                    fx=hx+ux*5*SS+px*f*3*SS; fy=hy+uy*5*SS+py*f*3*SS
                    d.line([(hx+px*f*2*SS,hy+py*f*2*SS),(fx,fy)],fill=BG if halo else col,width=2*SS+(2*SS if halo else 0))
    for ch in chains:
        for halo in (True,False):
            for k,a,b in ch:
                col=rgba(C[k])
                d.line([j[a],j[b]],fill=BG if halo else col,width=w+(3*SS if halo else 0))
                for p in (j[a],j[b]): circ(d,p,w/2+(1.5*SS if halo else 0),BG if halo else col)
                extremity(k,a,b,col,halo)
    hx,hy=j["head"]; r=9*SS; circ(d,(hx,hy),r+2*SS,BG); circ(d,(hx,hy),r,rgba(C["torso"]))
    return im.resize((S,S),Image.BOX)
walk=dict(head=(64,20),neck=(64,34),pelvis=(62,70),shoulder_L=(64,38),elbow_L=(76,52),wrist_L=(84,64),
 shoulder_R=(64,38),elbow_R=(54,52),wrist_R=(46,64),hip_L=(62,70),knee_L=(72,90),ankle_L=(80,110),
 hip_R=(62,70),knee_R=(54,90),ankle_R=(46,110))
jack=dict(head=(64,18),neck=(64,32),pelvis=(64,68),shoulder_L=(64,36),elbow_L=(46,26),wrist_L=(30,14),
 shoulder_R=(64,36),elbow_R=(82,26),wrist_R=(98,14),hip_L=(60,68),knee_L=(48,88),ankle_L=(38,110),
 hip_R=(68,68),knee_R=(80,88),ankle_R=(90,110))
sheet=Image.new("RGBA",(S*4,S),BG)
for i,(p,h) in enumerate([(walk,"none"),(walk,"blob"),(walk,"mitten"),(jack,"mitten")]):
    sheet.paste(draw(p,hands=h),(i*S,0))
sheet.save("scratch/hands_probe_128.png"); sheet.resize((S*12,S*3),Image.NEAREST).save("scratch/hands_probe_128_x3.png")
