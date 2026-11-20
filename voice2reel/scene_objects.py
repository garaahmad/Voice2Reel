"""Deterministic vector props for explanatory scenes. No remote images or code execution."""
import math
from PIL import ImageDraw

def draw_objects(image, objects, seconds, phase, accent="#4b89d1", ink="#18334b"):
    draw = ImageDraw.Draw(image)
    w, h = image.size
    u = w / 540
    t = phase * seconds
    for obj in objects:
        if not obj.start <= t < obj.end:
            continue
        age = t - obj.start
        progress = min(1.0, age / 0.38)
        progress = progress * progress * (3 - 2 * progress)
        pulse = 1 + 0.07 * math.sin(age * 7)
        factor = obj.scale * (progress if obj.animation == "pop_in" else 1)
        if obj.animation == "pulse":
            factor *= pulse
        if factor < 0.01:
            continue
        x = obj.x * w
        y = obj.y * h
        if obj.animation == "slide_in":
            x += (1 - progress) * 90 * u
        k = 110 * u * factor
        lw = max(2, round(4 * u))
        def line(points, color=ink, width=lw):
            draw.line(points, fill=color, width=width, joint="curve")
        def box(x0,y0,x1,y1, fill="#f6fbff", outline=ink, radius=9):
            draw.rounded_rectangle((x+x0*k,y+y0*k,x+x1*k,y+y1*k),
                                   radius=radius*u, fill=fill, outline=outline, width=lw)
        def ellipse(x0,y0,x1,y1, fill=None, outline=ink, width=lw):
            draw.ellipse((x+x0*k,y+y0*k,x+x1*k,y+y1*k),
                         fill=fill, outline=outline, width=width)
        kind = obj.kind
        if kind == "router":
            box(-0.85,-0.20,0.85,0.35,fill="#dce8f2")
            line((x-0.57*k,y-0.2*k,x-0.67*k,y-0.9*k))
            line((x+0.57*k,y-0.2*k,x+0.67*k,y-0.9*k))
            for p in (-0.5,-0.2,0.1):
                ellipse(p,-0.01,p+0.09,0.08,fill=accent,outline=accent)
        elif kind == "ethernet_cable":
            line((x-0.9*k,y-0.5*k,x-0.4*k,y-0.5*k,x-0.4*k,y+0.3*k,x+0.45*k,y+0.3*k),accent,max(3,round(7*u)))
            box(0.40,0.15,0.80,0.46,fill="#b6d7f3")
            for p in (0.49,0.58,0.67):
                line((x+p*k,y+0.15*k,x+p*k,y+0.24*k),width=max(1,round(2*u)))
        elif kind == "wifi_signal":
            for rad in (0.30,0.55,0.80):
                # Arc grows with age for a clear signal-emission effect.
                rr=rad*k*(0.75+0.25*math.sin(age*4)**2)
                draw.arc((x-rr,y-rr,x+rr,y+rr),195,345,fill=accent,width=lw)
            ellipse(-0.07,0.06,0.07,0.20,fill=accent,outline=accent)
        elif kind == "laptop":
            box(-0.75,-0.55,0.75,0.35,fill="#b6d7f3")
            line((x-0.95*k,y+0.45*k,x+0.95*k,y+0.45*k),width=max(lw,round(9*u)))
        elif kind == "phone":
            box(-0.40,-0.80,0.40,0.80,fill="#b6d7f3",radius=14)
            ellipse(-0.05,0.62,0.05,0.72,fill=accent,outline=accent)
        elif kind == "server":
            for q in (-0.62,-0.15,0.32):
                box(-0.68,q,0.68,q+0.38,fill="#dce8f2")
                ellipse(-0.52,q+0.13,-0.42,q+0.23,fill=accent,outline=accent)
        elif kind == "cloud":
            for dx,dy,rr in ((-0.40,0.05,0.32),(0,-0.22,0.46),(0.42,0.03,0.33)):
                ellipse(dx-rr,dy-rr,dx+rr,dy+rr,fill="#dce8f2")
            box(-0.70,0.05,0.70,0.34,fill="#dce8f2")
