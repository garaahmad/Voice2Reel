"""Silent, hand-authored animation study; not an AI-generated narrative demo."""

import argparse
import math
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from voice2reel.render import find_font


def ease(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


def mix(a, b, u):
    return a + (b - a) * u


def joint(root, target, upper, lower, bend=1):
    """Two-bone IK keeps limb lengths fixed while hands and feet meet targets."""
    dx, dy = target[0] - root[0], target[1] - root[1]
    distance = max(abs(upper - lower) + 1e-6, min(math.hypot(dx, dy), upper + lower - 1e-6))
    angle = math.atan2(dy, dx)
    offset = math.acos(max(-1, min(1, (upper**2 + distance**2 - lower**2) / (2 * upper * distance))))
    elbow = (root[0] + upper * math.cos(angle + bend * offset),
             root[1] + upper * math.sin(angle + bend * offset))
    end = (root[0] + distance * math.cos(angle), root[1] + distance * math.sin(angle))
    return elbow, end


class MotionStudy:
    duration = 9.0

    def __init__(self, width=540):
        if width < 270 or width % 2:
            raise ValueError("Use an even width >=270")
        self.width, self.height = width, round(width * 16 / 9 / 2) * 2
        self.scale = width / 540
        self.font = ImageFont.truetype(str(find_font()), 30)
        self.small = ImageFont.truetype(str(find_font()), 17)
        self.ink, self.teal, self.orange = '#203641', '#138c85', '#f29954'
        self.background = Image.new('RGB', (540, 960), '#f4f1e9')
        d = ImageDraw.Draw(self.background)
        # Architecture stays quiet so the character and action carry the shot.
        d.rounded_rectangle((40, 250, 500, 757), radius=45, fill='#e5eae1')
        d.ellipse((280, 330, 630, 680), fill='#d2e2d8')
        d.line((0, 758, 540, 758), fill='#c5cec5', width=2)
        for x in range(15, 540, 60):
            d.line((x, 782, x + 30, 782), fill='#dde1d8', width=2)

    def state(self, t):
        # Each action has anticipation, contact and recovery instead of a pose switch.
        walk = min(t / 2.5, 1)
        x = mix(-50, 210, ease(walk))
        stride = math.sin(t * math.tau * 1.8) * (1 - ease((t - 1.8) / .7))
        y = 0.0
        reach = ease((t - 2.65) / .8) * (1 - ease((t - 4.0) / .45))
        press = ease((t - 3.35) / .17) * (1 - ease((t - 3.62) / .3))
        surprise = ease((t - 3.65) / .25) * (1 - ease((t - 4.5) / .6))
        anticipation = ease((t - 5.05) / .25) * (1 - ease((t - 5.3) / .12))
        jump = max(0, math.sin(max(0, min(1, (t - 5.32) / .85)) * math.pi))
        settle = math.sin(max(t - 6.17, 0) * 21) * math.exp(-max(t - 6.17, 0) * 7)
        celebrate = ease((t - 5.4) / .3)
        y += 16 * anticipation - 98 * jump + 9 * settle
        return dict(x=x, y=y, stride=stride, reach=reach, press=press, surprise=surprise,
                    anticipation=anticipation, jump=jump, celebrate=celebrate)

    def frame(self, t):
        if not math.isfinite(t) or not 0 <= t <= self.duration:
            raise ValueError("Time outside preview")
        s = self.state(t)
        image = self.background.copy()
        d = ImageDraw.Draw(image)

        def line(points, width=10, fill=None):
            fill = fill or self.ink
            d.line(points, fill=fill, width=width, joint='curve')
            r = width / 2
            for x, y in points:
                d.ellipse((x-r, y-r, x+r, y+r), fill=fill)

        # A prop reacts exactly when the hand makes contact.
        d.ellipse((337, 742, 473, 766), fill='#c8d2c9')
        d.rounded_rectangle((365, 646, 445, 753), radius=15, fill='#b8c9bd')
        top = 632 + 9 * s['press']
        d.rounded_rectangle((350, top, 460, 659 + 6*s['press']), radius=14,
                            fill=self.orange if t < 3.5 else self.teal)
        d.rounded_rectangle((361, top+5, 449, top+11), radius=3, fill='#ffc894' if t < 3.5 else '#61b5a8')
        # The reveal grows out of the button rather than appearing as a new card.
        reveal = ease((t - 3.52) / .65)
        if reveal > 0:
            radius = 48 * reveal * (1 + .10*math.sin((t-3.52)*12)*math.exp(-(t-3.52)*3))
            cy = mix(610, 451, reveal)
            d.ellipse((405-radius, cy-radius, 405+radius, cy+radius), fill=self.teal)
            if reveal > .6:
                line([(385, cy), (400, cy+15), (428, cy-18)], 7, '#ffffff')
            for k in range(8):
                a = k * math.tau/8
                r = 54 + 62 * ease((t - 3.6) / .6)
                length = 18 * (1 - ease((t - 4.0) / .8))
                if length > .2:
                    line([(405+r*math.cos(a), cy+r*math.sin(a)),
                          (405+(r+length)*math.cos(a), cy+(r+length)*math.sin(a))], 4, self.orange)

        x = s['x'] + 20*s['reach'] - 13*s['surprise']
        hip = (x, 650 + s['y'] + abs(s['stride'])*3)
        shoulder = (x + 50*s['reach'] - 12*s['surprise'], hip[1]-100 + 10*s['anticipation'])
        head = (shoulder[0]+3*s['reach']-4*s['surprise'], shoulder[1]-64)
        # Feet remain planted in the settled pose; the gait has separate lift and swing.
        feet = [(s['x']-24 + 33*s['stride'], 753-20*max(s['stride'], 0)-110*s['jump']),
                (s['x']+24 - 33*s['stride'], 753-20*max(-s['stride'], 0)-110*s['jump'])]
        shadow_width = 52 * (1-.32*s['jump'])
        d.ellipse((x-shadow_width, 746, x+shadow_width, 761), fill='#c4cec4')
        for i, foot in enumerate(feet):
            knee, ankle = joint(hip, foot, 62, 61, 1 if i == 0 else -1)
            line([hip, knee, ankle], 11, '#47626a' if i == 0 else self.ink)
            line([(ankle[0]-5, ankle[1]), (ankle[0]+13, ankle[1])], 12)
        line([hip, shoulder], 13)
        idle_left = (x-43-14*s['stride'], shoulder[1]+84)
        idle_right = (x+44+14*s['stride'], shoulder[1]+83)
        # Reach targets and elbow bending are independent of the head/body animation.
        right = (mix(idle_right[0], 393, s['reach']), mix(idle_right[1], top+7, s['reach']))
        left = idle_left
        u = s['surprise']
        left = (mix(left[0], head[0]-48, u), mix(left[1], head[1]+8, u))
        right = (mix(right[0], head[0]+50, u), mix(right[1], head[1]+8, u))
        c = s['celebrate']
        left = (mix(left[0], x-62, c), mix(left[1], shoulder[1]-73, c))
        right = (mix(right[0], x+65, c), mix(right[1], shoulder[1]-73, c))
        for hand, bend in [(left, 1), (right, -1)]:
            elbow, end = joint(shoulder, hand, 68, 72, bend)
            line([shoulder, elbow, end], 10)
            d.ellipse((end[0]-7, end[1]-7, end[0]+7, end[1]+7), fill=self.ink)
        # Squash, blink, eye direction and mouth changes make the face participate.
        squash = 1 - .07*s['anticipation'] + .04*s['jump']
        hx, hy = head
        d.ellipse((hx-37/squash, hy-39*squash, hx+37/squash, hy+39*squash),
                  fill='#fffefa', outline=self.ink, width=7)
        blink = max(0, 1-abs(((t+1.3) % 3.4)-1.7)/.07)
        look = 5 * (1-ease((t-6.8)/.5))
        for eye in [-12, 12]:
            ex, ey = hx+eye+look, hy-3
            if blink > .2:
                line([(ex-4, ey), (ex+4, ey)], 3)
            else:
                d.ellipse((ex-3.3, ey-5, ex+3.3, ey+5), fill=self.ink)
        if s['surprise'] > .4:
            d.ellipse((hx+1, hy+12, hx+13, hy+27), outline=self.ink, width=3)
        else:
            d.arc((hx-12, hy+7, hx+17, hy+28), 10, 165, fill=self.ink, width=3)
        # A mild continuous camera push is applied only to the stage, keeping titles readable.
        zoom = 1 + .045*ease(t / self.duration)
        camera = (270, 590)
        inv = 1/zoom
        image = image.transform((540, 960), Image.Transform.AFFINE,
                                (inv, 0, camera[0]*(1-inv), 0, inv, camera[1]*(1-inv)),
                                resample=Image.Resampling.BICUBIC)
        d = ImageDraw.Draw(image)
        d.text((270, 81), 'Voice2Reel', font=self.small, fill=self.teal, anchor='mt')
        d.text((270, 125), 'حركة تحكي الفكرة', font=self.font, fill=self.ink, anchor='mt', direction='rtl')
        captions = 'خطوة، ثم فكرة' if t < 2.6 else 'لمسة لها أثر' if t < 4.8 else 'ردّة فعل، وشخصية'
        d.text((270, 841), captions, font=self.font, fill=self.ink, anchor='mt', direction='rtl')
        if self.width != 540:
            image = image.resize((self.width, self.height), Image.Resampling.LANCZOS)
        return image


def render_preview(output, width=540, fps=30):
    if not 1 <= fps <= 60:
        raise ValueError('FPS must be 1–60')
    rig = MotionStudy(width)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.stem + '.partial.mp4')
    process = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                                '-s', f'{rig.width}x{rig.height}', '-r', str(fps), '-i', '-', '-an',
                                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '19', '-threads', '2',
                                '-movflags', '+faststart', str(temp)], stdin=subprocess.PIPE)
    try:
        for n in range(round(rig.duration*fps)):
            process.stdin.write(rig.frame(n/fps).tobytes())
        process.stdin.close()
        if process.wait():
            raise RuntimeError('ffmpeg could not encode the motion preview')
        temp.replace(output)
    except BaseException:
        process.kill()
        process.wait()
        temp.unlink(missing_ok=True)
        raise
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='outputs/stickman-motion-study.mp4')
    p.add_argument('--width', type=int, default=540)
    p.add_argument('--fps', type=int, default=30)
    args = p.parse_args()
    print(render_preview(args.output, args.width, args.fps))


if __name__ == '__main__':
    main()
