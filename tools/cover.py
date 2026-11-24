"""Forgotten East reel covers (thumbnails), competitor style.

Every reel gets a designed 1080x1920 cover: bold 2-line Anton headline, the
country lifted in 3D and glowing, dark high-contrast map, everything important
inside the centre 1080x1440 band (the 3:4 crop that IG/FB profile grids show).

Usage (CLI):
  python3 cover.py "Laos" "WHY IS IT" "CALLED LAOS?" cover.jpg
  python3 cover.py "Thailand" "IT'S NOT 2026" "IT'S 2569" cover.jpg --accent ember
  python3 cover.py "United States of America" "NAMED AFTER" "AN ITALIAN?" cover.jpg --view -97,39,900

Then:
  * Metricool post: set info.videoThumbnailUrl to a public URL of cover.jpg
    (push it next to the MP4 in the forgotten-east-media repo, raw URL), and do
    NOT also send videoCoverMilliseconds.
  * Make the first frame strong too:  python3 cover.py --intro video.mp4 cover.jpg out.mp4
    (cover is on screen at frame 0 and dissolves into the animation by 0.8 s;
    audio untouched).

Country names are Natural Earth ADMIN names (e.g. "United States of America").
Needs fx.py assets in /home/claude/w/assets (Anton, Inter, LibreBaskerville,
countries.geojson, countries10.geojson).
"""
import sys, math, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from fx import *

ACCENTS = {"gold": GOLD, "ember": (255, 70, 50), "cyan": CYAN, "white": WHITE}
SAFE_TOP, SAFE_BOT = 240, 1680          # 3:4 grid crop band
MAP_CY = 1130                            # where the country sits


def _main_rings(rings):
    def area(r):
        x, y = r[:, 0], r[:, 1]; return abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2
    a = [area(r) for r in rings]; m = max(a)
    return [r for r, ar in zip(rings, a) if ar >= 0.06 * m]


def auto_view(rings, box_w=700, box_h=720):
    main = _main_rings(rings); pts = np.concatenate(main)
    lon = (pts[:, 0].min() + pts[:, 0].max()) / 2; lat = (pts[:, 1].min() + pts[:, 1].max()) / 2
    cam = Cam(lon, lat, 1000, 0, W / 2, MAP_CY)
    xy, _ = cam.proj(pts)
    w = xy[:, 0].max() - xy[:, 0].min(); h = xy[:, 1].max() - xy[:, 1].min()
    R = 1000 * min(box_w / max(w, 1), box_h / max(h, 1))
    return lon, lat, max(380, min(R, 26000))


def fit_text(text, max_w, size, fname="Anton-Regular.ttf", **kw):
    while size > 60:
        im = text_img(text, fname, size, **kw)
        if im.width - 140 <= max_w: return im
        size -= 8
    return text_img(text, fname, size, **kw)


def band(img, y0, y1, strength=0.82, up=True):
    """dark gradient behind the headline so it reads in a tiny grid tile"""
    a = np.asarray(img, np.float32)
    yy = np.arange(H, dtype=np.float32)
    t = np.clip((y1 - yy) / (y1 - y0), 0, 1) if up else np.clip((yy - y0) / (y1 - y0), 0, 1)
    k = 1 - strength * (t ** 1.3)
    return Image.fromarray(np.clip(a * k[:, None, None], 0, 255).astype(np.uint8))


def make_cover(country, top, big, out, accent="gold", view=None, also=(), countries=None):
    C = countries or load_countries()
    rings = C[country]
    lon, lat, R = view if view else auto_view(rings)
    cam = Cam(lon, lat, R, 0, W / 2, MAP_CY)
    fills = {n: (CYAN, 0.35) for n in also}
    img = base_map(cam, C, fills=fills)
    img = lifted_country(img, cam, rings, t=1.0, depth=30 if R > 1500 else 22, color=CYAN, sweep=0.47)
    img = vig(img)
    img = band(img, SAFE_TOP - 60, 900, 0.85, up=True)
    img = band(img, 1500, H, 0.6, up=False)
    col = ACCENTS.get(accent, GOLD)
    # headline: small line white, big line accent with glow; centre of the safe band top
    l1 = fit_text(top, 960, 150, fill=WHITE, stroke=3, stroke_fill=(0, 0, 0), shadow=True)
    l2 = fit_text(big, 980, 230, fill=col, stroke=4, stroke_fill=(20, 8, 0), glow_col=col, shadow=True)
    y1 = SAFE_TOP + 70 + (l1.height - 140) / 2
    y2 = y1 + (l1.height - 140) / 2 + 26 + (l2.height - 140) / 2
    img = paste(img, l1, (W / 2, y1 + 0))
    img = paste(img, l2, (W / 2, y2 + 0))
    # brand stamp, bottom of the safe band
    img = paste(img, text_img("FORGOTTEN EAST", "LibreBaskerville.ttf", 38, (240, 228, 200), 0, (0, 0, 0), None, True),
                (W / 2, SAFE_BOT - 70))
    img = finish(img)
    img = ImageEnhance.Contrast(img).enhance(1.08); img = ImageEnhance.Color(img).enhance(1.15)
    img.save(out, quality=92)
    return out


def add_intro(video, cover, out, hold=0.5, fade=0.3):
    """cover is frame 0; dissolves into the video by hold+fade seconds. Audio copied."""
    fc = (f"[1:v]scale=1080:1920,format=rgba,fade=t=out:st={hold}:d={fade}:alpha=1[c];"
          f"[0:v][c]overlay=0:0:enable='lte(t,{hold + fade})'[v]")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", video, "-loop", "1", "-t", str(hold + fade + 0.1), "-i", cover,
                    "-filter_complex", fc, "-map", "[v]", "-map", "0:a?", "-c:v", "libx264", "-preset", "medium",
                    "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", out], check=True)
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "--intro":
        add_intro(a[1], a[2], a[3]); print("INTRO", a[3]); sys.exit()
    country, top, big, out = a[:4]
    kw = {}
    if "--accent" in a: kw["accent"] = a[a.index("--accent") + 1]
    if "--view" in a: kw["view"] = tuple(float(v) for v in a[a.index("--view") + 1].split(","))
    if "--also" in a: kw["also"] = tuple(a[a.index("--also") + 1].split("|"))
    print("COVER", make_cover(country, top, big, out, **kw))
