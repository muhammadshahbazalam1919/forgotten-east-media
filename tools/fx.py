"""Forgotten East advanced map FX (globe dive, lifted 3D country, light sweep,
border comet, split-flap counters, kinetic text). Pillow + numpy only."""
import json, math, random, functools
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageChops

W, H = 1080, 1920
A = "/home/claude/w/assets"
CYAN = (60, 180, 255); EMBER = (255, 50, 40); GOLD = (255, 210, 30); WHITE = (255, 255, 255)
LAND = (46, 52, 44); BORDER = (78, 84, 72); OCEAN = (8, 16, 26)
BRIT = (220, 50, 60); FRAN = (60, 110, 255)


def clamp(t): return max(0.0, min(1.0, t))
def ease(t): t = clamp(t); return t * t * (3 - 2 * t)
def ease_out(t): t = clamp(t); return 1 - (1 - t) ** 3
def back_out(t, s=1.9): t = clamp(t) - 1; return t * t * ((s + 1) * t + s) + 1


@functools.lru_cache(None)
def font(name, size):
    f = ImageFont.truetype(f"{A}/{name}", int(size))
    if name == "Inter.ttf":
        try: f.set_variation_by_name("SemiBold")
        except Exception: pass
    if name == "NotoSansThai.ttf":
        try: f.set_variation_by_name("Bold")
        except Exception: pass
    return f


# ---------------- geo ----------------
def load_countries():
    out = {}
    for fn, near in (("countries.geojson", False), ("countries10.geojson", True)):
        d = json.load(open(f"{A}/{fn}"))
        for f in d["features"]:
            name = f["properties"]["ADMIN"]; g = f["geometry"]
            polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
            rings = [np.array(p[0], dtype=float) for p in polys]
            lon = np.concatenate([r[:, 0] for r in rings]); lat = np.concatenate([r[:, 1] for r in rings])
            inside = (lon.mean() > 85) and (lon.mean() < 115) and (lat.mean() > -5) and (lat.mean() < 30)
            if near == inside:
                out[name] = rings
    return out


class Cam:
    """Orthographic globe camera. R = globe radius in px."""
    def __init__(self, lon, lat, R, rot=0.0, cx=W / 2, cy=H / 2):
        self.lon, self.lat, self.R, self.rot, self.cx, self.cy = lon, lat, R, rot, cx, cy

    def lerp(self, o, t, f=ease):
        t = f(t)
        # zoom interpolates in log space so it feels constant-speed
        R = math.exp(math.log(self.R) + (math.log(o.R) - math.log(self.R)) * t)
        g = lambda a, b: a + (b - a) * t
        return Cam(g(self.lon, o.lon), g(self.lat, o.lat), R, g(self.rot, o.rot), g(self.cx, o.cx), g(self.cy, o.cy))

    def proj(self, lonlat):
        lon = np.radians(lonlat[..., 0]); lat = np.radians(lonlat[..., 1])
        l0 = math.radians(self.lon); p0 = math.radians(self.lat)
        cosc = math.sin(p0) * np.sin(lat) + math.cos(p0) * np.cos(lat) * np.cos(lon - l0)
        x = np.cos(lat) * np.sin(lon - l0)
        y = math.cos(p0) * np.sin(lat) - math.sin(p0) * np.cos(lat) * np.cos(lon - l0)
        back = cosc < 0
        if back.any():  # push hidden points to the horizon
            n = np.hypot(x, y) + 1e-9
            x = np.where(back, x / n, x); y = np.where(back, y / n, y)
        x *= self.R; y *= self.R
        r = math.radians(self.rot); c, s = math.cos(r), math.sin(r)
        X = self.cx + x * c - y * s; Y = self.cy - (x * s + y * c)
        return np.stack([X, Y], -1), cosc

    def p(self, lon, lat):
        xy, _ = self.proj(np.array([[lon, lat]], float)); return tuple(xy[0])


def ring_pts(cam, ring, minstep=1.2):
    xy, cosc = cam.proj(ring)
    if (cosc < 0).all(): return None
    x0, y0, x1, y1 = xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max()
    if x1 < -50 or y1 < -50 or x0 > W + 50 or y0 > H + 50: return None
    if (x1 - x0) < 1.5 and (y1 - y0) < 1.5: return None
    # decimate
    keep = [0]; last = xy[0]
    d = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
    acc = 0.0
    for i in range(1, len(xy)):
        acc += d[i - 1]
        if acc >= minstep: keep.append(i); acc = 0.0
    pts = xy[keep]
    if len(pts) < 3: return None
    return [tuple(p) for p in pts]


# ------------- precomputed layers -------------
def _stars():
    rnd = random.Random(7); im = Image.new("RGB", (W, H), (2, 4, 10)); d = ImageDraw.Draw(im)
    for _ in range(900):
        x, y = rnd.uniform(0, W), rnd.uniform(0, H); b = int(rnd.uniform(60, 255) * rnd.random())
        r = rnd.choice([0.6, 0.8, 1.0, 1.4]); d.ellipse([x - r, y - r, x + r, y + r], fill=(b, b, min(255, b + 20)))
    return im
STARS = _stars()


def _vignette():
    yy, xx = np.mgrid[0:H, 0:W]; dx = (xx - W / 2) / (W / 2); dy = (yy - H / 2) / (H / 2)
    v = np.clip(1 - 0.5 * (dx ** 2 * 0.8 + dy ** 2 * 0.7), 0.3, 1)
    return v[..., None].astype(np.float32)
VIG = _vignette()


def _grain():
    rnd = np.random.default_rng(3); g = rnd.normal(0, 6, (H // 2, W // 2)).astype(np.float32)
    im = Image.fromarray(np.clip(g + 128, 0, 255).astype(np.uint8)).resize((W, H)).filter(ImageFilter.GaussianBlur(1.2))
    return (np.asarray(im, np.float32) - 128)[..., None]
GRAIN = _grain()


def glow(layer, radius, strength=1.0):
    """additive bloom from an RGBA/RGB layer; computed at 1/4 res for speed"""
    small = layer.convert("RGB").resize((W // 4, H // 4), Image.BILINEAR).filter(ImageFilter.GaussianBlur(radius / 4))
    big = small.resize((W, H), Image.BILINEAR)
    if strength != 1.0:
        big = Image.fromarray(np.clip(np.asarray(big, np.float32) * strength, 0, 255).astype(np.uint8))
    return big


def shift(m, dx, dy):
    o = Image.new(m.mode, m.size, 0); o.paste(m, (int(dx), int(dy))); return o


def add(img, layer): return ImageChops.add(img, layer)


# ---------------- map ----------------
def base_map(cam, countries, fills=None, fade=None):
    """fills: {country: (rgb, alpha)}"""
    if cam.R < 1110:
        img = STARS.copy(); d = ImageDraw.Draw(img)
        cx, cy, R = cam.cx, cam.cy, cam.R
        # atmosphere
        atm = Image.new("RGB", (W, H)); da = ImageDraw.Draw(atm)
        da.ellipse([cx - R - 6, cy - R - 6, cx + R + 6, cy + R + 6], outline=(70, 150, 255), width=18)
        img = add(img, glow(atm, 46, 1.6))
        d = ImageDraw.Draw(img)
        d.ellipse([cx - R, cy - R, cx + R, cy + R], fill=OCEAN)
    else:
        img = Image.new("RGB", (W, H), OCEAN); d = ImageDraw.Draw(img)
    fill_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0)); df = ImageDraw.Draw(fill_layer)
    lw = 2 if cam.R > 2500 else 1
    for name, rings in countries.items():
        for ring in rings:
            pts = ring_pts(cam, ring, 1.5 if cam.R > 1500 else 2.5)
            if not pts: continue
            d.polygon(pts, fill=LAND, outline=BORDER, width=lw)
            if fills and name in fills:
                col, a = fills[name]
                if a > 0: df.polygon(pts, fill=col + (int(170 * a),), outline=col + (int(255 * a),), width=3)
    if fills: img.paste(fill_layer, (0, 0), fill_layer)
    if cam.R < 1110:  # terminator shading on the globe
        sh = Image.new("L", (W, H), 0); ds = ImageDraw.Draw(sh)
        cx, cy, R = cam.cx, cam.cy, cam.R
        for k in range(14):
            rr = R * (1 - k / 14); a = int(150 * (1 - k / 14) ** 2)
            ds.ellipse([cx - rr + R * 0.25, cy - rr + R * 0.2, cx + rr + R * 0.25, cy + rr + R * 0.2], fill=0)
        ring = Image.new("L", (W, H), 0); dr = ImageDraw.Draw(ring)
        dr.ellipse([cx - R, cy - R, cx + R, cy + R], fill=255)
        dr.ellipse([cx - R * 0.95 + R * 0.3, cy - R * 0.95 + R * 0.25, cx + R * 0.95 + R * 0.3, cy + R * 0.95 + R * 0.25], fill=0)
        ring = ring.filter(ImageFilter.GaussianBlur(R * 0.12))
        dark = Image.new("RGB", (W, H), (0, 0, 0))
        img = Image.composite(dark, img, ring.point(lambda v: int(v * 0.5)))
    return img


def lifted_country(img, cam, rings, t=1.0, depth=26, color=CYAN, sweep=None, edge=WHITE):
    """3D extruded country raised off the map with a moving light sweep."""
    if t <= 0: return img
    lift = depth * ease_out(t)
    polys = [p for p in (ring_pts(cam, r, 1.2) for r in rings) if p]
    if not polys: return img
    mask = Image.new("L", (W, H), 0); dm = ImageDraw.Draw(mask)
    for p in polys: dm.polygon(p, fill=255)
    # drop shadow
    sh = mask.resize((W // 4, H // 4)).filter(ImageFilter.GaussianBlur(5)).resize((W, H))
    sh = shift(sh, int(lift * 0.6), int(lift * 1.2)).point(lambda v: int(v * 0.8 * t))
    img = Image.composite(Image.new("RGB", (W, H), (0, 0, 0)), img, sh)
    # side walls: stack darker copies rising up to the top face
    L = int(lift); side = tuple(int(c * 0.35) for c in color)
    sidel = Image.new("RGB", (W, H), side)
    for k in range(0, L, 2):
        img.paste(sidel, (0, 0), shift(mask, 0, -k))
    top = shift(mask, 0, -L); steps = 0
    # top face gradient
    yy = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    c0 = np.array(color, np.float32); c1 = np.array([min(255, c * 1.25 + 40) for c in color], np.float32)
    grad = (c1 * (1 - yy) + c0 * yy) * np.ones((1, W, 1), np.float32)
    if sweep is not None:  # diagonal light band
        xx = np.arange(W, dtype=np.float32)[None, :]; yv = np.arange(H, dtype=np.float32)[:, None]
        pos = (xx + yv * 0.6) / (W + H * 0.6)
        band = np.exp(-((pos - sweep) / 0.05) ** 2)[..., None]
        grad = grad + band * 140
    face = Image.fromarray(np.clip(grad, 0, 255).astype(np.uint8))
    img.paste(face, (0, 0), top.point(lambda v: int(v * clamp(t * 1.5))))
    # edge + bloom
    el = Image.new("RGB", (W, H)); de = ImageDraw.Draw(el)
    off = -L
    for p in polys:
        de.line([(x, y + off) for x, y in p] + [(p[0][0], p[0][1] + off)], fill=edge, width=4)
    img = add(img, glow(Image.composite(Image.new("RGB", (W, H), color), Image.new("RGB", (W, H)), top), 40, 0.9 * t))
    img = Image.composite(el, img, el.convert("L").point(lambda v: int(min(255, v) * t)))
    return img


def comet(img, cam, ring, t, color=GOLD, trail=0.12, lift=0):
    """bright point racing around a border with a glowing trail"""
    pts = ring_pts(cam, ring, 1.0)
    if not pts or t <= 0: return img
    pts = np.array(pts); n = len(pts)
    head = int(clamp(t) * (n - 1)); tail = max(0, head - int(trail * n))
    seg = [tuple(p) for p in pts[tail:head + 1]]
    if len(seg) < 2: return img
    seg = [(x, y - lift) for x, y in seg]
    l = Image.new("RGB", (W, H)); d = ImageDraw.Draw(l)
    for i in range(len(seg) - 1):
        a = (i + 1) / len(seg)
        d.line([seg[i], seg[i + 1]], fill=tuple(int(c * a) for c in color), width=7)
    x, y = seg[-1]; d.ellipse([x - 11, y - 11, x + 11, y + 11], fill=WHITE)
    img = add(img, glow(l, 30, 1.4)); return add(img, l)


# ---------------- text ----------------
@functools.lru_cache(256)
def text_img(text, fname, size, fill=WHITE, stroke=0, stroke_fill=(0, 0, 0), glow_col=None, shadow=True):
    f = font(fname, size)
    d0 = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    bb = d0.textbbox((0, 0), text, font=f, stroke_width=stroke)
    pad = 70
    im = Image.new("RGBA", (bb[2] - bb[0] + pad * 2, bb[3] - bb[1] + pad * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    org = (pad - bb[0], pad - bb[1])
    if shadow:
        s = Image.new("RGBA", im.size, (0, 0, 0, 0)); ds = ImageDraw.Draw(s)
        ds.text((org[0] + 6, org[1] + 10), text, font=f, fill=(0, 0, 0, 210), stroke_width=stroke, stroke_fill=(0, 0, 0, 210))
        im = Image.alpha_composite(im, s.filter(ImageFilter.GaussianBlur(9)))
    if glow_col:
        g = Image.new("RGBA", im.size, (0, 0, 0, 0)); dg = ImageDraw.Draw(g)
        dg.text(org, text, font=f, fill=glow_col + (150,), stroke_width=stroke + 4, stroke_fill=glow_col + (150,))
        g = g.filter(ImageFilter.GaussianBlur(22))
        im = Image.alpha_composite(im, g)
    d = ImageDraw.Draw(im)
    d.text(org, text, font=f, fill=fill + (255,) if len(fill) == 3 else fill, stroke_width=stroke, stroke_fill=stroke_fill)
    return im


def paste(img, layer, center, scale=1.0, alpha=1.0, angle=0.0):
    if alpha <= 0.01 or scale <= 0.01: return img
    L = layer
    if scale != 1.0: L = L.resize((max(1, int(L.width * scale)), max(1, int(L.height * scale))), Image.BICUBIC)
    if angle: L = L.rotate(angle, resample=Image.BICUBIC, expand=True)
    if alpha < 1: L = L.copy(); L.putalpha(L.getchannel("A").point(lambda v: int(v * alpha)))
    img.paste(L, (int(center[0] - L.width / 2), int(center[1] - L.height / 2)), L)
    return img


def kinetic(img, text, center, t, size=150, fill=WHITE, glow_col=None, fname="Anton-Regular.ttf", out_t=None, angle=0.0, stroke=0):
    """pop in with overshoot; optional fade-out time (seconds since start)"""
    if t < 0: return img
    s = 0.6 + 0.4 * back_out(t / 0.35); a = clamp(t / 0.15)
    if out_t is not None and t > out_t: a *= 1 - clamp((t - out_t) / 0.25); s *= 1 + 0.15 * clamp((t - out_t) / 0.25)
    return paste(img, text_img(text, fname, size, fill, stroke, (0, 0, 0), glow_col), center, s, a, angle)


def words_in(img, words, center, t, size=110, fill=WHITE, glow_col=None, stagger=0.12, gap=22):
    """word-by-word kinetic line"""
    ims = [text_img(w, "Anton-Regular.ttf", size, fill, 0, (0, 0, 0), glow_col) for w in words]
    pad = 70
    widths = [im.width - 2 * pad for im in ims]; total = sum(widths) + gap * (len(ims) - 1)
    x = center[0] - total / 2
    for k, im in enumerate(ims):
        tk = t - k * stagger
        if tk > 0:
            dy = 40 * (1 - ease_out(tk / 0.3))
            paste(img, im, (x + widths[k] / 2, center[1] + dy), 0.8 + 0.2 * back_out(tk / 0.3), clamp(tk / 0.12))
        x += widths[k] + gap
    return img


def flap_counter(img, start, end, center, t, size=230, color=GOLD, dur=1.0, stagger=0.12):
    """split-flap style digits rolling from start to end"""
    s, e = str(start), str(end); n = len(e); s = s.rjust(n, "0")
    tile_w, tile_h = int(size * 0.62), int(size * 1.05)
    x0 = center[0] - (n * tile_w + (n - 1) * 14) / 2
    f = font("Anton-Regular.ttf", size)
    for k in range(n):
        tk = clamp((t - k * stagger) / dur)
        a, b = int(s[k]), int(e[k]); steps = (b - a) % 10
        pos = a + ease(tk) * steps if steps else a
        digit = int(round(pos)) % 10; frac = pos - math.floor(pos)
        x = x0 + k * (tile_w + 14); y = center[1] - tile_h / 2
        tile = Image.new("RGBA", (tile_w, tile_h), (14, 18, 26, 235)); dt = ImageDraw.Draw(tile)
        dt.rounded_rectangle([0, 0, tile_w - 1, tile_h - 1], 16, outline=color + (255,), width=4)
        # rolling digit
        cur = int(math.floor(pos)) % 10; nxt = (cur + 1) % 10
        off = -frac * tile_h if steps and tk < 1 else 0
        for dgt, yo in ((cur, off), (nxt, off + tile_h)):
            if yo > tile_h: continue
            dt.text((tile_w / 2, tile_h / 2 + yo), str(dgt), font=f, fill=color + (255,), anchor="mm")
        dt.line([(0, tile_h / 2), (tile_w, tile_h / 2)], fill=(0, 0, 0, 255), width=5)
        g = Image.new("RGBA", (tile_w + 80, tile_h + 80), (0, 0, 0, 0)); g.paste(tile, (40, 40), tile)
        img = paste(img, g, (x + tile_w / 2, center[1]), 1.0, 1.0)
    gl = Image.new("RGB", (W, H)); dg = ImageDraw.Draw(gl)
    dg.rectangle([x0, center[1] - tile_h / 2, x0 + n * tile_w + (n - 1) * 14, center[1] + tile_h / 2], outline=color, width=6)
    return add(img, glow(gl, 40, 0.8))


def caption(img, text):
    if not text: return img
    return paste(img, text_img(text, "Inter.ttf", 60, WHITE, 0, (0, 0, 0), None, True), (W / 2, int(H * 0.66)))


def header(img):
    return paste(img, text_img("FORGOTTEN EAST", "LibreBaskerville.ttf", 30, (235, 225, 205), 0, (0, 0, 0), None, True), (W / 2, 92))


def vig(img):
    return Image.fromarray(np.clip(np.asarray(img, np.float32) * VIG, 0, 255).astype(np.uint8))


def finish(img, flash=0.0, desat=0.0):
    a = np.asarray(img, np.float32)
    if desat > 0:
        g = a.mean(-1, keepdims=True); a = a * (1 - desat) + g * desat
    a = a + GRAIN
    if flash > 0: a = a + 255 * flash
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
