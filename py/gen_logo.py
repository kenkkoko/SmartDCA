"""Render the Smart DCA logo mark ("每月一格": a 4x4 month grid, filled dots = periods
already invested, the ultramarine square = this period, hollow dots = periods to come)
into every raster icon the site ships (public/*.png).

Geometry is the same 64x64 grid as the header SVG (src/app.jsx: LogoMark).
Run:  python py/gen_logo.py
"""
import os

from PIL import Image, ImageDraw

OUT = os.path.join(os.path.dirname(__file__), '..', 'public')
SS = 8  # supersampling

BG = (17, 17, 17)          # --ink (light theme)
DOT = (241, 239, 234)      # --ink (dark theme)
ACCENT = (143, 166, 255)   # ultramarine lifted for a dark ground


def mark(draw, ox, oy, scale):
    """Draw the 4x4 grid on a 64-unit box at offset (ox, oy) with `scale` px per unit."""
    for r in range(4):
        for c in range(4):
            i = r * 4 + c
            cx = ox + (12 + c * 13) * scale
            cy = oy + (12 + r * 13) * scale
            if i == 13:
                h = 6 * scale
                draw.rectangle([cx - h, cy - h, cx + h, cy + h], fill=ACCENT)
            elif i < 13:
                rr = 4.2 * scale
                draw.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=DOT)
            else:
                rr = 3.4 * scale
                draw.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=DOT, width=max(1, round(1.6 * scale)))


def icon(size, content=0.69, radius=0.0):
    """content: fraction of the icon the 64-unit mark box occupies (maskable icons stay in the 80% safe circle)."""
    S = size * SS
    img = Image.new('RGBA', (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if radius:
        d.rounded_rectangle([0, 0, S - 1, S - 1], radius=int(S * radius), fill=BG)
    else:
        d.rectangle([0, 0, S, S], fill=BG)
    box = S * content
    mark(d, (S - box) / 2, (S - box) / 2, box / 64)
    return img.resize((size, size), Image.LANCZOS)


def small_icon(size):
    """Tab-size variant: the same 4x4 month grid laid on whole pixels so it stays crisp.
    32px: 5px dots, 2px gaps; 16px: 2px squares, 1px gaps (dots read as squares at that size)."""
    if size == 16:
        margin, cell, gap, rad = 2, 2, 1, 3
    else:
        margin, cell, gap, rad = 4, 5, 2, 6
    grid = 4 * cell + 3 * gap
    off = (size - grid) // 2
    S = size * SS
    img = Image.new('RGBA', (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, S - 1, S - 1], radius=rad * SS, fill=BG)
    for r in range(4):
        for c in range(4):
            i = r * 4 + c
            x0 = (off + c * (cell + gap)) * SS
            y0 = (off + r * (cell + gap)) * SS
            box = [x0, y0, x0 + cell * SS - 1, y0 + cell * SS - 1]
            if i == 13:
                d.rectangle(box, fill=ACCENT)
            elif i < 13:
                (d.rectangle if size == 16 else d.ellipse)(box, fill=DOT)
            else:
                if size == 16:
                    d.rectangle(box, fill=(92, 92, 92))  # periods to come: dimmed at 16px, where a 2px outline would read as filled
                else:
                    d.ellipse(box, outline=DOT, width=SS)
    return img.resize((size, size), Image.LANCZOS)


SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
<rect width="64" height="64" rx="12" fill="#111111"/>
{cells}
</svg>
"""


def favicon_svg():
    """Scalable tab icon: the full 4x4 mark on the ink tile (preferred by modern browsers)."""
    parts = []
    for i in range(16):
        r, c = divmod(i, 4)
        x, y = 12.5 + c * 13, 12.5 + r * 13
        if i == 13:
            parts.append(f'<rect x="{x - 5.5}" y="{y - 5.5}" width="11" height="11" fill="#8fa6ff"/>')
        elif i < 13:
            parts.append(f'<circle cx="{x}" cy="{y}" r="4.6" fill="#f1efea"/>')
        else:
            parts.append(f'<circle cx="{x}" cy="{y}" r="3.6" fill="none" stroke="#f1efea" stroke-width="1.8"/>')
    return SVG.format(cells=chr(10).join(parts))


if __name__ == '__main__':
    targets = {
        # Maskable PWA icons: full-bleed square, mark inside the safe zone
        'icon-192.png': icon(192, content=0.56),
        'icon-512.png': icon(512, content=0.56),
        # iOS rounds this itself
        'apple-touch-icon.png': icon(180, content=0.62),
        # Notification icon + generic app image
        'app-icon.png': icon(512, content=0.62),
        # Browser tab icons: small rounded square so the mark reads on light and dark tab strips
        'favicon-32.png': small_icon(32),
        'favicon-16.png': small_icon(16),
    }
    for name, im in targets.items():
        im.save(os.path.join(OUT, name))
        print(name, im.size)
    # favicon.ico: browsers request /favicon.ico on their own; multi-size, crisp 2x2 at 16/32
    ico = small_icon(32)
    ico.save(os.path.join(OUT, 'favicon.ico'), sizes=[(16, 16), (32, 32), (48, 48)],
             append_images=[small_icon(16), icon(48, content=0.8, radius=0.18)])
    print('favicon.ico')
    with open(os.path.join(OUT, 'favicon.svg'), 'w', encoding='utf-8') as f:
        f.write(favicon_svg())
    print('favicon.svg')
