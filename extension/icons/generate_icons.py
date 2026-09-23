"""
Generates crisp PNG icons (16x16, 48x48, 128x128) for the NETRA extension.
Pure Python with zlib (no external dependencies needed).
Design: Recognizable Eye shape (eyelid contour, violet iris, white pupil).
"""
import zlib
import struct
from pathlib import Path

def create_png(width, height, rgba_data):
    """Creates a raw PNG file from RGBA byte array."""
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff)

    png = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    png += chunk(b"IHDR", ihdr)

    raw_scanlines = bytearray()
    for y in range(height):
        raw_scanlines.append(0)
        start = y * width * 4
        raw_scanlines.extend(rgba_data[start:start + width * 4])

    compressed = zlib.compress(bytes(raw_scanlines), level=9)
    png += chunk(b"IDAT", compressed)
    png += chunk(b"IEND", b"")
    return png

def render_eye_icon(size):
    rgba = bytearray(size * size * 4)
    cx, cy = size / 2.0, size / 2.0
    w_half = size * 0.44
    h_max = size * 0.30
    r_iris = size * 0.20
    r_pupil = size * 0.08
    border_thick = max(1.0, size * 0.07)

    for y in range(size):
        for x in range(size):
            idx = (y * size + x) * 4
            dx = x + 0.5 - cx
            dy = y + 0.5 - cy
            dist_center = (dx * dx + dy * dy) ** 0.5

            # Eye contour curve
            if abs(dx) < w_half:
                norm_x = dx / w_half
                eyelid_limit = h_max * (1.0 - norm_x * norm_x)
            else:
                eyelid_limit = -1.0

            if abs(dy) <= eyelid_limit:
                # Inside the eye opening
                dist_to_edge = eyelid_limit - abs(dy)
                if dist_to_edge < border_thick:
                    # Eyelid border (#202020)
                    rgba[idx] = 32
                    rgba[idx + 1] = 32
                    rgba[idx + 2] = 32
                    rgba[idx + 3] = 255
                elif dist_center <= r_pupil:
                    # White pupil
                    rgba[idx] = 255
                    rgba[idx + 1] = 255
                    rgba[idx + 2] = 255
                    rgba[idx + 3] = 255
                elif dist_center <= r_iris:
                    # Brand violet iris (#6647f0)
                    rgba[idx] = 102
                    rgba[idx + 1] = 71
                    rgba[idx + 2] = 240
                    rgba[idx + 3] = 255
                else:
                    # Eyeball white (#f8f9fa)
                    rgba[idx] = 248
                    rgba[idx + 1] = 249
                    rgba[idx + 2] = 250
                    rgba[idx + 3] = 255
            elif eyelid_limit > 0 and abs(dy) <= eyelid_limit + 0.8:
                # Anti-aliasing on eyelid outer edge
                alpha = int(255 * max(0.0, min(1.0, (eyelid_limit + 0.8 - abs(dy)) / 0.8)))
                rgba[idx] = 32
                rgba[idx + 1] = 32
                rgba[idx + 2] = 32
                rgba[idx + 3] = alpha
            else:
                rgba[idx + 3] = 0

    return create_png(size, size, rgba)

def main():
    icons_dir = Path(__file__).resolve().parent
    icons_dir.mkdir(parents=True, exist_ok=True)
    for size in (16, 48, 128):
        data = render_eye_icon(size)
        out_path = icons_dir / f"icon-{size}.png"
        out_path.write_bytes(data)
        print(f"Generated {out_path} ({size}x{size})")

if __name__ == "__main__":
    main()
