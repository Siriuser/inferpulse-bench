"""Generate a deterministic, non-sensitive PNG workload using the standard library."""

import hashlib
import struct
import zlib
from pathlib import Path


def generate():
    width, height = 1024, 768
    rows = []
    for y in range(height):
        row = bytearray([0])
        for x in range(width):
            color = (248, 250, 252)
            if x % 128 == 0 or y % 128 == 0:
                color = (220, 226, 234)
            if 96 <= x < 288 and 96 <= y < 288:
                color = (220, 60, 60)
            if (x - 512) ** 2 + (y - 192) ** 2 <= 96 ** 2:
                color = (35, 160, 100)
            if 96 <= y <= 288 and abs(x - 832) <= (y - 96) // 2:
                color = (45, 105, 215)
            for left, top, rgb in (
                (128, 512, (220, 60, 60)),
                (352, 448, (35, 160, 100)),
                (576, 384, (45, 105, 215)),
                (800, 320, (235, 175, 40)),
            ):
                if left <= x < left + 96 and top <= y < 672:
                    color = rgb
            row.extend(color)
        rows.append(bytes(row))

    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data)))

    data = (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"".join(rows), level=9))
            + chunk(b"IEND", b""))
    path = Path(__file__).with_name("agent-image-grid.png")
    path.write_bytes(data)
    print(f"{path.name}: {width}x{height}, {len(data)} bytes, sha256={hashlib.sha256(data).hexdigest()}")


if __name__ == "__main__":
    generate()
