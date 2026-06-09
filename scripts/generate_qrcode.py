#!/usr/bin/env python3
"""
Generate QR codes for project pages.
Usage:
    python scripts/generate_qrcode.py <url> [--output <path>]

Example:
    # Generate QR code for a specific project page
    python scripts/generate_qrcode.py https://yourdomain.com/project/coursework-demo/

    # Generate QR code with custom output path
    python scripts/generate_qrcode.py https://yourdomain.com/project/coursework-demo/ --output static/img/qrcode-coursework.png
"""

import argparse
import os
import qrcode
from qrcode.image.styledpil import StyledPilImage
from qrcode.image.styles.moduledrawers import RoundedModuleDrawer


def generate_qr(url, output_path=None, show=True):
    """Generate a QR code for the given URL."""
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)

    # Create styled image with rounded modules
    img = qr.make_image(
        image_factory=StyledPilImage,
        module_drawer=RoundedModuleDrawer(),
        fill_color="black",
        back_color="white",
    )

    if output_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img.save(output_path)
        print(f"QR code saved to: {output_path}")
    else:
        # Default output path
        default_path = "static/img/qrcode.png"
        os.makedirs(os.path.dirname(default_path), exist_ok=True)
        img.save(default_path)
        print(f"QR code saved to: {default_path}")

    if show:
        img.show()

    return img


def main():
    parser = argparse.ArgumentParser(
        description="Generate QR codes for your course project pages."
    )
    parser.add_argument("url", help="The URL to encode in the QR code")
    parser.add_argument(
        "--output", "-o", default=None, help="Output file path (default: static/img/qrcode.png)"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the image after generation",
    )

    args = parser.parse_args()
    generate_qr(args.url, args.output, show=not args.no_show)


if __name__ == "__main__":
    main()
