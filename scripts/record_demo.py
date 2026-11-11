#!/usr/bin/env python3
"""
Record a demo GIF of TrustyFile analyzing the sample invoice.

Uses Playwright to:
1. Start the website (uvicorn src.web.app:app) as a subprocess
2. Open headless Chromium and navigate to the app
3. Upload the sample invoice served on the site (/demo)
4. Wait for analysis to complete
5. Scroll through results
6. Save video and convert to GIF with ffmpeg

Prerequisites:
    pip install playwright
    playwright install chromium
    sudo apt install ffmpeg  (or brew install ffmpeg on macOS)

Usage:
    python scripts/record_demo.py            # Full pipeline: record + convert to GIF
    python scripts/record_demo.py --video-only  # Record .webm only (no ffmpeg needed)
"""

import shutil
import subprocess
import sys
import time
import signal
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent
# The sample invoice the landing page offers ("Try it with an example"),
# committed with the site, so the demo needs nothing from samples/.
DEMO_PDF = ROOT / "public" / "static" / "demo" / "facture-exemple.pdf"
OUTPUT_GIF = ROOT / "docs" / "demo.gif"
OUTPUT_WEBM = ROOT / "docs" / "demo.webm"
APP_PORT = 8502  # Use a non-default port to avoid conflicts


def check_prerequisites(video_only: bool) -> None:
    """
    Verify that all required tools are installed before starting.

    Args:
        video_only: If True, skip the ffmpeg check.
    """
    # Check Playwright
    try:
        from playwright.sync_api import sync_playwright  # noqa: F401
    except ImportError:
        print("ERROR: Playwright not installed.")
        print("  pip install playwright && playwright install chromium")
        sys.exit(1)

    # Check ffmpeg (only needed for GIF conversion)
    if not video_only:
        result = shutil.which("ffmpeg")
        if result is None:
            print("ERROR: ffmpeg not found. Install it or use --video-only.")
            print("  Ubuntu/Debian: sudo apt install ffmpeg")
            print("  macOS: brew install ffmpeg")
            sys.exit(1)


def wait_for_server(url: str, timeout: int = 30) -> bool:
    """
    Poll a URL until it responds (or timeout).

    Args:
        url: The URL to check.
        timeout: Max seconds to wait.

    Returns:
        True if the server responded, False if timed out.
    """
    import urllib.request
    import urllib.error

    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(0.5)
    return False


def record_demo(video_only: bool = False):
    """
    Main function: start the app, record the demo, convert to GIF.

    Args:
        video_only: If True, keep the .webm and skip GIF conversion.
    """
    check_prerequisites(video_only)

    # Ensure the demo PDF exists
    if not DEMO_PDF.exists():
        print(f"Demo PDF not found at {DEMO_PDF}")
        print("Run: python scripts/make_demo_invoice.py")
        sys.exit(1)

    # Ensure output directory exists
    OUTPUT_GIF.parent.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Step 1: Start the app in background
    # =========================================================================
    print("Starting the app...")
    app_proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn", "src.web.app:app",
            "--port", str(APP_PORT),
        ],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    app_url = f"http://localhost:{APP_PORT}"

    try:
        # Wait for the server to be ready
        print(f"Waiting for the app at {app_url}...")
        if not wait_for_server(app_url):
            print("ERROR: the app did not start within 30 seconds.")
            sys.exit(1)
        print("The app is ready!")

        # Give it a moment to fully initialize
        time.sleep(2)

        # =====================================================================
        # Step 2: Record with Playwright
        # =====================================================================
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            # Launch browser with video recording
            video_dir = ROOT / "docs" / "_video_tmp"
            video_dir.mkdir(parents=True, exist_ok=True)

            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                viewport={"width": 1200, "height": 800},
                record_video_dir=str(video_dir),
                record_video_size={"width": 1200, "height": 800},
            )
            page = context.new_page()

            # Navigate to the app, in English (the README is in English)
            print("Navigating to app...")
            page.goto(f"{app_url}/en", wait_until="networkidle")
            time.sleep(2)

            # -----------------------------------------------------------------
            # Upload the demo PDF
            # -----------------------------------------------------------------
            print("Uploading demo PDF...")

            # The drop zone wraps a file input; choosing a file submits the
            # form, and the server answers with the results page.
            file_input = page.locator('#tf-upload input[type="file"]')
            file_input.set_input_files(str(DEMO_PDF))

            # Wait for analysis to complete: the loading overlay shows until
            # the results page arrives; the feedback block exists only there.
            print("Waiting for analysis to complete...")
            page.wait_for_selector("#tf-feedback", timeout=60000)
            # Give the UI time to render all results
            time.sleep(3)

            # -----------------------------------------------------------------
            # Scroll through results slowly for the recording
            # -----------------------------------------------------------------
            print("Scrolling through results...")

            # The page itself scrolls.
            scroll_data = page.evaluate("""() => {
                const el = document.documentElement;
                return {height: el.scrollHeight, client: window.innerHeight};
            }""")
            total = scroll_data["height"]
            client = scroll_data["client"]
            print(f"  Content height: {total}px, viewport: {client}px")

            # Scroll down in smooth steps
            step = 150
            scroll_pos = 0
            max_scroll = total - client
            while scroll_pos < max_scroll:
                scroll_pos = min(scroll_pos + step, max_scroll)
                page.evaluate(f"window.scrollTo(0, {scroll_pos})")
                time.sleep(0.5)

            # Pause at the bottom
            time.sleep(3)

            # Scroll back to top
            while scroll_pos > 0:
                scroll_pos = max(scroll_pos - 300, 0)
                page.evaluate(f"window.scrollTo(0, {scroll_pos})")
                time.sleep(0.2)
            time.sleep(2)

            # -----------------------------------------------------------------
            # Close browser to finalize video
            # -----------------------------------------------------------------
            print("Finalizing recording...")
            video_path = page.video.path()
            context.close()
            browser.close()

        # =====================================================================
        # Step 3: Convert video to GIF (or keep .webm)
        # =====================================================================
        if video_only:
            # Just copy the .webm to docs/
            shutil.copy2(str(video_path), str(OUTPUT_WEBM))
            shutil.rmtree(str(video_dir), ignore_errors=True)
            size_mb = OUTPUT_WEBM.stat().st_size / (1024 * 1024)
            print(f"Demo video saved: {OUTPUT_WEBM} ({size_mb:.1f} MB)")
            print("To convert to GIF later:")
            print(f"  ffmpeg -i {OUTPUT_WEBM} -vf 'fps=8,scale=1000:-1' -loop 0 {OUTPUT_GIF}")
        else:
            print(f"Converting to GIF: {OUTPUT_GIF}")

            # Two-pass approach for better quality:
            # 1. Generate a palette from the video
            # 2. Use the palette to create a high-quality GIF
            palette_path = video_dir / "palette.png"

            # Pass 1: Generate palette
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", str(video_path),
                    "-vf", "fps=8,scale=1000:-1:flags=lanczos,palettegen=stats_mode=diff",
                    str(palette_path),
                ],
                check=True,
                capture_output=True,
            )

            # Pass 2: Create GIF using palette
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", str(video_path),
                    "-i", str(palette_path),
                    "-lavfi", "fps=8,scale=1000:-1:flags=lanczos [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=5",
                    "-loop", "0",
                    str(OUTPUT_GIF),
                ],
                check=True,
                capture_output=True,
            )

            # Clean up temp video files
            shutil.rmtree(str(video_dir), ignore_errors=True)

            gif_size_mb = OUTPUT_GIF.stat().st_size / (1024 * 1024)
            print(f"Demo GIF created: {OUTPUT_GIF} ({gif_size_mb:.1f} MB)")

    finally:
        # =====================================================================
        # Cleanup: stop the app
        # =====================================================================
        print("Stopping the app...")
        app_proc.send_signal(signal.SIGTERM)
        app_proc.wait(timeout=10)


if __name__ == "__main__":
    flag_video_only = "--video-only" in sys.argv
    record_demo(video_only=flag_video_only)
