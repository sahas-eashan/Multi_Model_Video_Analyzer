import subprocess
import sys


def install_ffmpeg():
    """Install FFmpeg on the system."""

    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--upgrade", "setuptools"]
    )
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade", "ffmpeg-python"]
        )
        print("FFmpeg installed successfully.")
    except subprocess.CalledProcessError:
        print("Failed to install FFmpeg.")

    try:
        subprocess.check_call(
            ["wget", "https://ffmpeg.org/releases/ffmpeg-release-full.zip"]
        )
        subprocess.check_call(["unzip", "ffmpeg-release-full.zip"])
        result = subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            capture_output=True,
            text=True,
        )
        ffmpeg_path = result.stdout.strip()

        subprocess.check_call(["cp", ffmpeg_path, "/usr/local/bin/ffmpeg"])

        subprocess.check_call(["chmod", "+x", "/usr/local/bin/ffmpeg"])

        print(f"FFmpeg installed successfully at {ffmpeg_path}.")

    except subprocess.CalledProcessError:
        print("Failed to install FFmpeg.")

    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        print("FFmpeg version:")
        print(result.stdout.strip())
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("FFmpeg is not installed.")
        return False
