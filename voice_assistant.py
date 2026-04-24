import logging
import os
import subprocess
import wave
from pathlib import Path
from typing import Any

import yaml
import ollama
from orpheus_tts import OrpheusClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# OrpheusClient streams raw PCM: int16, 48 kHz, mono
_SAMPLE_RATE = 48_000
_SAMPLE_WIDTH = 2
_CHANNELS = 1

_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _load_config() -> dict[str, Any]:
    """Load config.yaml and return the active profile merged with globals."""
    with _CONFIG_PATH.open() as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    # ENV env var overrides active_profile in the file — easy prod switch
    profile_name = os.getenv("ENV", cfg.get("active_profile", "local"))
    profile = cfg["profiles"][profile_name]
    logger.info("Config loaded: profile=%s", profile_name)

    return {
        "ollama_host":    profile["ollama"]["host"],
        "ollama_model":   profile["ollama"]["model"],
        "tts_ws_url":     profile["tts"]["ws_url"],
        "tts_voice":      profile["tts"]["voice"],
        "audio_path":     profile["audio"]["output_path"],
        "system_prompt":  cfg["assistant"]["system_prompt"].strip(),
    }


_CFG = _load_config()


def generate_response(prompt: str) -> str:
    """Query the local Ollama LLM and return the assistant reply."""
    logger.info("LLM inference: prompt length=%d chars", len(prompt))
    try:
        client = ollama.Client(host=_CFG["ollama_host"])
        resp = client.generate(
            model=_CFG["ollama_model"],
            prompt=f"{_CFG['system_prompt']}\n\nPatient: {prompt}\nAssistant:",
        )
        return resp.response
    except Exception:
        logger.exception("Ollama LLM generation failed")
        raise


def speak(text: str, tts: OrpheusClient) -> None:
    """Synthesize text via Orpheus TTS WebSocket stream and play via ffplay."""
    print(f"Assistant: {text}")
    audio_path = _CFG["audio_path"]
    logger.info("TTS synthesis: %d chars → %s", len(text), audio_path)
    try:
        with wave.open(audio_path, "wb") as wf:
            wf.setnchannels(_CHANNELS)
            wf.setsampwidth(_SAMPLE_WIDTH)
            wf.setframerate(_SAMPLE_RATE)
            for chunk in tts.stream(text, voice=_CFG["tts_voice"]):
                wf.writeframes(chunk)
        subprocess.run(
            ["ffplay", "-nodisp", "-autoexit", audio_path],
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except subprocess.CalledProcessError:
        logger.error("ffplay failed — install ffmpeg (brew install ffmpeg / apt install ffmpeg)")
    except Exception:
        logger.exception("TTS synthesis or playback failed")


def run() -> None:
    """Main REPL loop."""
    voice, ws_url = _CFG["tts_voice"], _CFG["tts_ws_url"]
    logger.info("Connecting to Orpheus TTS at %s (voice=%s)", ws_url, voice)
    with OrpheusClient(voice_endpoint_map={voice: ws_url}) as tts:
        tts.connect(voice=voice)
        print("Medical Voice Assistant Ready.\n")
        while True:
            try:
                user = input("Patient: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break
            if not user or user.lower() in ("quit", "exit"):
                break
            reply = generate_response(user)
            speak(reply, tts)


if __name__ == "__main__":
    run()
