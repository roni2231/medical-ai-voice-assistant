import logging
import os
import subprocess
import wave
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import yaml
import ollama

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Orpheus TTS outputs 24 kHz mono int16 PCM regardless of backend
_SAMPLE_RATE = 24_000
_SAMPLE_WIDTH = 2
_CHANNELS = 1

_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _load_config() -> dict[str, Any]:
    with _CONFIG_PATH.open() as f:
        cfg: dict[str, Any] = yaml.safe_load(f)
    profile_name = os.getenv("ENV", cfg.get("active_profile", "local"))
    profile = cfg["profiles"][profile_name]
    logger.info("Config loaded: profile=%s", profile_name)
    return {
        "ollama_host":    profile["ollama"]["host"],
        "ollama_model":   profile["ollama"]["model"],
        "tts_backend":    profile["tts"]["backend"],
        "tts_model_name": profile["tts"]["model_name"],
        "tts_voice":      profile["tts"]["voice"],
        "audio_path":     profile["audio"]["output_path"],
        "system_prompt":  cfg["assistant"]["system_prompt"].strip(),
    }


_CFG = _load_config()


def _init_tts() -> Callable[[str, str], Iterator[bytes]]:
    """Return a (text, voice) -> bytes-chunk iterator for the configured backend."""
    backend = _CFG["tts_backend"]

    if backend == "cpp":
        # Local inference via llama.cpp (Metal on M4, CPU elsewhere)
        # Install: CMAKE_ARGS="-DGGML_METAL=on" pip install orpheus-cpp llama-cpp-python --no-binary llama-cpp-python
        import numpy as np
        from orpheus_cpp import OrpheusCpp  # type: ignore[import]

        logger.info("TTS: initializing orpheus-cpp (backend=cpp)")
        model = OrpheusCpp(verbose=False, lang="en")

        def _stream_cpp(text: str, voice: str) -> Iterator[bytes]:
            for _sr, chunk in model.stream_tts_sync(text, options={"voice_id": voice}):
                arr = np.asarray(chunk).squeeze()
                if arr.dtype != np.int16:
                    arr = (arr * 32767).clip(-32768, 32767).astype(np.int16)
                yield arr.tobytes()

        return _stream_cpp

    elif backend == "vllm":
        # GPU inference via vllm (CUDA required, e.g. RTX 5070)
        # Install: pip install orpheus-speech
        from orpheus_tts import OrpheusModel  # type: ignore[import]

        logger.info("TTS: initializing OrpheusModel (%s)", _CFG["tts_model_name"])
        model = OrpheusModel(model_name=_CFG["tts_model_name"])

        def _stream_vllm(text: str, voice: str) -> Iterator[bytes]:
            yield from model.generate_speech(prompt=text, voice=voice)

        return _stream_vllm

    else:
        raise ValueError(
            f"Unknown tts.backend: {backend!r}. Valid options: 'cpp', 'vllm'."
        )


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


def speak(text: str, stream: Callable[[str, str], Iterator[bytes]]) -> None:
    """Synthesize text via the configured TTS backend and play via ffplay."""
    print(f"Assistant: {text}")
    audio_path = _CFG["audio_path"]
    logger.info("TTS synthesis: %d chars -> %s", len(text), audio_path)
    try:
        with wave.open(audio_path, "wb") as wf:
            wf.setnchannels(_CHANNELS)
            wf.setsampwidth(_SAMPLE_WIDTH)
            wf.setframerate(_SAMPLE_RATE)
            for chunk in stream(text, _CFG["tts_voice"]):
                wf.writeframes(chunk)
        subprocess.run(
            ["ffplay", "-nodisp", "-autoexit", audio_path],
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except subprocess.CalledProcessError:
        logger.error("ffplay failed -- install ffmpeg (brew install ffmpeg / apt install ffmpeg)")
    except Exception:
        logger.exception("TTS synthesis or playback failed")


def run() -> None:
    """Main REPL loop."""
    logger.info("Initializing TTS backend: %s", _CFG["tts_backend"])
    stream = _init_tts()
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
        speak(reply, stream)


if __name__ == "__main__":
    run()
