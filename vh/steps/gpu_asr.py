"""GPU ASR worker — runs in the `deepspeed` env (torch cu130 + transformers).

Invoked as a subprocess by steps/transcribe.py because faster-whisper's
CTranslate2 aarch64 wheel has no CUDA support, but this box's torch DOES drive
the GB10 GPU. Reads a 16 kHz mono wav, writes word-level segments as JSON.

Usage:  python gpu_asr.py <wav> <out.json> <lang> <model>
Only JSON goes to <out.json>; all logs go to stderr.
"""
import sys, json, wave, warnings
warnings.filterwarnings("ignore")


def main():
    wav_path, out_path, lang, model = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    lang = None if lang in ("", "auto", "none") else lang

    import numpy as np, torch
    from transformers import pipeline

    w = wave.open(wav_path, "rb")
    audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    sr = w.getframerate()

    dev = 0 if torch.cuda.is_available() else -1
    print(f"[gpu_asr] device={'cuda' if dev==0 else 'cpu'} model={model}", file=sys.stderr)
    asr = pipeline(
        "automatic-speech-recognition", model=f"openai/whisper-{model}",
        device=dev, dtype=torch.float16 if dev == 0 else torch.float32,
        chunk_length_s=30, batch_size=16,
    )
    gk = {"task": "transcribe"}
    if lang:
        gk["language"] = lang
    out = asr({"raw": audio, "sampling_rate": sr},
              return_timestamps="word", generate_kwargs=gk)

    words = []
    prev_end = 0.0
    for c in out.get("chunks", []):
        ts = c.get("timestamp") or (None, None)
        start = ts[0] if ts[0] is not None else prev_end
        end = ts[1] if ts[1] is not None else start + 0.3
        prev_end = end
        txt = (c.get("text") or "").strip()
        if txt:
            words.append({"start": float(start), "end": float(end), "text": txt})

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(words, f, ensure_ascii=False)
    print(f"[gpu_asr] {len(words)} words -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
