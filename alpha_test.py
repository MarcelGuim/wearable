#!/usr/bin/env python3
"""
Guided eyes-open / eyes-closed alpha test (the "Berger effect").

When you close your eyes, alpha waves (8-13 Hz) get clearly stronger, typically
1.5-3x, most visibly on the ear electrodes (TP9/TP10). If this test shows that
increase, the whole chain (Muse -> ESP32 -> USB audio -> eeg_bands.py) is working.

Steps
-----
1. Contact check: waits until at least 2 electrodes (including TP9 or TP10) show
   good contact for 5 s in a row. Adjust the headband while it's waiting.
2. Test: alternates EYES OPEN / EYES CLOSED blocks (default 3 x 20 s each, 2 min).
   A spoken cue (macOS "say") tells you when to switch, so you can keep your eyes closed.
3. Analysis: compares alpha power eyes-closed vs eyes-open per electrode, using only
   artifact-free 2 s epochs, and prints a PASS / WEAK / FAIL verdict.

The recording is saved as alpha_test_<date>_<time>.npz (EEG in uV at 256 Hz + markers)
and a summary figure as .png next to it.

Usage
-----
    python alpha_test.py                    # full guided test
    python alpha_test.py --plot             # ...and show the figure at the end
    python alpha_test.py --analyze alpha_test_20261113_113000.npz --plot   # re-analyse
"""

import argparse
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime

import numpy as np
from scipy import signal, stats

import eeg_bands as eb

ALPHA = eb.BANDS["alpha"]
EPOCH_S = 2.0           # analysis epoch length
SKIP_START_S = 3.0      # ignore the first seconds of each block (reacting to the cue)
SKIP_END_S = 1.0        # ...and the last second
MAX_PTP_UV = 300.0      # epoch rejected if the 1-45 Hz signal swings more than this (blinks, movement)
MIN_EPOCHS = 4          # clean epochs needed per condition per electrode


# --------------------------------------------------------------------------- #
# Recording
# --------------------------------------------------------------------------- #
class Recorder:
    """Ring buffer for the live contact check + full recording of every audio block."""

    def __init__(self, fs):
        self.ring = eb.AudioRingBuffer(int(4 * fs), 4)
        self.blocks = []
        self.n = 0
        self.recording = False
        self.lock = threading.Lock()

    def callback(self, indata, frames, t, status):
        block = indata[:, :4].copy()
        self.ring.push(block)
        if self.recording:
            with self.lock:
                self.blocks.append(block)
                self.n += len(block)

    def position(self):
        with self.lock:
            return self.n

    def data(self):
        with self.lock:
            return np.concatenate(self.blocks) if self.blocks else np.zeros((0, 4), np.float32)


def say(text, quiet):
    print(f"\n>>> {text.upper()}")
    if quiet:
        return
    if shutil.which("say"):
        subprocess.Popen(["say", text])
    else:
        sys.stdout.write("\a")
        sys.stdout.flush()


def live_quality(rec, fs, scale):
    raw = rec.ring.snapshot()
    if len(raw) < fs * 2:
        return None, None
    clipped = eb.is_clipped(raw)
    eeg = eb.resample_to_eeg(raw.astype(np.float64), fs) * scale
    return eb.channel_quality(eeg, clipped)


def contact_ok(status):
    ears_ok = status[eb.CHANNELS.index("TP9")] == eb.GOOD or status[eb.CHANNELS.index("TP10")] == eb.GOOD
    return np.sum(status == eb.GOOD) >= 2 and ears_ok


def wait_for_contact(rec, fs, scale, hold_s):
    print("\nCONTACT CHECK - adjust the headband until at least 2 electrodes are green,")
    print("including TP9 or TP10 (behind the ears). Tips: hair away from the sensors,")
    print("press the ear pads against the skin, glasses off, relax the jaw. Ctrl+C to quit.\n")
    good_since = None
    while True:
        time.sleep(0.25)
        status, std = live_quality(rec, fs, scale)
        if status is None:
            sys.stdout.write("\r  buffering...\x1b[K")
            sys.stdout.flush()
            continue
        if contact_ok(status):
            good_since = good_since or time.time()
            held = time.time() - good_since
            msg = f"OK, hold still... {max(0, hold_s - held):.0f}s"
            if held >= hold_s:
                sys.stdout.write("\r  " + eb.contact_line(status, std) + "   contact good!\x1b[K\n")
                return
        else:
            good_since = None
            msg = "waiting for good contact"
        sys.stdout.write("\r  " + eb.contact_line(status, std) + f"   {msg}\x1b[K")
        sys.stdout.flush()


def run_protocol(rec, fs, scale, cycles, block_s, quiet):
    """Runs the open/closed blocks. Returns list of (label, start_sample, end_sample) at audio rate."""
    say("Get comfortable and keep still. The test starts in 5 seconds, with your eyes open.", quiet)
    time.sleep(6)
    rec.recording = True
    phases = []
    plan = [("open", "Eyes open. Look softly at one point.") if i % 2 == 0
            else ("closed", "Close your eyes and relax.") for i in range(2 * cycles)]
    for i, (label, cue) in enumerate(plan):
        say(cue, quiet)
        start = rec.position()
        t_end = time.time() + block_s
        while time.time() < t_end:
            time.sleep(0.25)
            status, std = live_quality(rec, fs, scale)
            line = eb.contact_line(status, std) if status is not None else ""
            sys.stdout.write(f"\r  [{i + 1}/{len(plan)} EYES {label.upper():<6}] "
                             f"{t_end - time.time():4.0f}s left   {line}\x1b[K")
            sys.stdout.flush()
        phases.append((label, start, rec.position()))
    rec.recording = False
    say("Test complete. You can open your eyes.", quiet)
    print()
    return phases


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def make_epochs(eeg, fs, phases):
    """Cut each block into EPOCH_S epochs, skipping the transitions. Returns list of (label, start, end)."""
    n = int(EPOCH_S * fs)
    out = []
    for label, s, e in phases:
        a, b = s + int(SKIP_START_S * fs), e - int(SKIP_END_S * fs)
        for k in range(a, b - n + 1, n):
            out.append((label, k, k + n))
    return out


def analyze(eeg, fs, phases, scale=eb.UV_PER_UNIT):
    """
    eeg: (n, 4) uV at fs. phases: list of (label, start, end) in samples at fs.
    Returns a results dict (also used for plotting).
    """
    filt = eb.bandlimit(eeg, fs)
    clip_uv = eb.CLIP_LEVEL * scale * 0.98
    epochs = make_epochs(eeg, fs, phases)

    n_ch = eeg.shape[1]
    per_epoch = []   # dicts with label, t, alpha (n_ch), clean (n_ch), psd (f, n_ch)
    freqs = None
    for label, a, b in epochs:
        seg = eeg[a:b]
        fseg = filt[a:b]
        std = fseg.std(axis=0)
        ptp = np.ptp(fseg, axis=0)
        clipped = np.mean(np.abs(seg) >= clip_uv, axis=0) > eb.CLIP_FRACTION
        clean = ~clipped & (std < eb.BAD_STD_UV) & (std > eb.FLAT_STD_UV) & (ptp < MAX_PTP_UV)
        x = eb.preprocess(seg, fs)
        freqs, psd = signal.welch(x, fs=fs, nperseg=int(fs), axis=0)   # 1 Hz resolution
        m = (freqs >= ALPHA[0]) & (freqs < ALPHA[1])
        alpha = eb._trapz(psd[m], freqs[m], axis=0)
        mt = (freqs >= 1) & (freqs < 45)
        total = eb._trapz(psd[mt], freqs[mt], axis=0)
        per_epoch.append(dict(label=label, t=a / fs, alpha=alpha, rel=alpha / total, clean=clean, psd=psd))

    channels = []
    for c in range(n_ch):
        o = [ep for ep in per_epoch if ep["label"] == "open" and ep["clean"][c]]
        cl = [ep for ep in per_epoch if ep["label"] == "closed" and ep["clean"][c]]
        total_o = sum(ep["label"] == "open" for ep in per_epoch)
        total_c = sum(ep["label"] == "closed" for ep in per_epoch)
        res = dict(name=eb.CHANNELS[c], n_open=len(o), n_closed=len(cl), n_open_total=total_o,
                   n_closed_total=total_c, ratio=np.nan, p=np.nan, verdict="not enough clean data",
                   psd_open=None, psd_closed=None, peak_hz=np.nan, rel_open=np.nan, rel_closed=np.nan)
        if len(o) >= MIN_EPOCHS and len(cl) >= MIN_EPOCHS:
            ao = np.array([ep["alpha"][c] for ep in o])
            ac = np.array([ep["alpha"][c] for ep in cl])
            res["ratio"] = float(np.median(ac) / np.median(ao))
            res["p"] = float(stats.mannwhitneyu(ac, ao, alternative="greater").pvalue)
            res["rel_open"] = float(np.median([ep["rel"][c] for ep in o]))
            res["rel_closed"] = float(np.median([ep["rel"][c] for ep in cl]))
            res["psd_open"] = np.mean([ep["psd"][:, c] for ep in o], axis=0)
            res["psd_closed"] = np.mean([ep["psd"][:, c] for ep in cl], axis=0)
            m = (freqs >= 7) & (freqs <= 14)
            gain = res["psd_closed"][m] / res["psd_open"][m]
            res["peak_hz"] = float(freqs[m][np.argmax(gain)])
            if res["ratio"] >= 1.5 and res["p"] < 0.05:
                res["verdict"] = "CLEAR alpha increase"
            elif res["ratio"] >= 1.2 and res["p"] < 0.1:
                res["verdict"] = "weak alpha increase"
            else:
                res["verdict"] = "no alpha increase"
        channels.append(res)

    verdicts = [r["verdict"] for r in channels]
    if any(v.startswith("CLEAR") for v in verdicts):
        overall = "PASS - clear alpha response; the pipeline is working."
    elif any(v.startswith("weak") for v in verdicts):
        overall = "WEAK - a small alpha response; likely working, but contact/relaxation could be better."
    elif all(v == "not enough clean data" for v in verdicts):
        overall = "INCONCLUSIVE - not enough artifact-free data. Improve contact and keep still, then retry."
    else:
        overall = "FAIL - no alpha increase on any electrode with clean data."
    return dict(freqs=freqs, epochs=per_epoch, channels=channels, overall=overall, phases=phases, fs=fs)


def print_report(res):
    print("\n" + "=" * 78)
    print("ALPHA TEST RESULTS   (alpha = 8-13 Hz power, eyes closed vs eyes open)")
    print("=" * 78)
    print(f"{'electrode':<10}{'clean epochs':>14}{'closed/open':>13}{'p-value':>10}"
          f"{'alpha % open->closed':>23}   verdict")
    for r in res["channels"]:
        epochs = f"{r['n_open']}/{r['n_open_total']} | {r['n_closed']}/{r['n_closed_total']}"
        ratio = f"{r['ratio']:.2f}x" if np.isfinite(r["ratio"]) else "-"
        p = f"{r['p']:.3f}" if np.isfinite(r["p"]) else "-"
        rel = (f"{100 * r['rel_open']:.0f}% -> {100 * r['rel_closed']:.0f}%"
               if np.isfinite(r["rel_open"]) else "-")
        print(f"{r['name']:<10}{epochs:>14}{ratio:>13}{p:>10}{rel:>23}   {r['verdict']}")
    print("\n(clean epochs = artifact-free 2 s epochs used, eyes open | eyes closed)")
    best = [r for r in res["channels"] if np.isfinite(r["ratio"])]
    if best:
        b = max(best, key=lambda r: r["ratio"])
        print(f"Strongest response on {b['name']}: {b['ratio']:.2f}x, biggest gain at {b['peak_hz']:.0f} Hz")
    print(f"\nVERDICT: {res['overall']}")
    print("=" * 78)


def plot_results(res, eeg, png_path, show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fs, freqs = res["fs"], res["freqs"]
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], hspace=0.35, wspace=0.3)
    m = (freqs >= 1) & (freqs <= 30)
    legend_done = False
    for c, r in enumerate(res["channels"]):
        ax = fig.add_subplot(gs[0, c])
        ax.axvspan(*ALPHA, color="#2ecc71", alpha=0.15)
        if r["psd_open"] is not None:
            ax.semilogy(freqs[m], r["psd_open"][m], label="eyes open", color="#3b6ea8")
            ax.semilogy(freqs[m], r["psd_closed"][m], label="eyes closed", color="#c0392b")
            ax.set_title(f"{r['name']}: {r['ratio']:.2f}x", fontsize=10)
            if not legend_done:
                ax.legend(fontsize=8)
                legend_done = True
        else:
            ax.set_title(f"{r['name']}: not enough clean data", fontsize=9)
        ax.set_xlabel("Hz")
        if c == 0:
            ax.set_ylabel("PSD (uV^2/Hz)")

    ax = fig.add_subplot(gs[1, :])
    for label, s, e in res["phases"]:
        if label == "closed":
            ax.axvspan(s / fs, e / fs, color="gray", alpha=0.2)
    for c, name in enumerate(eb.CHANNELS):
        pts = [(ep["t"] + EPOCH_S / 2, ep["alpha"][c]) for ep in res["epochs"] if ep["clean"][c]]
        if pts:
            t, a = zip(*pts)
            ax.semilogy(t, a, "o-", ms=3, lw=1, label=name)
    ax.set_xlabel("time (s)   [gray = eyes closed]")
    ax.set_ylabel("alpha power (uV^2), clean epochs")
    ax.legend(ncol=4, fontsize=8)
    ax.grid(alpha=0.3)
    fig.suptitle(res["overall"], fontsize=11)
    fig.savefig(png_path, dpi=110, bbox_inches="tight")
    print(f"Figure saved to {png_path}")
    if show:
        plt.show()


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="Guided eyes-open / eyes-closed alpha test")
    p.add_argument("--cycles", type=int, default=3, help="number of open+closed pairs (default 3)")
    p.add_argument("--block", type=float, default=20.0, help="seconds per block (default 20)")
    p.add_argument("--device", help="audio device index or name substring (default: 'muse')")
    p.add_argument("--scale", type=float, default=eb.UV_PER_UNIT, help="microvolts per audio unit")
    p.add_argument("--hold", type=float, default=5.0, help="seconds of good contact required before starting")
    p.add_argument("--skip-check", action="store_true", help="skip the contact check")
    p.add_argument("--quiet", action="store_true", help="no spoken cues")
    p.add_argument("--plot", action="store_true", help="show the results figure (always saved as .png)")
    p.add_argument("--analyze", help="re-analyse a saved alpha_test_*.npz instead of recording")
    args = p.parse_args()

    if args.analyze:
        d = np.load(args.analyze, allow_pickle=False)
        eeg, fs = d["eeg"].astype(np.float64), int(d["fs"])
        phases = list(zip(d["labels"].tolist(), d["starts"].tolist(), d["ends"].tolist()))
        res = analyze(eeg, fs, phases, args.scale)
        print_report(res)
        plot_results(res, eeg, args.analyze.replace(".npz", ".png"), args.plot)
        return

    import sounddevice as sd
    dev = eb.find_device(sd, args.device)
    info = sd.query_devices(dev)
    fs = int(info["default_samplerate"])
    print(f"Using device {dev}: {info['name']} @ {fs} Hz")

    rec = Recorder(fs)
    with sd.InputStream(device=dev, channels=4, samplerate=fs, dtype="float32",
                        blocksize=int(fs * 0.02), callback=rec.callback):
        try:
            if not args.skip_check:
                wait_for_contact(rec, fs, args.scale, args.hold)
            phases_audio = run_protocol(rec, fs, args.scale, args.cycles, args.block, args.quiet)
        except KeyboardInterrupt:
            print("\nAborted.")
            return

    raw = rec.data().astype(np.float64)
    eeg = eb.resample_to_eeg(raw, fs) * args.scale
    k = eb.EEG_FS / fs
    phases = [(lab, int(round(s * k)), int(round(e * k))) for lab, s, e in phases_audio]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    npz = f"alpha_test_{stamp}.npz"
    np.savez_compressed(npz, eeg=eeg.astype(np.float32), fs=eb.EEG_FS,
                        labels=np.array([p[0] for p in phases]),
                        starts=np.array([p[1] for p in phases]), ends=np.array([p[2] for p in phases]),
                        channels=np.array(eb.CHANNELS))
    print(f"Recording saved to {npz}")

    res = analyze(eeg, eb.EEG_FS, phases, args.scale)
    print_report(res)
    plot_results(res, eeg, npz.replace(".npz", ".png"), args.plot)


if __name__ == "__main__":
    main()
