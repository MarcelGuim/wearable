#!/usr/bin/env python3
"""
Muse 2 -> ESP32 -> USB audio  ==>  EEG brainwave band powers.

The ESP32 enumerates as a 4-channel USB audio input ("MuseYcal EEG Stream",
48 kHz). Each audio channel carries one Muse EEG electrode (sampled at 256 Hz
by the headband and interpolated up to 48 kHz by the ESP32).

Pipeline (per update):
  48 kHz audio  ->  anti-alias + resample to 256 Hz  ->  scale to microvolts
  ->  detrend + 60 Hz notch  ->  Welch PSD  ->  integrate PSD over each band
  ->  average across the electrodes with good contact (one reading, like the Muse app)

Bands (Hz):  delta 1-4 | theta 4-8 | alpha 8-13 | beta 13-30 | gamma 30-45

Usage
-----
  Live, combined reading in the terminal (refreshes every 0.1 s):
      python eeg_bands.py
  Also show each electrode separately:
      python eeg_bands.py --per-channel
  Live plot + CSV log:
      python eeg_bands.py --plot --csv session.csv
  Offline, from a 4-channel WAV exported out of Audacity:
      python eeg_bands.py --file recording.wav --plot
  List audio devices:
      python eeg_bands.py --list-devices
"""

import argparse
import csv
import sys
import threading
import time
from collections import deque
from math import gcd

import numpy as np
from scipy import signal

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
EEG_FS = 256                                   # Muse 2 native EEG sample rate
CHANNELS = ["TP9", "AF7", "AF8", "TP10"]       # Muse 2 electrode order
BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
# The ESP32 maps the Muse 12-bit raw value (0..4095) to audio as raw/4096 - 0.5.
# Muse scale is 0.48828125 uV per count, so 1.0 audio unit = 4096 * 0.488 = 2000 uV.
UV_PER_UNIT = 4096 * 0.48828125
CLIP_LEVEL = 0.499                             # audio value considered saturated
CLIP_FRACTION = 0.01                           # >1% saturated samples => bad contact

# Contact quality (std of the 1-45 Hz signal over the analysis window, in uV).
# Calibrated on this headset: good contact measured 5-30 uV.
GOOD, NOISY, BAD = 0, 1, 2
QUALITY_LABELS = ["good", "noisy", "BAD"]
NOISY_STD_UV = 40.0
BAD_STD_UV = 100.0
FLAT_STD_UV = 1.0

_trapz = getattr(np, "trapezoid", None) or np.trapz


# --------------------------------------------------------------------------- #
# Signal processing
# --------------------------------------------------------------------------- #
def resample_to_eeg(x, fs_in):
    """Resample (n_samples, n_ch) from fs_in to EEG_FS with a polyphase anti-alias filter."""
    fs_in = int(round(fs_in))
    if fs_in == EEG_FS:
        return x
    g = gcd(fs_in, EEG_FS)
    # Remove the DC offset first so the filter's zero-padding doesn't bend the edges
    # (the right edge is the most recent data, which the live raw plot shows)
    mean = x.mean(axis=0)
    return signal.resample_poly(x - mean, EEG_FS // g, fs_in // g, axis=0) + mean


def preprocess(eeg_uv, fs=EEG_FS, notch_hz=60.0):
    """Detrend and remove mains interference. eeg_uv: (n_samples, n_ch) in microvolts."""
    with np.errstate(all="ignore"):            # silence spurious macOS Accelerate matmul warnings
        x = signal.detrend(eeg_uv, axis=0)
    if notch_hz and notch_hz < fs / 2:
        b, a = signal.iirnotch(notch_hz, Q=30, fs=fs)
        x = signal.filtfilt(b, a, x, axis=0)
    return x


def band_powers(eeg_uv, fs=EEG_FS, notch_hz=60.0):
    """
    Compute band powers for each channel.

    Returns
    -------
    absolute : dict band -> array(n_ch)   power in uV^2
    relative : dict band -> array(n_ch)   fraction of total 1-45 Hz power
    freqs, psd : the Welch spectrum (psd shape (n_freqs, n_ch))
    """
    x = preprocess(eeg_uv, fs, notch_hz)
    nperseg = min(len(x), 2 * fs)              # 2 s segments -> 0.5 Hz resolution
    freqs, psd = signal.welch(x, fs=fs, nperseg=nperseg, axis=0)

    absolute = {}
    for name, (lo, hi) in BANDS.items():
        m = (freqs >= lo) & (freqs < hi)
        absolute[name] = _trapz(psd[m], freqs[m], axis=0)

    total = sum(absolute.values())
    total = np.where(total > 0, total, np.nan)
    relative = {k: v / total for k, v in absolute.items()}
    return absolute, relative, freqs, psd


def is_clipped(raw):
    """Per-channel bad-contact flag from raw audio values (n_samples, n_ch)."""
    return np.mean(np.abs(raw) >= CLIP_LEVEL, axis=0) > CLIP_FRACTION


def bandlimit(eeg_uv, fs=EEG_FS):
    """1-45 Hz band-pass (the range the brainwave bands cover)."""
    sos = signal.butter(4, [1.0, 45.0], btype="band", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, eeg_uv, axis=0)


def channel_quality(eeg_uv, clipped, fs=EEG_FS):
    """
    Contact quality per electrode, like the Muse app's "horseshoe" indicator.

    Uses the standard deviation of the 1-45 Hz signal: awake EEG with good contact is
    roughly 5-30 uV. Loose electrodes, movement and muscle tension push it far higher;
    a disconnected / flat channel pushes it toward zero.

    Returns status (array of GOOD/NOISY/BAD) and std (array, uV).
    """
    std = bandlimit(eeg_uv, fs).std(axis=0)
    status = np.where(std > NOISY_STD_UV, NOISY, GOOD)
    status = np.where(clipped | (std > BAD_STD_UV) | (std < FLAT_STD_UV), BAD, status)
    return status, std


def combine(absolute, status):
    """
    Single overall reading (what the Muse app shows), averaged across the electrodes
    with GOOD contact. If none are good it falls back to NOISY ones, then to all.

    Relative power is averaged per electrode so each one counts equally (averaging
    absolute power instead would let the noisiest/loudest electrode dominate).

    Returns avg_abs (band -> float), avg_rel (band -> float), used (channel indices),
    level (worst quality among the electrodes used: GOOD, NOISY or BAD).
    """
    status = np.asarray(status)
    for level in (GOOD, NOISY, BAD):
        used = np.flatnonzero(status <= level)
        if len(used):
            break
    avg_abs = {b: float(np.mean(v[used])) for b, v in absolute.items()}
    total = sum(absolute.values())[used]
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = {b: absolute[b][used] / total for b in BANDS}
    avg_rel = {b: float(np.nanmean(v)) if np.any(np.isfinite(v)) else float("nan") for b, v in rel.items()}
    return avg_abs, avg_rel, used, level


def band_waveforms(eeg_uv, fs=EEG_FS):
    """Optional helper: band-pass filtered time series per band, dict band -> (n_samples, n_ch)."""
    out = {}
    for name, (lo, hi) in BANDS.items():
        sos = signal.butter(4, [lo, hi], btype="band", fs=fs, output="sos")
        out[name] = signal.sosfiltfilt(sos, eeg_uv, axis=0)
    return out


# --------------------------------------------------------------------------- #
# Live audio input
# --------------------------------------------------------------------------- #
class AudioRingBuffer:
    """Thread-safe ring buffer fed by the sounddevice callback."""

    def __init__(self, n_samples, n_ch):
        self.buf = np.zeros((n_samples, n_ch), dtype=np.float32)
        self.n = n_samples
        self.pos = 0
        self.filled = 0
        self.lock = threading.Lock()

    def push(self, block):
        with self.lock:
            k = len(block)
            if k >= self.n:
                self.buf[:] = block[-self.n:]
                self.pos, self.filled = 0, self.n
                return
            end = self.pos + k
            if end <= self.n:
                self.buf[self.pos:end] = block
            else:
                cut = self.n - self.pos
                self.buf[self.pos:] = block[:cut]
                self.buf[:k - cut] = block[cut:]
            self.pos = end % self.n
            self.filled = min(self.n, self.filled + k)

    def snapshot(self):
        with self.lock:
            data = np.roll(self.buf, -self.pos, axis=0)
            return data[self.n - self.filled:].copy()


def find_device(sd, name_hint):
    if name_hint is not None and name_hint.isdigit():
        return int(name_hint)
    hint = (name_hint or "muse").lower()
    for i, d in enumerate(sd.query_devices()):
        if hint in d["name"].lower() and d["max_input_channels"] >= 4:
            return i
    sys.exit(f"Could not find a 4-channel input device matching '{hint}'. "
             f"Run with --list-devices and pass --device <index>.")


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #
def _bar(frac, width=40):
    if not np.isfinite(frac):
        return "·" * width
    n = int(round(min(max(frac, 0.0), 1.0) * width))
    return "█" * n + "·" * (width - n)


def _dominant(rel):
    vals = {b: v for b, v in rel.items() if np.isfinite(v)}
    return max(vals, key=vals.get) if vals else "-"


_ANSI = {GOOD: "\x1b[32m", NOISY: "\x1b[33m", BAD: "\x1b[31m"}
_RESET = "\x1b[0m"


def contact_line(status, std):
    """One-line coloured contact indicator, e.g. 'TP9 ● good 12uV   AF7 ● BAD 340uV ...'."""
    parts = []
    for c, ch in enumerate(CHANNELS):
        s = int(status[c])
        parts.append(f"{ch} {_ANSI[s]}● {QUALITY_LABELS[s]:<5}{_RESET} {std[c]:5.0f}uV")
    return "   ".join(parts)


def print_display(t, relative, status, std, avg_abs, avg_rel, used, level, per_channel):
    lines = [f"t = {t:7.1f} s", ""]
    lines.append("CONTACT  " + contact_line(status, std))
    lines.append("")
    lines.append("OVERALL (average of electrodes with good contact)")
    for b in BANDS:
        lines.append(f"  {b:<6} {_bar(avg_rel[b])} {100 * avg_rel[b]:5.1f}%  [{avg_abs[b]:8.1f} uV^2]")
    lines.append(f"  dominant: {_dominant(avg_rel)}")
    lines.append(f"  using: {', '.join(CHANNELS[i] for i in used)}")
    if level == NOISY:
        lines.append(f"  {_ANSI[NOISY]}(!) no electrode has good contact - using noisy ones, treat with caution{_RESET}")
    elif level == BAD:
        lines.append(f"  {_ANSI[BAD]}(!) no usable electrode - reading is NOT reliable, adjust the headband{_RESET}")

    if per_channel:
        lines.append("")
        lines.append("PER ELECTRODE (relative %)")
        lines.append("        " + "".join(f"{b:>8}" for b in BANDS) + "   dominant")
        for c, ch in enumerate(CHANNELS):
            rel_c = {b: float(relative[b][c]) for b in BANDS}
            row = f"  {ch:<5} " + "".join(f"{100 * rel_c[b]:7.1f}%" for b in BANDS)
            row += f"   {_dominant(rel_c):<6}" + f"  {_ANSI[int(status[c])]}{QUALITY_LABELS[int(status[c])]}{_RESET}"
            lines.append(row)

    # Cursor home, overwrite each line, clear the rest of the screen (no flicker)
    out = "\x1b[H" + "".join(line + "\x1b[K\n" for line in lines) + "\x1b[J"
    sys.stdout.write(out)
    sys.stdout.flush()


class CsvLogger:
    def __init__(self, path, smooth_s):
        self.f = open(path, "w", newline="")
        self.w = csv.writer(self.f)
        header = ["time_s", "channels_used"]
        for ch in CHANNELS:
            header += [f"{ch}_quality", f"{ch}_std_uV"]
        for b in BANDS:
            header += [f"avg_{b}_abs_uV2", f"avg_{b}_rel"]
        # moving average of avg_<band>_rel over the last smooth_s seconds (0-1, bands sum to 1)
        header += [f"avg_{b}_rel_{smooth_s:g}s" for b in BANDS]
        for ch in CHANNELS:
            for b in BANDS:
                header += [f"{ch}_{b}_abs_uV2", f"{ch}_{b}_rel"]
        self.w.writerow(header)

    def write(self, t, absolute, relative, avg_abs, avg_rel, avg_rel_smooth, used, status, std):
        row = [f"{t:.3f}", "+".join(CHANNELS[i] for i in used)]
        for c in range(len(CHANNELS)):
            row += [QUALITY_LABELS[int(status[c])].lower(), f"{std[c]:.1f}"]
        for b in BANDS:
            row += [f"{avg_abs[b]:.4f}", f"{avg_rel[b]:.4f}"]
        row += [f"{avg_rel_smooth[b]:.4f}" for b in BANDS]
        for c in range(len(CHANNELS)):
            for b in BANDS:
                row += [f"{absolute[b][c]:.4f}", f"{relative[b][c]:.4f}"]
        self.w.writerow(row)
        self.f.flush()

    def close(self):
        self.f.close()


RAW_COLOR = "#3b6ea8"
BAD_COLOR = "#c0392b"
QUALITY_COLORS = {GOOD: RAW_COLOR, NOISY: "#d68910", BAD: BAD_COLOR}
QUALITY_TEXT_COLORS = {GOOD: "#1e8449", NOISY: "#d68910", BAD: BAD_COLOR}


def _make_figure(plt, figsize, smooth_s):
    """
    Top: overall band power, instantaneous. Middle: same, as a moving average over the
    last smooth_s seconds. Bottom: one raw EEG trace per electrode (shared x).
    """
    fig = plt.figure(figsize=figsize)
    outer = fig.add_gridspec(3, 1, height_ratios=[2, 2, len(CHANNELS)], hspace=0.32)
    inner = outer[2].subgridspec(len(CHANNELS), 1, hspace=0.25)
    ax_bands = fig.add_subplot(outer[0])
    ax_smooth = fig.add_subplot(outer[1], sharex=ax_bands)
    raw_axes = []
    for c, ch in enumerate(CHANNELS):
        ax = fig.add_subplot(inner[c], sharex=raw_axes[0] if raw_axes else None)
        ax.set_ylabel(f"{ch}\n(uV)")
        ax.grid(alpha=0.3)
        if c < len(CHANNELS) - 1:
            ax.tick_params(labelbottom=False)
        raw_axes.append(ax)
    ax_bands.set_title("Overall brainwaves - instantaneous (average of electrodes with good contact)",
                       fontsize=10)
    ax_smooth.set_title(f"Overall brainwaves - {smooth_s:g} s moving average", fontsize=10)
    for ax in (ax_bands, ax_smooth):
        ax.set_ylabel("relative power")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    ax_bands.tick_params(labelbottom=False)
    raw_axes[0].set_title("Raw EEG per electrode (DC offset removed)", fontsize=10)
    return fig, ax_bands, ax_smooth, raw_axes


class MovingAverage:
    """
    Trailing moving average of the band values over the last window_s seconds.
    Feed it one reading per update; it returns the smoothed reading (band -> float).

    Relative powers are fractions in 0-1 that sum to 1 across bands; their average
    keeps both properties, so the smoothed values are also 0-1 and sum to 1.
    """

    def __init__(self, window_s):
        self.window_s = window_s
        self.buf = deque()                     # (t, {band: value})

    def update(self, t, values):
        self.buf.append((t, values))
        while self.buf[0][0] <= t - self.window_s:
            self.buf.popleft()
        out = {}
        for b in BANDS:
            v = np.array([vals[b] for _, vals in self.buf], dtype=float)
            out[b] = float(np.nanmean(v)) if np.any(np.isfinite(v)) else float("nan")
        return out


class LivePlot:
    """Overall band power (instantaneous + moving average) over time, and live raw EEG per electrode."""

    def __init__(self, history_s, raw_s, smooth_s):
        import matplotlib.pyplot as plt
        self.plt = plt
        plt.ion()
        self.fig, self.ax_bands, self.ax_smooth, self.raw_axes = _make_figure(plt, (11, 12), smooth_s)
        self.history_s = history_s
        self.raw_s = raw_s
        self.smooth_s = smooth_s

        self.t = []
        self.hist = {b: [] for b in BANDS}
        self.smooth = {b: [] for b in BANDS}
        self.band_lines = {b: self.ax_bands.plot([], [], label=b, lw=1.5)[0] for b in BANDS}
        self.smooth_lines = {b: self.ax_smooth.plot([], [], label=b, lw=2)[0] for b in BANDS}
        self.ax_bands.legend(loc="upper left", ncol=len(BANDS), fontsize=8)
        self.ax_smooth.set_xlabel("time (s)")

        self.raw_lines = [ax.plot([], [], lw=0.7, color=RAW_COLOR)[0] for ax in self.raw_axes]
        self.raw_lim = [50.0] * len(CHANNELS)
        for ax in self.raw_axes:
            ax.set_xlim(-raw_s, 0)
        self.raw_axes[-1].set_xlabel("seconds ago")
        self.fig.subplots_adjust(left=0.09, right=0.98, top=0.96, bottom=0.06)

    def update(self, t, avg_rel, avg_rel_smooth, raw_uv, status):
        """avg_rel / avg_rel_smooth: band -> scalar (instantaneous / moving average).
        raw_uv: (n_samples, n_ch) at EEG_FS, most recent last.
        status: contact quality per electrode (GOOD/NOISY/BAD)."""
        # --- overall band power history
        self.t.append(t)
        for b in BANDS:
            self.hist[b].append(avg_rel[b])
            self.smooth[b].append(avg_rel_smooth[b])
        while self.t and self.t[-1] - self.t[0] > self.history_s:
            self.t.pop(0)
            for b in BANDS:
                self.hist[b].pop(0)
                self.smooth[b].pop(0)
        for b in BANDS:
            self.band_lines[b].set_data(self.t, self.hist[b])
            self.smooth_lines[b].set_data(self.t, self.smooth[b])
        self.ax_bands.set_xlim(max(0, t - self.history_s), max(self.history_s, t))

        # --- raw traces
        n = len(raw_uv)
        tt = (np.arange(n) - n) / EEG_FS
        for c, (ax, ln) in enumerate(zip(self.raw_axes, self.raw_lines)):
            y = raw_uv[:, c] - raw_uv[:, c].mean()
            ln.set_data(tt, y)
            # Auto-scale symmetric y-axis: expand instantly, shrink slowly (avoids jitter)
            target = max(1.2 * np.percentile(np.abs(y), 99.5), 10.0)
            self.raw_lim[c] = target if target > self.raw_lim[c] else 0.97 * self.raw_lim[c] + 0.03 * target
            ax.set_ylim(-self.raw_lim[c], self.raw_lim[c])
            s = int(status[c])
            ln.set_color(QUALITY_COLORS[s])
            ax.set_ylabel(f"{CHANNELS[c]} (uV)\n● {QUALITY_LABELS[s]}", color=QUALITY_TEXT_COLORS[s])

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def is_open(self):
        return self.plt.fignum_exists(self.fig.number)


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
def run_live(args):
    import sounddevice as sd

    dev = find_device(sd, args.device)
    info = sd.query_devices(dev)
    fs = int(args.samplerate or info["default_samplerate"])
    print(f"Using device {dev}: {info['name']} @ {fs} Hz")

    # Keep enough audio for both the band-power window and the raw-trace view
    ring = AudioRingBuffer(int(max(args.window, args.raw_seconds) * fs), 4)
    win_audio = int(args.window * fs)
    win_eeg = int(args.window * EEG_FS)
    raw_eeg = int(args.raw_seconds * EEG_FS)

    def callback(indata, frames, t, status):
        if status:
            print(status, file=sys.stderr)
        ring.push(indata[:, :4])

    logger = CsvLogger(args.csv, args.smooth) if args.csv else None
    plot = LivePlot(args.history, args.raw_seconds, args.smooth) if args.plot else None
    smoother = MovingAverage(args.smooth)

    sys.stdout.write("\x1b[2J")                # clear the screen once
    t0 = time.time()
    next_tick = time.time()
    with sd.InputStream(device=dev, channels=4, samplerate=fs, dtype="float32",
                        blocksize=int(fs * 0.02), callback=callback):
        try:
            while plot is None or plot.is_open():
                # Fixed-rate loop: wait until the next tick, accounting for processing time
                next_tick += args.interval
                delay = next_tick - time.time()
                if delay > 0:
                    if plot:
                        plot.plt.pause(delay)
                    else:
                        time.sleep(delay)
                else:
                    next_tick = time.time()    # running behind; don't try to catch up

                raw = ring.snapshot()
                if len(raw) < fs * 2:           # wait for at least 2 s of data
                    sys.stdout.write(f"\x1b[Hbuffering... {len(raw) / fs:.1f} s\x1b[K")
                    sys.stdout.flush()
                    continue
                clipped = is_clipped(raw[-win_audio:])
                eeg_all = resample_to_eeg(raw.astype(np.float64), fs) * args.scale
                eeg_uv = eeg_all[-win_eeg:]                 # band-power window
                absolute, relative, _, _ = band_powers(eeg_uv, notch_hz=args.notch)
                status, std = channel_quality(eeg_uv, clipped)
                avg_abs, avg_rel, used, level = combine(absolute, status)
                t = time.time() - t0
                avg_rel_smooth = smoother.update(t, avg_rel)

                print_display(t, relative, status, std, avg_abs, avg_rel, used, level, args.per_channel)
                if logger:
                    logger.write(t, absolute, relative, avg_abs, avg_rel, avg_rel_smooth, used, status, std)
                if plot:
                    plot.update(t, avg_rel, avg_rel_smooth, eeg_all[-raw_eeg:], status)
        except KeyboardInterrupt:
            pass
        finally:
            if logger:
                logger.close()


def run_file(args):
    from scipy.io import wavfile

    fs, data = wavfile.read(args.file)
    if data.ndim == 1 or data.shape[1] < 4:
        sys.exit("Expected a WAV with 4 channels (one per Muse electrode).")
    data = data[:, :4]
    # Normalise integer WAVs to the same -1..1 float range Audacity uses
    if np.issubdtype(data.dtype, np.integer):
        data = data / float(np.iinfo(data.dtype).max + 1)
    data = data.astype(np.float64)

    print(f"Loaded {args.file}: {len(data) / fs:.1f} s @ {fs} Hz")
    eeg_uv = resample_to_eeg(data, fs) * args.scale
    win = int(args.window * EEG_FS)
    step = max(1, int(args.interval * EEG_FS))
    raw_per_eeg = fs / EEG_FS

    logger = CsvLogger(args.csv, args.smooth) if args.csv else None
    smoother = MovingAverage(args.smooth)
    times, rel_hist, smooth_hist = [], [], []
    for start in range(0, len(eeg_uv) - win + 1, step):
        seg = eeg_uv[start:start + win]
        raw_seg = data[int(start * raw_per_eeg):int((start + win) * raw_per_eeg)]
        clipped = is_clipped(raw_seg)
        absolute, relative, _, _ = band_powers(seg, notch_hz=args.notch)
        status, std = channel_quality(seg, clipped)
        avg_abs, avg_rel, used, level = combine(absolute, status)
        t = (start + win) / EEG_FS
        avg_rel_smooth = smoother.update(t, avg_rel)
        times.append(t)
        rel_hist.append(avg_rel)
        smooth_hist.append(avg_rel_smooth)
        if logger:
            logger.write(t, absolute, relative, avg_abs, avg_rel, avg_rel_smooth, used, status, std)
    if logger:
        logger.close()
        print(f"Wrote {len(times)} windows to {args.csv}")

    # Whole-recording summary
    absolute, relative, _, _ = band_powers(eeg_uv, notch_hz=args.notch)
    clipped = is_clipped(data)
    status, std = channel_quality(eeg_uv, clipped)
    avg_abs, avg_rel, used, level = combine(absolute, status)
    print(f"\nWhole recording ({len(eeg_uv) / EEG_FS:.1f} s):")
    print_display(len(eeg_uv) / EEG_FS, relative, status, std, avg_abs, avg_rel, used, level, True)

    if args.plot:
        import matplotlib.pyplot as plt
        fig, ax_bands, ax_smooth, raw_axes = _make_figure(plt, (12, 12), args.smooth)
        for b in BANDS:
            ax_bands.plot(times, [r[b] for r in rel_hist], label=b, lw=1.5)
            ax_smooth.plot(times, [r[b] for r in smooth_hist], label=b, lw=2)
        ax_bands.legend(loc="upper left", ncol=len(BANDS), fontsize=8)
        ax_smooth.set_xlabel("time (s)")
        tt = np.arange(len(eeg_uv)) / EEG_FS
        for c, ax in enumerate(raw_axes):
            y = eeg_uv[:, c] - eeg_uv[:, c].mean()
            ax.plot(tt, y, lw=0.5, color=QUALITY_COLORS[int(status[c])])
            lim = max(1.2 * np.percentile(np.abs(y), 99.5), 10.0)
            ax.set_ylim(-lim, lim)
        raw_axes[-1].set_xlabel("time (s)")
        fig.subplots_adjust(left=0.09, right=0.98, top=0.96, bottom=0.06)
        plt.show()


def main():
    p = argparse.ArgumentParser(description="Muse 2 EEG (via ESP32 USB audio) -> brainwave band powers")
    p.add_argument("--file", help="analyse a 4-channel WAV instead of live input")
    p.add_argument("--device", help="audio device index or name substring (default: 'muse')")
    p.add_argument("--samplerate", type=int, help="override capture sample rate (default: device rate)")
    p.add_argument("--window", type=float, default=4.0,
                   help="analysis window in seconds; each reading covers this much history (default 4)")
    p.add_argument("--interval", type=float, default=0.1, help="refresh interval in seconds (default 0.1)")
    p.add_argument("--notch", type=float, default=60.0, help="mains notch frequency, 0 to disable (default 60)")
    p.add_argument("--scale", type=float, default=UV_PER_UNIT, help="microvolts per audio unit (default 2000)")
    p.add_argument("--per-channel", action="store_true", help="also show each electrode separately")
    p.add_argument("--csv", help="write band powers to this CSV file")
    p.add_argument("--plot", action="store_true", help="show a live / summary plot")
    p.add_argument("--history", type=float, default=60.0, help="seconds of band-power history in the live plot")
    p.add_argument("--raw-seconds", type=float, default=5.0, help="seconds of raw EEG shown in the live plot (default 5)")
    p.add_argument("--smooth", type=float, default=5.0,
                   help="moving-average window in seconds for the smoothed plot and the "
                        "avg_<band>_rel_<N>s CSV columns (default 5)")
    p.add_argument("--list-devices", action="store_true")
    args = p.parse_args()

    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return
    if args.file:
        run_file(args)
    else:
        run_live(args)


if __name__ == "__main__":
    main()
