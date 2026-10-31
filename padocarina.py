#!/usr/bin/env python3
"""padocarina: play a PlayStation-style controller like an ocarina.

Usage: python padocarina.py [--debug] [--raw]
  --raw    ignore SDL's standard gamepad mapping and use the CONFIG indices
  --debug  print raw button/axis indices as you press/move them
"""
import argparse
import os
import sys
import threading
import time

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="importlib")

import numpy as np
import pygame
import sounddevice as sd

try:
    from pygame._sdl2 import controller as sdl_controller
except ImportError:
    sdl_controller = None

# ---------------------------------------------------------------------------
# CONFIG. Indices are only used with --raw (or if SDL doesn't recognize the
# pad as a gamepad). Run with --debug to find yours.
# ---------------------------------------------------------------------------
CONFIG = {
    "buttons": {"cross": 0, "circle": 1, "square": 2, "triangle": 3},
    "quit_buttons": [5, 6],          # PS, Options
    "button_l1": 9,                  # sharp
    "button_r1": 10,                 # freeze + shimmer (D-pad uses hat 0)
    "button_r3": 8,                  # hold + D-pad Left/Right = change scale
    "button_l3": 7,                  # atmosphere reverb toggle (also the touchpad click)
    "axis_lx": 0,
    "axis_ly": 1,
    "axis_rx": 2,
    "axis_ry": 3,
    "axis_l2": 4,
    "axis_r2": 5,
    "scale": "pentatonic",           # starting scale, any key of SCALES
    "trigger_threshold": 0.5,        # normalized 0..1, for non-analog triggers
    "analog_r2": True,               # R2 swoops up the octave as you squeeze
    "analog_l2": False,              # L2 slides the fifth instead of snapping
    "trigger_lo": 0.1,               # analog: below this = no shift
    "trigger_hi": 0.9,               # analog: above this = locked on the shifted note
    "stick_deadzone": 0.08,
    "bend_range": 2.0,               # semitones, left stick Y
    "bend_snap": "semitone",         # "semitone", "scale" (in-key notes only) or "off"
    "bend_capture": 0.3,             # how much stick travel each snap target grabs (0..0.5)
    "harmony_level": 0.55,           # second voice a third above, toggled with D-pad Right
    "vibrato_max": 0.6,              # semitones of extra vibrato, left stick X
    "sample_rate": 48000,
    "blocksize": 256,
    "master_gain": 0.35,
    "attack_s": 0.045,
    "release_s": 0.14,
    "glide_s": 0.03,                 # note-to-note portamento time constant
    "reverb_mix": 1.0,               # 0 = dry
    "reverb_room": 0.85,             # tail length (<1)
    "atmos_room": 0.95,              # tail length with the atmosphere toggle on
    "atmos_boost": 0.9,              # extra reverb level with atmosphere on (0.9 = +90%)
    "shimmer": 0.3,                  # octave-up feedback into the reverb while R1 is held
    "voice": "ocarina",              # starting voice, any key of VOICES
    "ensemble_level": 0.35,          # two detuned, panned extra ocarinas, toggled with D-pad Up
    "drone_level": 0.07,             # C3 + G3 pad, toggled with D-pad Down
    "delay_s": 0.33,                 # ping-pong echo time, toggled with D-pad Left
    "delay_feedback": 0.5,
    "delay_mix": 0.4,
    "rumble": True,
}

# Rising clockwise around the diamond from the bottom (semitones from C4).
# Each scale: (plain notes, notes while L2 is held).
# Cycled in this order with R3 + D-pad Left/Right.
SCALES = {
    "pentatonic": ([0, 2, 4, 7], [9, 12, 14, 16]),          # C D E G | A C D E
    "minor pentatonic": ([0, 3, 5, 7], [10, 12, 15, 17]),   # C Eb F G | Bb C Eb F
    "blues": ([0, 3, 5, 6], [7, 10, 12, 15]),               # C Eb F F# | G Bb C Eb
    "japanese": ([0, 1, 5, 7], [8, 12, 13, 17]),            # C Db F G | Ab C Db F  (In scale)
    "major": ([0, 2, 4, 5], [7, 9, 11, 12]),                # C D E F | G A B C
}
_ORDER = ("cross", "square", "triangle", "circle")


def set_scale(name):
    global NOTES, NOTES_L2, SCALE_PCS
    CONFIG["scale"] = name
    plain, upper = SCALES[name]
    NOTES = dict(zip(_ORDER, plain))
    NOTES_L2 = dict(zip(_ORDER, upper))
    SCALE_PCS = tuple(sorted({m % 12 for m in plain + upper}))


set_scale(CONFIG["scale"])
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_hz(m):
    return 440.0 * 2.0 ** ((m - 69) / 12.0)


def note_name(m):
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"


def shape(v):
    """Deadzone + curve: fine control near center, full range at the edge."""
    dz = CONFIG["stick_deadzone"]
    a = abs(v)
    if a < dz:
        return 0.0
    a = min((a - dz) / (1.0 - dz), 1.0)
    return a ** 1.6 * (1 if v > 0 else -1)


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------
class SynthState:
    """Written by the input loop, read by the audio callback, under a lock."""

    def __init__(self):
        self.lock = threading.Lock()
        self.note = None      # MIDI note or None
        self.bend = 0.0       # semitones
        self.vibrato = 0.0    # 0..1 extra vibrato
        self.pressure = 1.0   # breath pressure, 1 = normal
        self.press_id = 0     # increments on every new button press
        self.shimmer = False  # R1 held: freeze + shimmer
        self.ensemble = False
        self.drone = False
        self.delay = False
        self.harmony = None   # semitones above the lead note, or None
        self.atmos = False    # big, long reverb (no shimmer)
        self.voice = CONFIG["voice"]
        self.vowel = 2.0      # choir vowel position, 0 (oo) .. 4 (ee)


class Reverb:
    """Small stereo Freeverb-style reverb. All delays exceed the block size,
    so each feedback loop can be computed a whole block at a time."""

    COMBS = [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617]
    ALLPASSES = [556, 441, 341]

    def __init__(self, sr, block):
        scale = sr / 44100.0
        self.chunk = block
        self.channels = []
        for spread in (0, 23):
            combs = [np.zeros(max(int((d + spread) * scale), block + 2) + 2) for d in self.COMBS]
            aps = [np.zeros(max(int((d + spread) * scale), block + 2)) for d in self.ALLPASSES]
            self.channels.append((combs, aps))

    def _run(self, x, combs, aps, freeze, room=None):
        # freeze (0..1) pushes the feedback toward 1 and opens the damping, so the tail hangs.
        room = room or CONFIG["reverb_room"]
        g = room + (0.998 - room) * freeze
        damp = 0.6 - 0.45 * freeze
        F = len(x)
        xin = x * 0.015 * (1.0 - 0.6 * freeze)
        acc = np.zeros(F)
        for i, h in enumerate(combs):
            lp = 0.25 * h[:F] + 0.5 * h[1:F + 1] + 0.25 * h[2:F + 2]  # darkens each pass
            y = xin + g * ((1 - damp) * h[2:F + 2] + damp * lp)
            combs[i] = np.concatenate((h[F:], y))
            acc += y
        for i, h in enumerate(aps):
            d = h[:F]
            aps[i] = np.concatenate((h[F:], acc + 0.5 * d))
            acc = d - acc
        return acc

    def process(self, x, freeze=0.0, room=None):
        outs = []
        for combs, aps in self.channels:
            parts = [self._run(x[i:i + self.chunk], combs, aps, freeze, room)
                     for i in range(0, len(x), self.chunk)]
            outs.append(np.concatenate(parts))
        return outs


class Shimmer:
    """Octave-up pitch shifter: two crossfaded read heads moving at double speed."""

    W, R = 4096, 16384

    def __init__(self):
        self.buf = np.zeros(self.R)
        self.pos = 0
        self.ph = 0
        self.last = 0.0

    def process(self, x):
        W, R = self.W, self.R
        F = len(x)
        n = np.arange(F)
        self.buf[(self.pos + n) % R] = x
        d1 = (self.ph - n) % W
        d2 = (d1 + W // 2) % W
        out = (self.buf[(self.pos + n - d1) % R] * np.sin(np.pi * d1 / W) ** 2
               + self.buf[(self.pos + n - d2) % R] * np.sin(np.pi * d2 / W) ** 2)
        self.ph = (self.ph - F) % W
        self.pos += F
        out = 0.5 * (out + np.concatenate(([self.last], out[:-1])))  # tame the fizz
        self.last = float(out[-1])
        return out


class PingPong:
    """Stereo echo bouncing left/right. Delay exceeds the block size, so it runs per block."""

    def __init__(self, sr):
        self.sr = sr
        D = int(CONFIG["delay_s"] * sr)
        self.l = np.zeros(D + 1)
        self.r = np.zeros(D + 1)
        self.send = 0.0

    def process(self, x, on):
        F = len(x)
        n = np.arange(1, F + 1)
        ramp = n / (0.02 * self.sr)
        send = np.minimum(self.send + ramp, 1.0) if on else np.maximum(self.send - ramp, 0.0)
        dl = 0.5 * (self.l[:F] + self.l[1:F + 1])
        dr = 0.5 * (self.r[:F] + self.r[1:F + 1])
        fb = CONFIG["delay_feedback"]
        self.l = np.concatenate((self.l[F:], x * send + fb * dr))
        self.r = np.concatenate((self.r[F:], fb * dl))
        self.send = float(send[-1])
        mix = CONFIG["delay_mix"]
        return mix * dl, mix * dr


class Drone:
    """Soft C3 + G3 pad (one voice slightly detuned for a slow beat) that swells gently."""

    FREQS = (130.81, 196.00, 130.81 * 1.0015)
    AMPS = (1.0, 0.6, 0.5)

    def __init__(self, sr):
        self.sr = sr
        self.ph = np.zeros(len(self.FREQS))
        self.swell = 0.0
        self.level = 0.0

    def process(self, frames, on):
        n = np.arange(1, frames + 1)
        ramp = n / (2.0 * self.sr)  # 2 s fade
        lvl = np.minimum(self.level + ramp, 1.0) if on else np.maximum(self.level - ramp, 0.0)
        self.level = float(lvl[-1])
        if self.level == 0.0 and not on:
            return np.zeros(frames)
        sig = np.zeros(frames)
        for i, (f, a) in enumerate(zip(self.FREQS, self.AMPS)):
            ph = self.ph[i] + 2 * np.pi * f * n / self.sr
            sig += a * (np.sin(ph) + 0.15 * np.sin(2 * ph))
            self.ph[i] = ph[-1] % (2 * np.pi)
        sw = self.swell + 2 * np.pi * 0.12 * n / self.sr
        self.swell = float(sw[-1] % (2 * np.pi))
        return CONFIG["drone_level"] * lvl * (1.0 + 0.15 * np.sin(sw)) * sig


# Per-voice envelope and feel. attack/release None = use CONFIG's ocarina values.
# Cycled in this order with R3 + D-pad Up/Down.
VOICES = {
    "ocarina": dict(attack=None, release=None, gain=1.0, vibrato=1.0, scoop=True),
    "choir": dict(attack=0.15, release=0.45, gain=1.1, vibrato=1.0, scoop=True),
    # Blade Runner / CS-80 style: two detuned saws through a resonant low-pass that
    # swells open like brass, breath opens it further, in a huge dark hall.
    "vangelis": dict(attack=0.12, release=1.4, gain=1.0, vibrato=1.0, scoop=False,
                     glide=0.05, vib_depth=0.1, vib_rate=5.5, room=0.93, reverb=1.6),
}

# Choir vowels: formant centre frequencies (Hz), swept with right stick X.
VOWELS = np.array([[300, 870, 2240],    # oo
                   [450, 800, 2830],    # oh
                   [730, 1090, 2440],   # ah
                   [530, 1840, 2480],   # eh
                   [270, 2290, 3010]])  # ee
VOWEL_NAMES = ("oo", "oh", "ah", "eh", "ee")


def pan_gains(pans):
    pans = np.asarray(pans, dtype=float)
    return np.sqrt((1 - pans) / 2)[:, None], np.sqrt((1 + pans) / 2)[:, None]


class Synth:
    CHOIR_CENTS = np.array([-8, 0, 8])
    CHOIR_L, CHOIR_R = pan_gains([-0.7, 0, 0.7])

    def __init__(self, state):
        self.s = state
        self.sr = CONFIG["sample_rate"]
        self.log_f = np.log2(midi_to_hz(60))
        self.bend = 0.0
        self.env = 0.0
        self.pressure = 1.0
        self.vib_phase = 0.0
        self.vib_depth = 0.0
        self.note_age = 0.0
        self.press_id = 0
        self.rng = np.random.default_rng()
        self.reverb = Reverb(self.sr, CONFIG["blocksize"])
        self.shimmer = Shimmer()
        self.delay = PingPong(self.sr)
        self.drone = Drone(self.sr)
        self.shim_amt = 0.0
        self.shim_prev = np.zeros(CONFIG["blocksize"])
        self.wet_gain = 1.0
        self.out_gain = 1.0
        self.fenv = 0.0
        self.atmos_amt = 0.0
        self.fenv_rising = False
        self.vg_amps = {}
        self.xruns = 0
        self.h_lvl = 0.0
        self.h_off = 4.0
        self.ens_drift = 0.0
        self.ens_lvl = 0.0
        self.acc = {}          # oscillator phases, keyed by (player, voice)
        self.air = {}          # ocarina air-noise filter tails, keyed like acc
        self.jitter = np.zeros(2)  # slow random drift: (pitch in semitones, amplitude)
        k = np.bartlett(2 * int(self.sr / 400) + 1)[1:-1]
        self.AIR_KERNEL = k / np.sqrt((k ** 2).sum())
        self.vowel = 2.0

    def _phases(self, key, freq, ratios):
        """(oscillators, frames) phase array; each oscillator runs at freq * ratio."""
        ratios = np.asarray(ratios, dtype=float)
        acc = self.acc.get(key)
        if acc is None or len(acc) != len(ratios):
            acc = self.rng.uniform(0, 2 * np.pi, len(ratios))  # free-running start phases
        ph = acc[:, None] + np.cumsum(2 * np.pi * freq / self.sr)[None, :] * ratios[:, None]
        self.acc[key] = ph[:, -1] % (2 * np.pi)
        return ph

    def voice_tone(self, voice, player, freq, b):
        """One player's sound for this block as (left, right)."""
        key = (player, voice)
        if voice == "vangelis":
            # Saw harmonics weighted by a resonant low-pass response: the filter is
            # applied in the spectrum, so it's alias-free and sweeps without zipper noise.
            ratios = 2.0 ** (np.array([-7.0, 7.0]) / 1200)
            ph = self._phases(key, freq, ratios)
            f0 = float(freq[0])
            K = int(max(1, min(32, 0.45 * self.sr / (f0 * 1.01))))
            k = np.arange(1, K + 1)
            wc = np.clip(f0 * (0.9 + 5.0 * self.fenv + 3.0 * float(b[-1])), 150.0, 12000.0)
            x = k * f0 / wc
            H = 1.0 / np.sqrt((1 - x ** 2) ** 2 + (x / 1.6) ** 2) / np.sqrt(1 + x ** 4)
            amps = H / k
            prev = self.vg_amps.get(key)
            if prev is None or len(prev) != K:
                prev = amps
            self.vg_amps[key] = amps
            t = np.linspace(0.0, 1.0, ph.shape[1])
            A = prev[:, None] + (amps - prev)[:, None] * t[None, :]
            osc = np.einsum("kf,jkf->jf", A, np.sin(k[None, :, None] * ph[:, None, :]))
            return 0.4 * (0.8 * osc[0] + 0.45 * osc[1]), 0.4 * (0.45 * osc[0] + 0.8 * osc[1])
        if voice == "choir":
            # Three singers; each is a stack of harmonics shaped by the vowel's formants.
            ratios = 2.0 ** (self.CHOIR_CENTS / 1200)
            ph = self._phases(key, freq, ratios)
            lo, hi = int(np.floor(self.vowel)), int(np.ceil(self.vowel))
            fm = VOWELS[lo] + (VOWELS[hi] - VOWELS[lo]) * (self.vowel - lo)
            f0 = float(freq[0])
            K = int(max(1, min(32, 4200 / f0)))
            k = np.arange(1, K + 1)
            hz = k * f0
            amps = sum(g * np.exp(-((hz - f) / w) ** 2)
                       for f, w, g in zip(fm, (90, 120, 160), (1.0, 0.6, 0.25 + 0.3 * float(b[-1]))))
            amps = (amps + 0.02) / np.sqrt(k)
            amps /= np.sqrt((amps ** 2).sum())
            out = np.einsum("k,jkf->jf", amps, np.sin(k[None, :, None] * ph[:, None, :]))
            return 0.7 * (out * self.CHOIR_L).sum(0), 0.7 * (out * self.CHOIR_R).sum(0)
        # Ocarina: a vessel flute (Helmholtz resonator), so almost a pure sine with only
        # faint overtones that grow when you blow harder.
        ph = self._phases(key, freq, [1.0])[0]
        tone = (np.sin(ph) + (0.015 + 0.05 * b) * np.sin(2 * ph)
                + (0.008 + 0.025 * b) * np.sin(3 * ph))
        # Air: noise narrowed around the pitch itself (slow noise riding the sine), so it
        # sounds like a breathy, living tone rather than background hiss.
        tail = self.air.get(key)
        if tail is None:
            tail = np.zeros(len(self.AIR_KERNEL) - 1)
        raw = np.concatenate((tail, self.rng.standard_normal(len(ph))))
        self.air[key] = raw[-(len(self.AIR_KERNEL) - 1):]
        slow = np.convolve(raw, self.AIR_KERNEL, "valid")
        tone = tone + (0.04 + 0.08 * b) * slow * np.sin(ph + 0.6)
        return tone, tone

    def callback(self, outdata, frames, time_info, status):
        if status and status.output_underflow:
            self.xruns += 1  # audio dropout (heard as crackle)
        with self.s.lock:
            note, bend, vib_amt = self.s.note, self.s.bend, self.s.vibrato
            pressure, press_id = self.s.pressure, self.s.press_id
            shimmer_on, drone_on, delay_on = self.s.shimmer, self.s.drone, self.s.delay
            atmos_on = self.s.atmos
            ensemble_on = self.s.ensemble
            harmony = self.s.harmony
            voice, vowel = self.s.voice, self.s.vowel
        spec = VOICES[voice]

        sr = self.sr
        n = np.arange(1, frames + 1)

        if note is not None:
            target = np.log2(midi_to_hz(note))
            if press_id != self.press_id:
                self.fenv_rising = True
                self.note_age = 0.0        # vibrato blooms again on new notes
                if self.env < 1e-3 and spec["scoop"]:
                    self.log_f = target - 0.2 / 12  # scoop up into the first note
            self.note_age += frames / sr
        else:
            target = self.log_f
        self.press_id = press_id

        # Envelope: linear ramps toward the gate.
        attack = spec["attack"] or CONFIG["attack_s"]
        release = spec["release"] or CONFIG["release_s"]
        if note is not None:
            env = np.minimum(self.env + n / (attack * sr), 1.0)
        else:
            env = np.maximum(self.env - n / (release * sr), 0.0)

        self.vowel += (vowel - self.vowel) * min(1.0, frames / (0.06 * sr))

        # Portamento between notes (exponential approach in log-frequency).
        a = np.exp(-1.0 / (spec.get("glide", CONFIG["glide_s"]) * sr))
        log_f = target + (self.log_f - target) * a ** n

        # Bend and pressure smoothing so stick movement feels fluid, not steppy.
        bend_arr = bend + (self.bend - bend) * np.exp(-1.0 / (0.015 * sr)) ** n
        pr = pressure + (self.pressure - pressure) * np.exp(-1.0 / (0.04 * sr)) ** n

        # Vibrato: a natural one that fades in on held notes, plus left stick X.
        # Breath is never perfectly steady: a slow random wander in pitch and loudness.
        j_prev = self.jitter.copy()
        self.jitter = 0.97 * self.jitter + 0.03 * self.rng.standard_normal(2) * (0.25, 0.6)
        jit = j_prev[:, None] + (self.jitter - j_prev)[:, None] * (n / frames)[None, :]

        # A wind player's vibrato is mostly in the breath (loudness), only a little in pitch.
        bloom = spec["vibrato"] * np.clip((self.note_age - 0.35) / 0.7, 0.0, 1.0)
        auto = spec.get("vib_depth", 0.05) * bloom
        depth_target = auto + vib_amt * CONFIG["vibrato_max"]
        depth = np.linspace(self.vib_depth, depth_target, frames)
        rate = spec.get("vib_rate", 5.0) + 1.7 * vib_amt + 0.3 * self.jitter[0]
        vib_ph = self.vib_phase + 2 * np.pi * rate * n / sr
        vib = depth * np.sin(vib_ph)

        # Blowing harder pushes an ocarina sharp, softer goes flat (about ±15 cents).
        breath_pitch = 0.15 * (pr - 1.0) if voice == "ocarina" else 0.0
        freq = 2.0 ** (log_f + (bend_arr + vib + jit[0] * 0.12 + breath_pitch) / 12.0)
        b = np.clip(pr / 2.0, 0.0, 1.0)
        amp = pr * (1.0 + (0.1 * bloom + 0.15 * vib_amt) * np.sin(vib_ph) + 0.06 * jit[1])
        level = CONFIG["master_gain"] * spec["gain"] * amp * env

        # Filter envelope (vangelis): swells open over ~0.2 s, settles back a little,
        # closes slowly after release. Restarts on each new press.
        dt = frames / sr
        if note is None:
            self.fenv += (0.0 - self.fenv) * (1 - np.exp(-dt / 0.6))
        elif self.fenv_rising:
            self.fenv += (1.05 - self.fenv) * (1 - np.exp(-dt / 0.2))
            if self.fenv >= 1.0:
                self.fenv_rising = False
        else:
            self.fenv += (0.55 - self.fenv) * (1 - np.exp(-dt / 0.8))
        dry_l, dry_r = self.voice_tone(voice, "lead", freq, b)

        # Harmony voice: follows the lead (bends, vibrato and all) an in-scale third up.
        step = n / (0.03 * sr)
        h_lvl = (np.minimum(self.h_lvl + step, 1.0) if harmony is not None
                 else np.maximum(self.h_lvl - step, 0.0))
        h_target = self.h_off if harmony is None else harmony
        h_off = h_target + (self.h_off - h_target) * a ** n
        self.h_lvl, self.h_off = float(h_lvl[-1]), float(h_off[-1])
        if self.h_lvl > 0 or harmony is not None:
            hl, hr = self.voice_tone(voice, "harmony", freq * 2.0 ** (h_off / 12.0), b)
            w2 = CONFIG["harmony_level"] * h_lvl
            dry_l, dry_r = dry_l + w2 * hl, dry_r + w2 * hr

        # Ensemble: two more players a few cents sharp/flat, slowly drifting, panned
        # left and right, like a small group playing in unison.
        ramp = n / (0.2 * sr)
        e_lvl = (np.minimum(self.ens_lvl + ramp, 1.0) if ensemble_on
                 else np.maximum(self.ens_lvl - ramp, 0.0))
        self.ens_lvl = float(e_lvl[-1])
        if self.ens_lvl > 0 or ensemble_on:
            dr = self.ens_drift + 2 * np.pi * 0.23 * n / sr
            self.ens_drift = float(dr[-1] % (2 * np.pi))
            w = CONFIG["ensemble_level"] * e_lvl
            for i, sign in enumerate((1.0, -1.0)):
                cents = sign * (4.0 + 1.5 * np.sin(dr + i * 1.7))
                el, er = self.voice_tone(voice, f"ens{i}", freq * 2.0 ** (cents / 1200), b)
                mono = 0.5 * (el + er)
                dry_l = dry_l + w * mono * (1.4 if i == 0 else 0.3)
                dry_r = dry_r + w * mono * (0.3 if i == 0 else 1.4)

        dry_l, dry_r = level * dry_l, level * dry_r
        dry = 0.5 * (dry_l + dry_r)
        drone = self.drone.process(frames, drone_on)

        echo_l, echo_r = self.delay.process(dry, delay_on)

        # Freeze fades in/out over ~150 ms; while frozen the reverb is fed its own
        # tail shifted up an octave (shimmer).
        step = frames / (0.15 * sr)
        self.shim_amt = (min(self.shim_amt + step, 1.0) if shimmer_on
                         else max(self.shim_amt - step, 0.0))
        shim = self.shim_prev[:frames] if len(self.shim_prev) >= frames else np.zeros(frames)
        rev_in = (dry + 0.25 * drone + 0.5 * (echo_l + echo_r)
                  + CONFIG["shimmer"] * self.shim_amt * shim)
        # Atmosphere: fades over ~0.5 s into a longer hall with more of it in the mix.
        step = frames / (0.5 * sr)
        self.atmos_amt = (min(self.atmos_amt + step, 1.0) if atmos_on
                          else max(self.atmos_amt - step, 0.0))
        room = spec.get("room", CONFIG["reverb_room"])
        room += (CONFIG["atmos_room"] - room) * self.atmos_amt
        wet_l, wet_r = self.reverb.process(rev_in, self.shim_amt, room)

        # Gentle limiter on the wet signal so shimmer feedback can't run away.
        mix = CONFIG["reverb_mix"] * spec.get("reverb", 1.0) * (1.0 + CONFIG["atmos_boost"] * self.atmos_amt)
        peak = max(np.abs(wet_l).max(), np.abs(wet_r).max()) * mix
        target = min(1.0, 0.45 / peak) if peak > 0 else 1.0
        new_gain = target if target < self.wet_gain else self.wet_gain + (target - self.wet_gain) * 0.02
        gain = np.linspace(self.wet_gain, new_gain, frames)
        self.wet_gain = new_gain
        wet_l, wet_r = wet_l * gain, wet_r * gain
        self.shim_prev = self.shimmer.process(0.5 * (wet_l + wet_r))

        out_l = dry_l + drone + echo_l + mix * wet_l
        out_r = dry_r + drone + echo_r + mix * wet_r

        # Master limiter: drops gain at once if the mix would clip, recovers over ~0.3 s.
        # Clean gain riding instead of saturation, so stacked effects stay undistorted.
        peak = max(np.abs(out_l).max(), np.abs(out_r).max())
        target = min(1.0, 0.9 / peak) if peak > 0 else 1.0
        if target < self.out_gain:
            new_gain = target
            gain = np.full(frames, target)
        else:
            new_gain = min(target, self.out_gain + frames / (0.3 * sr))
            gain = np.linspace(self.out_gain, new_gain, frames)
        self.out_gain = new_gain
        outdata[:, 0] = np.clip(out_l * gain, -1.0, 1.0).astype(np.float32)
        outdata[:, 1] = np.clip(out_r * gain, -1.0, 1.0).astype(np.float32)

        self.log_f = float(log_f[-1])
        self.bend = float(bend_arr[-1])
        self.pressure = float(pr[-1])
        self.env = float(env[-1])
        self.vib_depth = float(depth_target)
        self.vib_phase = float(vib_ph[-1] % (2.0 * np.pi))


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
class GamepadInput:
    """SDL GameController API: standard layout on every OS, no index guessing."""

    def __init__(self, index):
        self.c = sdl_controller.Controller(index)
        self.name = sdl_controller.name_forindex(index)
        self.buttons = {"cross": pygame.CONTROLLER_BUTTON_A, "circle": pygame.CONTROLLER_BUTTON_B,
                        "square": pygame.CONTROLLER_BUTTON_X, "triangle": pygame.CONTROLLER_BUTTON_Y}

    def poll(self):
        c = self.c
        ax = lambda a: c.get_axis(a) / 32767.0
        return {
            "held": {k for k, b in self.buttons.items() if c.get_button(b)},
            "quit": c.get_button(pygame.CONTROLLER_BUTTON_GUIDE) or c.get_button(pygame.CONTROLLER_BUTTON_START),
            "l1": c.get_button(pygame.CONTROLLER_BUTTON_LEFTSHOULDER),
            "r1": c.get_button(pygame.CONTROLLER_BUTTON_RIGHTSHOULDER),
            "r3": c.get_button(pygame.CONTROLLER_BUTTON_RIGHTSTICK),
            "l3": c.get_button(pygame.CONTROLLER_BUTTON_LEFTSTICK) or self._touchpad(),
            "dpad_down": c.get_button(pygame.CONTROLLER_BUTTON_DPAD_DOWN),
            "dpad_up": c.get_button(pygame.CONTROLLER_BUTTON_DPAD_UP),
            "dpad_left": c.get_button(pygame.CONTROLLER_BUTTON_DPAD_LEFT),
            "dpad_right": c.get_button(pygame.CONTROLLER_BUTTON_DPAD_RIGHT),
            "l2": ax(pygame.CONTROLLER_AXIS_TRIGGERLEFT),
            "r2": ax(pygame.CONTROLLER_AXIS_TRIGGERRIGHT),
            "lx": ax(pygame.CONTROLLER_AXIS_LEFTX),
            "ly": ax(pygame.CONTROLLER_AXIS_LEFTY),
            "rx": ax(pygame.CONTROLLER_AXIS_RIGHTX),
            "ry": ax(pygame.CONTROLLER_AXIS_RIGHTY),
        }

    def _touchpad(self):
        try:
            return self.c.get_button(20)  # SDL_CONTROLLER_BUTTON_TOUCHPAD
        except Exception:
            return False

    def rumble(self, low, high, ms):
        try:
            self.c.rumble(low, high, ms)
        except Exception:
            pass


class RawInput:
    """Plain joystick API using CONFIG indices."""

    def __init__(self, index):
        self.js = pygame.joystick.Joystick(index)
        self.js.init()
        self.name = self.js.get_name()
        self.signed = {"l2": False, "r2": False}

    def _axis(self, idx):
        return self.js.get_axis(idx) if idx < self.js.get_numaxes() else 0.0

    def _trigger(self, key):
        raw = self._axis(CONFIG["axis_" + key])
        if raw < -0.1:
            self.signed[key] = True      # -1..1 range
        return (raw + 1.0) / 2.0 if self.signed[key] else max(raw, 0.0)

    def poll(self):
        js = self.js
        nb = js.get_numbuttons()
        btn = lambda i: i < nb and js.get_button(i)
        hx, hy = js.get_hat(0) if js.get_numhats() else (0, 0)
        return {
            "held": {k for k, i in CONFIG["buttons"].items() if btn(i)},
            "quit": any(btn(i) for i in CONFIG["quit_buttons"]),
            "l1": btn(CONFIG["button_l1"]),
            "r1": btn(CONFIG["button_r1"]),
            "r3": btn(CONFIG["button_r3"]),
            "l3": btn(CONFIG["button_l3"]),
            "dpad_down": hy < 0,
            "dpad_up": hy > 0,
            "dpad_left": hx < 0,
            "dpad_right": hx > 0,
            "l2": self._trigger("l2"),
            "r2": self._trigger("r2"),
            "lx": self._axis(CONFIG["axis_lx"]),
            "ly": self._axis(CONFIG["axis_ly"]),
            "rx": self._axis(CONFIG["axis_rx"]),
            "ry": self._axis(CONFIG["axis_ry"]),
        }

    def rumble(self, low, high, ms):
        try:
            self.js.rumble(low, high, ms)
        except Exception:
            pass


def init_pygame():
    pygame.init()
    try:
        pygame.display.init()
    except pygame.error:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.display.init()
    pygame.joystick.init()
    if sdl_controller:
        sdl_controller.init()


def open_pad(raw):
    if pygame.joystick.get_count() == 0:
        print("No controller found, waiting... (Ctrl+C to quit)")
    while pygame.joystick.get_count() == 0:
        pygame.event.pump()
        time.sleep(0.2)
    if not raw and sdl_controller and sdl_controller.is_controller(0):
        pad, mode = GamepadInput(0), "standard gamepad mapping"
    else:
        pad, mode = RawInput(0), "raw indices from CONFIG"
    print(f"Controller detected: {pad.name} ({mode})")
    return pad


def trigger_amount(v, analog):
    """0..1 shift amount: snaps at the threshold, or eases across the middle of travel."""
    if not analog:
        return 1.0 if v > CONFIG["trigger_threshold"] else 0.0
    lo, hi = CONFIG["trigger_lo"], CONFIG["trigger_hi"]
    t = min(max((v - lo) / (hi - lo), 0.0), 1.0)
    return t * t * (3 - 2 * t)




def snap_bend(raw, note):
    """Magnetic bend: the stick sticks to whole notes, so landing in tune is easy.
    Returns (bend, target) where target is the snapped offset, or None mid-slide."""
    mode = CONFIG["bend_snap"]
    if mode == "off":
        return raw, None
    rng = int(round(CONFIG["bend_range"]))
    base = int(round(note))
    if mode == "scale":
        pcs = SCALE_PCS
        targets = [o for o in range(-rng, rng + 1) if o == 0 or (base + o) % 12 in pcs]
    else:
        targets = list(range(-rng, rng + 1))
    raw = min(max(raw, targets[0]), targets[-1])
    for a, b in zip(targets, targets[1:]):
        if a <= raw <= b:
            u = (raw - a) / (b - a)
            cap = CONFIG["bend_capture"]
            if u <= cap:
                return float(a), a
            if u >= 1 - cap:
                return float(b), b
            v = (u - cap) / (1 - 2 * cap)
            return a + v * v * (3 - 2 * v) * (b - a), None
    return float(targets[0]), targets[0]


def harmony_offset(note):
    """Semitones up to an in-scale harmony note: a third where the scale has one,
    otherwise a fourth, so it stays sweet in every scale."""
    pc = int(round(note)) % 12
    for up in (4, 3, 5, 2, 6):
        if (pc + up) % 12 in SCALE_PCS:
            return up
    return 4


def status_line(note, bend, pressure, l2, r2, fx=""):
    if note is None:
        text = "  --  "
    else:
        sounding = note + bend
        nearest = int(round(sounding))
        cents = int(round((sounding - nearest) * 100))
        text = f"{note_name(nearest):>4}"
        text += f" {cents:+3d}c" if cents else "     "
        if abs(bend) >= 0.005:
            text += f"  (bend {bend * 100:+.0f}c)"
    bars = int(round(pressure / 2.0 * 12))
    mods = ("L2 " if l2 else "   ") + ("R2" if r2 else "  ")
    return f"\r  {text:<28} breath [{'#' * bars}{'-' * (12 - bars)}]  {mods}  {fx}\033[K"


def main():
    ap = argparse.ArgumentParser(description="Controller ocarina")
    ap.add_argument("--debug", action="store_true", help="print raw button/axis indices")
    ap.add_argument("--raw", action="store_true", help="use CONFIG indices instead of SDL mapping")
    args = ap.parse_args()

    init_pygame()
    pad = open_pad(args.raw)

    state = SynthState()
    synth = Synth(state)
    held = []  # button names in press order
    last_axes = {}
    last_status = None
    last_status_t = 0.0
    prev_pad = {}         # previous poll, for toggle edges
    last_rumble_t = 0.0
    harmony_on = False
    locked = None         # bend target the stick is currently snapped to

    stream = sd.OutputStream(samplerate=CONFIG["sample_rate"], blocksize=CONFIG["blocksize"],
                             channels=2, dtype="float32", latency="low",
                             callback=synth.callback)
    names = lambda notes: " ".join(note_name(60 + notes[k])[:-1] for k in _ORDER)
    scale_info = lambda: f"{CONFIG['scale']}: {names(NOTES)} | L2: {names(NOTES_L2)}"
    print(f"Cross/Square/Triangle/Circle (clockwise from bottom). Scale {scale_info()}")
    print("Hold R2 = octave up, hold L1 = sharp.")
    print("Left stick up/down = bend, left/right = vibrato. Right stick up/down = breath.")
    print("Hold R1 = freeze + shimmer reverb. Touchpad click or L3 = atmosphere reverb.")
    print("D-pad Up = ensemble, Down = drone, Left = ping-pong echo, Right = harmony.")
    print("Hold R3 (click right stick) + D-pad Left/Right = change scale, Up/Down = change voice.")
    print("Choir: right stick left/right = vowel (oo-oh-ah-eh-ee).")
    print(f"Voice: {state.voice}")
    print("PS/Options or Ctrl+C to quit.\n")

    def debug(msg):
        if args.debug:
            sys.stdout.write(f"\r[debug] {msg}\033[K\n")

    clock = pygame.time.Clock()
    try:
        with stream:
            while True:
                for ev in pygame.event.get():
                    if ev.type == pygame.JOYBUTTONDOWN:
                        debug(f"button {ev.button} down")
                    elif ev.type == pygame.JOYBUTTONUP:
                        debug(f"button {ev.button} up")
                    elif ev.type == pygame.JOYAXISMOTION:
                        prev = last_axes.get(ev.axis)
                        if prev is None or abs(ev.value - prev) > 0.05:
                            last_axes[ev.axis] = ev.value
                            debug(f"axis {ev.axis} = {ev.value:+.3f}")
                    elif ev.type == pygame.JOYHATMOTION:
                        debug(f"hat {ev.hat} = {ev.value}")
                    elif ev.type == pygame.JOYDEVICEREMOVED:
                        print("\nController disconnected.")
                        held.clear()
                        with state.lock:
                            state.note = None
                        pad = open_pad(args.raw)

                p = pad.poll()
                if p["quit"]:
                    break

                new = [k for k in ("circle", "cross", "square", "triangle")
                       if k in p["held"] and k not in held]
                held = [k for k in held if k in p["held"]] + new

                pressed = lambda k: p[k] and not prev_pad.get(k)
                if p["r3"]:
                    # R3 held: D-pad Up/Down = voice, Left/Right = scale.
                    if pressed("dpad_up") or pressed("dpad_down"):
                        names_ = list(VOICES)
                        step = 1 if p["dpad_up"] else -1
                        state.voice = names_[(names_.index(state.voice) + step) % len(names_)]
                        sys.stdout.write(f"\r  Voice: {state.voice}\033[K\n")
                        last_status = None
                    if pressed("dpad_left") or pressed("dpad_right"):
                        names_ = list(SCALES)
                        step = 1 if p["dpad_right"] else -1
                        set_scale(names_[(names_.index(CONFIG["scale"]) + step) % len(names_)])
                        sys.stdout.write(f"\r  Scale {scale_info()}\033[K\n")
                        last_status = None
                else:
                    if pressed("dpad_up"):
                        state.ensemble = not state.ensemble
                if pressed("l3"):
                    state.atmos = not state.atmos
                    if pressed("dpad_down"):
                        state.drone = not state.drone
                    if pressed("dpad_left"):
                        state.delay = not state.delay
                    if pressed("dpad_right"):
                        harmony_on = not harmony_on
                prev_pad = p

                l2_amt = trigger_amount(p["l2"], CONFIG["analog_l2"])
                r2_amt = trigger_amount(p["r2"], CONFIG["analog_r2"])
                l2_on, r2_on = l2_amt > 0, r2_amt > 0
                note = None
                if held:
                    b = held[-1]
                    note = (60 + NOTES[b] + l2_amt * (NOTES_L2[b] - NOTES[b])
                            + 12 * r2_amt + (1 if p["l1"] else 0))

                raw_bend = -shape(p["ly"]) * CONFIG["bend_range"]
                if note is not None:
                    bend, target = snap_bend(raw_bend, note)
                else:
                    bend, target = 0.0, None
                vibrato = abs(shape(p["lx"]))
                pressure = max(0.15, min(2.0, 1.0 - shape(p["ry"])))  # stick up = louder

                with state.lock:
                    state.note = note
                    state.bend = bend
                    state.vibrato = vibrato
                    state.pressure = pressure
                    state.vowel = (shape(p["rx"]) + 1.0) * 2.0
                    state.press_id += len(new)
                    state.shimmer = bool(p["r1"])
                    state.harmony = harmony_offset(note + bend) if harmony_on and note is not None else None

                # Rumble: a soft tap on each attack, a purr when blowing hard.
                now = time.monotonic()
                if CONFIG["rumble"]:
                    # Bend haptics: a crisp click when the stick locks onto a note,
                    # a faint buzz while you're sliding between notes.
                    if target is not None and target != locked and target != 0:
                        pad.rumble(0.0, 0.7, 35)
                        last_rumble_t = now
                    elif target is None and note is not None and now - last_rumble_t > 0.08:
                        pad.rumble(0.0, 0.12, 100)
                        last_rumble_t = now
                    locked = target
                    if new:
                        pad.rumble(0.2, 0.35, 50)
                        last_rumble_t = now
                    elif note is not None and pressure > 1.3 and now - last_rumble_t > 0.1:
                        pad.rumble(min((pressure - 1.3) / 0.7, 1.0) * 0.3, 0.0, 150)
                        last_rumble_t = now

                fx = state.voice.upper()
                if state.voice == "choir":
                    fx += f"({VOWEL_NAMES[int(round(state.vowel))]})"
                fx += " " + " ".join(
                    name for name, on in (("FREEZE", p["r1"]), ("ENSEMBLE", state.ensemble),
                                          ("DRONE", state.drone),
                                          ("ECHO", state.delay), ("HARMONY", harmony_on),
                                          ("ATMOS", state.atmos)) if on)
                if synth.xruns:
                    fx += f"  dropouts:{synth.xruns}"
                line = status_line(note, bend, pressure, l2_on, r2_on, fx)
                if line != last_status and now - last_status_t > 1 / 30:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    last_status, last_status_t = line, now

                clock.tick(250)
    except KeyboardInterrupt:
        pass
    finally:
        pygame.quit()
        print("\nBye.")


if __name__ == "__main__":
    main()
