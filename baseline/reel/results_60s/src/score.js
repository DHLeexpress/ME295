  // SCORE BODY — a light lo-fi bed (ducked under the voice by score-head) + sound effects tied to the picture.
  const BPMs = 100, BT = 60 / BPMs, END = DURATION;
  const prog = [['D3', 'Fs3', 'A3'], ['B2', 'D3', 'Fs3'], ['G2', 'B2', 'D3'], ['A2', 'Cs3', 'E3']];
  const oct = (n, d) => n.replace(/-?\d/, (x) => String(+x + d));
  to = 'm';
  for (let bar = 0; bar * 4 * BT < END; bar++) {
    const t = bar * 4 * BT, ch = prog[bar % 4];
    pad(t, Math.min(END + 0.2, t + 4 * BT + 0.05), ch, 0.022, { type: 'triangle', cut: 1100, att: 0.08, rel: 0.12, send: 0.3 });
    bass(t, nz(oct(ch[0], -1)), 0.22); bass(t + 2 * BT, nz(oct(ch[0], -1)), 0.16);
    for (let k = 0; k < 8; k++) { const tk = t + k * BT / 2; if (tk > END) break; pluck(tk, nz(oct(ch[k % 3], 2)), k % 2 ? 0.045 : 0.07, 0, 0.22, 2600, 0.15); }
    for (let b = 0; b < 4; b++) { const tb = t + b * BT; if (tb > END) break; if (b % 2 === 0) sub(tb, 0.16); noiseHit(tb + BT / 2, 0.035, 'highpass', 7000, 0.7, 0.05); }
  }
  to = 's';
  for (const [t0] of SHOT_LIST.slice(1)) sweep(t0 - 0.16, t0, 0.02, 900, 4200, 0);      // whoosh into every cut
  for (const e of SFX_CUES()) {                                                         // cues from src/scenes.js
    if (e.k === 'type') for (let i = 0; i < e.n; i++) noiseHit(e.t + e.d * i / e.n, 0.018, 'bandpass', 3800 + (i % 3) * 600, 2, 0.06);
    if (e.k === 'tick') blip(e.t, 1300, 1700, 0.05, 0.07);
    if (e.k === 'pop') blip(e.t, 700, 1400, 0.1, 0.09);
    if (e.k === 'thud') { sub(e.t, 0.35); noiseHit(e.t, 0.09, 'lowpass', 1600, 0.8, 0.22); }
    if (e.k === 'chime') chime(e.t, [nz('D5'), nz('Fs5'), nz('A5')], 0.09);
  }
