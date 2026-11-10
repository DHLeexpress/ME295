// hard cuts only (one scene per line, extra cuts on "expert" and "That's"); no cameras (src/bridges.js)

// =====================================================================
//  SCENES — the results of baseline/RESULTS.md in 60 s. The headline table is built on screen cell by cell as the
//  voice reads it; a second table shows where the fine-tuned policy's 39 misses happen. Every number is from the
//  evaluation (results/eval); the monitors play high-resolution re-runs of evaluation test cases (src/clips.js).
//  Layout: sign 140-296, terminal 322-400, monitor x 60-700 / y 450-930 or table x 50-950 / y 430-1290,
//  counters in the right column (x ~838), stamps lower left, caption chips 1450-1600, the mascot bottom right.
// =====================================================================
CLAWD_SCALE = 1.0;
const CLAWD_X = 900, CLAWD_Y = 1660;
const KICK = 'ME295 · VLA BASELINE';
const MON = { x: 60, y: 450, w: 640, h: 480 };
const SCENES = {};

// ---- piece helpers (prefixed rl* to stay clear of the kit's names)
function rlClip(key, x, y, w, h, t0, dur) {   // one sprite-sheet frame of a sim clip, by time since t0
  const c = RL_CLIPS[key], img = asset(key + '.jpg'); if (!img || !c) return;
  const k = Math.max(0, Math.min(c.n - 1, Math.floor((TT - t0) / dur * c.n)));
  ctx.drawImage(img, (k % c.cols) * c.fw, Math.floor(k / c.cols) * c.fh, c.fw, c.fh, x, y, w, h);
}
function rlMonitor(key, tag, draw, o = {}) {  // a paper monitor on a stand; draw(x, y, w, h) fills the screen
  const { x, y, w, h } = o.box ?? MON;
  pp(pR(x + w / 2 - 36, y + h + 20, 72, 64, 4), PR.greyD, { key: 'mst' + key, sb: 8, sy: 4 });
  pp(pR(x + w / 2 - 140, y + h + 78, 280, 24, 8), PR.char, { key: 'msb' + key, sb: 10, sy: 5 });
  pp(pR(x - 20, y - 20, w + 40, h + 40, 18), PR.char, { key: 'mon' + key, sb: 22, sy: 12, tex: 0.35, rim: 'rgba(255,255,255,0.16)' });
  ctx.save(); ctx.fillStyle = '#0e0f12'; ctx.fillRect(x, y, w, h); ctx.beginPath(); ctx.rect(x, y, w, h); ctx.clip();
  draw(x, y, w, h); ctx.restore();
  const tw = txtW(tag, 22, { weight: 700, font: F_MONO }) + 26;
  ctx.save(); ctx.fillStyle = 'rgba(16,18,22,0.84)'; ctx.beginPath(); ctx.roundRect(x + 12, y + 12, tw, 38, 6); ctx.fill(); ctx.restore();
  txt(tag, x + 25, y + 39, 22, '#f4f1e8', { weight: 700, font: F_MONO });
  if (Math.floor(TT * 2) % 2 === 0) { ctx.save(); ctx.fillStyle = '#e0453a'; ctx.beginPath(); ctx.arc(x + w - 30, y + 31, 10, 0, TAU); ctx.fill(); ctx.restore(); }
}
function rlArrow(x0, y0, x1, y1, frac, color = PR.yellow, wdt = 10) {   // a paper arrow that draws on
  const u = clamp(frac); if (u <= 0) return;
  const xe = lerp(x0, x1, u), ye = lerp(y0, y1, u), a = Math.atan2(ye - y0, xe - x0);
  ctx.save(); ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = wdt; ctx.lineCap = 'round';
  ctx.shadowColor = 'rgba(0,0,0,0.45)'; ctx.shadowBlur = 8; ctx.shadowOffsetY = 3;
  ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(xe, ye); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(xe + Math.cos(a) * 22, ye + Math.sin(a) * 22);
  ctx.lineTo(xe + Math.cos(a + 2.4) * 26, ye + Math.sin(a + 2.4) * 26); ctx.lineTo(xe + Math.cos(a - 2.4) * 26, ye + Math.sin(a - 2.4) * 26);
  ctx.closePath(); ctx.fill(); ctx.restore();
}
function rlRing(x, y, r, t) {   // a highlight ring that draws on at t
  if (TT < t) return;
  ctx.save(); ctx.strokeStyle = PR.yellow; ctx.lineWidth = 9; ctx.shadowColor = 'rgba(0,0,0,0.5)'; ctx.shadowBlur = 8;
  ctx.beginPath(); ctx.ellipse(x, y, r * popS(t), r * 1.25 * popS(t), -0.12, 0, TAU * clamp((TT - t) / 0.35)); ctx.stroke(); ctx.restore();
}
function rlBooks(x, y, key, n = 4) {
  const cols = [PR.red, PR.blue, PR.mustard, PR.green, PR.plum];
  for (let i = 0; i < n; i++) pp(pR(x + (i % 2) * 14 - 6, y - (i + 1) * 40, 230 - i * 16, 36, 4), cols[i % cols.length], { key: key + i, sb: 8, sy: 4 });
}
function rlLamp(x, y, key) {
  pp(pR(x - 10, y, 20, 300, 6), PR.char, { key: key + 's', sb: 8, sy: 4 });
  pp(pG([[x - 80, y + 10], [x + 80, y + 10], [x + 50, y - 80], [x - 50, y - 80]]), PR.mustard, { key: key + 'h', sb: 10, sy: 6 });
  pp(pR(x - 60, y + 290, 120, 22, 6), PR.char, { key: key + 'b', sb: 8, sy: 4 });
}
function rlCone(x, y, key) {
  pp(pG([[x - 50, y], [x + 50, y], [x + 12, y - 130], [x - 12, y - 130]]), PR.orange, { key: key, sb: 10, sy: 5 });
  pp(pR(x - 36, y - 78, 72, 18, 2), PR.creamL, { key: key + 'b', shadow: false, tex: 0.4 });
  pp(pR(x - 64, y - 6, 128, 18, 4), PR.orange, { key: key + 'f', sb: 6, sy: 3 });
}
function rlKettle(x, y, label, key) {
  ctx.save(); ctx.strokeStyle = PR.char; ctx.lineWidth = 26; ctx.beginPath(); ctx.arc(x, y - 120, 62, Math.PI * 1.05, Math.PI * 1.95); ctx.stroke(); ctx.restore();
  pp(pE(x, y - 40, 112, 100), PR.char, { key: key, sb: 14, sy: 7, tex: 0.4 });
  txt(label, x, y - 24, 44, PR.creamL, { align: 'center', weight: 800 });
}
function rlInvestigation(key) {
  prWall(PR.plum, { panel: 'rgba(0,0,0,0.10)', gap: 120, streak: 420 });
  prFloor(1560, PR.woodD, { trim: false });
  prNote(905, 1010, 190, 150, '#f3e3a5', 0.05, key + 'n1');
  ctx.save(); ctx.strokeStyle = PR.red; ctx.lineWidth = 4; ctx.beginPath(); ctx.moveTo(720, 700); ctx.quadraticCurveTo(850, 800, 900, 940); ctx.stroke(); ctx.restore();
  pp(pE(900, 940, 10, 10), PR.red, { key: key + 'pin', sb: 4, sy: 2 });
  ctx.save(); ctx.strokeStyle = PR.char; ctx.lineWidth = 16; ctx.beginPath(); ctx.arc(860, 590, 66, 0, TAU); ctx.stroke(); ctx.lineWidth = 22; ctx.lineCap = 'round'; ctx.beginPath(); ctx.moveTo(908, 638); ctx.lineTo(970, 715); ctx.stroke(); ctx.restore();
  ctx.save(); ctx.fillStyle = 'rgba(200,230,255,0.25)'; ctx.beginPath(); ctx.arc(860, 590, 58, 0, TAU); ctx.fill(); ctx.restore();
}
function rlCaveats(x, y, w, items, t0, dt) {   // a dark panel of caveats; each gets an amber '!' badge when said
  pp(pR(x, y, w, 120 + items.length * 110, 14), PR.char, { key: 'cav', sb: 18, sy: 9, tex: 0.35 });
  txt('CAVEATS', x + 36, y + 70, 32, PR.mustard, { weight: 800 });
  items.forEach((it, i) => {
    const yy = y + 104 + i * 110, t = t0 + i * dt;
    pp(pR(x + 30, yy, w - 60, 90, 10), '#33363d', { key: 'cavi' + i, shadow: false, tex: 0.3 });
    txt(it, x + 130, yy + 58, fitSize(it, w - 200, 36), '#eef2ea', { weight: 700 });
    if (TT >= t) withT(x + 80, yy + 45, 0, popS(t), () => { pp(pE(0, 0, 30, 30), PR.mustard, { key: 'cavb' + i, sb: 6, sy: 3 }); txt('!', 0, 15, 42, PR.ink, { align: 'center', weight: 800 }); });
  });
}
function rlSkin(x, y, wk, ws) {   // the PDMS skin: 5 x 7 taxels that light up from wk; a shear arrow from ws
  withT(x, y, 0.04, popS(E0 + 0.15), () => {
    pp(pR(-170, -250, 340, 480, 26), '#e9b6a8', { key: 'skin', sb: 18, sy: 9, tex: 0.5, rim: 'rgba(255,255,255,0.4)' });
    for (let r = 0; r < 7; r++) for (let c = 0; c < 5; c++) {
      const cx = -120 + c * 60, cy = -195 + r * 62, d = Math.hypot(c - 2 - 0.8 * Math.sin(TT * 2.2), r - 3), lit = TT > wk ? clamp(1.25 - d / 2.2) * clamp((TT - wk) / 0.3) : 0;
      ctx.save(); ctx.fillStyle = lit > 0.05 ? `rgba(214,69,58,${0.25 + 0.75 * lit})` : 'rgba(120,70,60,0.35)'; ctx.beginPath(); ctx.arc(cx, cy, 17 + 7 * lit, 0, TAU); ctx.fill(); ctx.restore();
    }
  });
  rlArrow(x - 140, y + 320, x + 160, y + 320, ev(ws, 0.45), PR.navy, 12);
}

// ---- THE RESULTS TABLE, drawn in paper: columns = policies, rows = test suites (the headline table, transposed to fit
//  9:16). Each cell: the success count, the percentage, and a bar in the policy's colour (the report's figure slots).
const TCOLS = [
  { k: 'zs', name: 'ZERO-SHOT', sub: 'π0.5-DROID', col: '#eb6834' },
  { k: 'ft', name: 'FINE-TUNED', sub: 'π0.5 + LoRA', col: '#2a78d6' },
  { k: 'ex', name: 'EXPERT', sub: 'scripted', col: '#1baf7a' },
];
const TROWS = [
  { name: 'Nominal', sub: 'training scene', img: 'thumb_nominal.jpg' },
  { name: 'Shuffled', sub: '8-12 mm gaps', img: 'thumb_random.jpg' },
  { name: 'Narrow', sub: '6.5-7.5 mm gaps', img: 'thumb_narrow.jpg' },
  { name: 'Heavy', sub: '2 kg, μ 0.5', img: 'thumb_heavy.jpg' },
];
const TVAL = { zs: ['0/20', '0/20', null, null], ft: ['50/50', '36/50', '22/30', '13/30'], ex: ['50/50', '50/50', '30/30', '17/30'] };
const TPCT = { zs: [0, 0, null, null], ft: [100, 72, 73, 43], ex: [100, 100, 100, 57] };
const TB = { x: 50, y: 440, w: 900, hh: 108, rh: 188 };
// o.rows[i]: time row i appears; o.cells['ft1']: time a cell appears (-1 = already there); o.hi: [row, t] highlight;
// o.skin: time an extra '+ TOUCH' column with '?' cells appears
function rlTable(o) {
  const skin = o.skin != null && TT >= o.skin, cols = skin ? TCOLS.concat([{ k: 'sk', name: '+ TOUCH', sub: 'next test', col: PR.clawd }]) : TCOLS;
  const lw = skin ? 250 : 280, cw = (TB.w - lw) / cols.length, H = TB.hh + TB.rh * 4;
  pp(pR(TB.x - 18, TB.y - 18, TB.w + 36, H + 36, 16), PR.paperW, { key: 'tbl', sb: 22, sy: 12, tex: 0.35 });
  // header
  cols.forEach((c, j) => {
    const cx = TB.x + lw + cw * j + cw / 2, s = c.k === 'sk' ? popS(o.skin) : 1;
    withT(cx, TB.y + 52, 0, s, () => {
      pp(pR(-cw / 2 + 12, -40, cw - 24, 10, 4), c.col, { key: 'th' + c.k, shadow: false, tex: 0.3 });
      txt(c.name, 0, 0, fitSize(c.name, cw - 20, 26), PR.ink, { align: 'center', weight: 800 });
      txt(c.sub, 0, 32, 19, PR.greyD, { align: 'center', weight: 600 });
    });
  });
  // rows
  TROWS.forEach((r, i) => {
    const ta = o.rows ? o.rows[i] : -1; if (ta == null || TT < ta) return;
    const y = TB.y + TB.hh + TB.rh * i, s = ta < 0 ? 1 : popS(ta);
    if (o.hi && o.hi[0] === i && TT >= o.hi[1]) { ctx.save(); ctx.fillStyle = 'rgba(242,201,76,0.38)'; ctx.fillRect(TB.x - 6, y + 4, TB.w + 12, TB.rh - 8); ctx.restore(); }
    ctx.save(); ctx.fillStyle = 'rgba(24,27,36,0.12)'; ctx.fillRect(TB.x, y, TB.w, 3); ctx.restore();
    withT(TB.x, y + TB.rh / 2, 0, s, () => {
      drawAsset(r.img, 0, -64, 118, 88, { fit: 'cover', r: 8 });
      txt(r.name, 132, -6, fitSize(r.name, lw - 140, 36), PR.ink, { weight: 800 });
      txt(r.sub, 132, 26, fitSize(r.sub, lw - 140, 20), PR.greyD, { weight: 600 });
    });
    cols.forEach((c, j) => {
      const cx = TB.x + lw + cw * j + cw / 2, cy = y + TB.rh / 2;
      if (c.k === 'sk') { if (TT >= o.skin + 0.15 * (i + 1)) withT(cx, cy, 0, popS(o.skin + 0.15 * (i + 1)), () => txt('?', 0, 22, 70, PR.clawd, { align: 'center', weight: 800 })); return; }
      const tc = o.cells ? o.cells[c.k + i] : -1; if (tc == null || TT < tc) return;
      const v = TVAL[c.k][i], pct = TPCT[c.k][i];
      if (v == null) { txt('–', cx, cy + 14, 44, PR.grey, { align: 'center', weight: 800 }); return; }
      const u = tc < 0 ? 1 : EZ.o3(ev(tc, 0.5));
      pp(pR(cx - cw / 2 + 14, cy + 44, cw - 28, 16, 6), 'rgba(24,27,36,0.10)', { shadow: false, tex: false });
      if (pct > 0) pp(pR(cx - cw / 2 + 14, cy + 44, (cw - 28) * pct / 100 * u, 16, 6), c.col, { key: 'bar' + c.k + i, shadow: false, tex: 0.3 });
      withT(cx, cy, 0, tc < 0 ? 1 : popS(tc), () => {
        txt(v, 0, -2, fitSize(v, cw - 22, 50), PR.ink, { align: 'center', weight: 800 });
        txt(Math.round(pct * u) + ' %', 0, 32, 22, PR.greyD, { align: 'center', weight: 700 });
      });
    });
  });
}
// ---- where the fine-tuned policy's 39 misses happen (randomized + narrow + heavy suites; RESULTS.md section 3.2)
const FROWS = [
  ['pushed in', 12, true], ['slipped in lift', 12, true], ['jammed', 5, true], ['protective stop', 2, true],
  ['insertion miss', 1, true], ['partial pull', 1, true], ['never reached', 4, false], ['wrong book', 2, false],
];
function rlFailTable(t0, tc) {
  const x = 60, y = 440, w = 880, rh = 92, lw = 330, bw = 400;
  pp(pR(x - 18, y - 18, w + 36, 96 + rh * FROWS.length + 30, 16), PR.paperW, { key: 'ftb', sb: 22, sy: 12, tex: 0.35 });
  txt('FAILURE MODE', x + 10, y + 52, 26, PR.ink, { weight: 800 });
  txt('EPISODES', x + lw + 10, y + 52, 26, PR.ink, { weight: 800 });
  txt('\u03c00.5 fine-tuned \u00b7 3 suites', x + w - 10, y + 52, 20, PR.greyD, { weight: 700, align: 'right' });
  FROWS.forEach(([name, n, contact], i) => {
    const yy = y + 96 + rh * i, ta = t0 + 0.12 * i;
    if (TT < ta) return;
    if (contact && TT >= tc) { ctx.save(); ctx.fillStyle = 'rgba(42,120,214,0.13)'; ctx.fillRect(x - 6, yy + 4, w + 12, rh - 8); ctx.restore(); }
    ctx.save(); ctx.fillStyle = 'rgba(24,27,36,0.12)'; ctx.fillRect(x, yy, w, 3); ctx.restore();
    txt(name, x + 10, yy + rh / 2 + 12, 34, PR.ink, { weight: 700 });
    const u = EZ.o3(ev(ta, 0.45)), bl = bw * n / 12 * u;
    pp(pR(x + lw, yy + rh / 2 - 20, Math.max(4, bl), 40, 6), contact ? '#2a78d6' : PR.grey, { key: 'fb' + i, shadow: false, tex: 0.3 });
    txt(String(n), x + lw + bl + 16, yy + rh / 2 + 14, 38, PR.ink, { weight: 800 });
  });
  // the contact rows, bracketed on the word "contact"
  if (TT >= tc) {
    const yb0 = y + 96 + 8, yb1 = y + 96 + rh * 6 - 8, xb = x + w - 30, u = EZ.o3(ev(tc, 0.35));
    ctx.save(); ctx.strokeStyle = PR.red; ctx.lineWidth = 7; ctx.lineCap = 'round'; ctx.beginPath();
    ctx.moveTo(xb - 24, yb0); ctx.lineTo(xb, yb0); ctx.lineTo(xb, lerp(yb0, yb1, u)); if (u >= 1) ctx.lineTo(xb - 24, yb1); ctx.stroke(); ctx.restore();
  }
}

// ---- 1. the question
SCENES.hook = () => {
  const t0 = E0, wn = VW('hook', 'named').t0;
  prWall('#2b3b5e', { panel: 'rgba(255,255,255,0.04)', streak: 300 });
  prFloor(1560, PR.woodD, { trim: false });
  prClock(860, 560, 62, 3, 50 + TT * 6, 'clk1');
  rlLamp(880, 900, 'lamp1');
  rlBooks(80, 1560, 'bs1');
  rlMonitor('m1', 'SIM · UR10e · 2x', (x, y, w, h) => rlClip('hook', x, y, w, h, t0, 4.0));
  rlRing(MON.x + MON.w * 0.47, MON.y + MON.h * 0.45, 88, wn);
  clawd(CLAWD_X, CLAWD_Y, { hat: 'boater', key: 'c1', s: 0.9, look: -14, mood: TT > VW('hook', 'bookcase').t0 ? 'wow' : 'neutral' });
  prSign('CAN A VLA PICK THIS BOOK?', { kicker: KICK });
  prTerm('pick up the dark blue book', ev(t0 + 0.2, 1.3));
  prChips();
};

// ---- 2. the setup: the cell, four suites
SCENES.setup = () => {
  const t0 = E0, w4 = VW('setup', 'four').t0;
  prWall(PR.slate, { streak: 620 });
  prFloor(1560, PR.floor, { studs: true });
  prNote(160, 1180, 200, 150, '#f3e3a5', -0.05, 'n2a'); prNote(420, 1200, 170, 140, '#a9cbe0', 0.06, 'n2b');
  prStar(1000, 1050, 22); prStar(960, 1120, 14);
  rlMonitor('m2', 'SIM · THE CELL · 3x', (x, y, w, h) => rlClip('cell', x, y, w, h, t0, 4.9));
  prCounter('TEST SUITES', TT >= w4 ? 4 : '…', 838, 640, { w: 230, h: 180, big: 96, rot: 0.04, s: popS(t0 + 0.2) });
  clawd(CLAWD_X, CLAWD_Y, { hat: 'cap', key: 'c2', s: 0.9, look: -14, mood: TT > w4 ? 'happy' : 'neutral' });
  prSign('THE BASELINE TO BEAT', { kicker: KICK });
  prTerm('pi0.5 + UR10e + blade gripper', ev(t0 + 0.3, 1.2));
  prChips();
};

// ---- 3. the four suites = the rows of the table
SCENES.suites = () => {
  const rows = [VW('suites', 'training').t0, VW('suites', 'shuffled').t0, VW('suites', 'narrower').t0, VW('suites', 'heavy').t0];
  prWall('#2b3b5e', { panel: 'rgba(255,255,255,0.04)', streak: 820 });
  prFloor(1560, PR.woodD, { trim: false });
  rlTable({ rows, cells: {} });
  clawd(CLAWD_X + 30, CLAWD_Y + 80, { hat: 'mortar', key: 'c3', s: 0.72, look: -14 });
  prSign('FOUR TEST SUITES', { kicker: KICK });
  prChips();
};

// ---- 4. zero-shot
SCENES.zero = () => {
  const t0 = E0, wz = VW('zero', '0').t0, wc = VW('zero', 'crashed').t0;
  prWall('#4a4566', { panel: 'rgba(0,0,0,0.08)', streak: 700 });
  prFloor(1560, '#3b3855', { trim: false });
  for (let i = 0; i < 8; i++) pp(pG([[i * 150 - 40, 1600], [i * 150 + 40, 1600], [i * 150 + 120, 1660], [i * 150 + 40, 1660]]), i % 2 ? PR.ink : PR.mustard, { key: 'haz' + i, shadow: false, tex: 0.5 });
  rlCone(120, 1440, 'cone1'); rlCone(560, 1440, 'cone2');
  const on = Math.floor(TT * 3) % 2 === 0;
  pp(pR(820, 1000, 60, 40, 6), PR.char, { key: 'bcn', sb: 6, sy: 3 }); pp(pA(850, 1002, 34, Math.PI, TAU), on ? '#ff5a4a' : '#7a2a26', { key: 'bcl', shadow: false, tex: 0.3 });
  rlMonitor('m4', 'π0.5-DROID · ZERO-SHOT', (x, y, w, h) => rlClip('zero', x, y, w, h, t0, 3.8));
  prCounter('SUCCESS', TT >= wz ? '0/20' : '…', 838, 640, { w: 230, h: 180, big: 76, rot: 0.04, s: popS(t0 + 0.1) });
  prCross(838, 850, wz + 0.1, 54);
  prStamp('PROTECTIVE STOP', 400, 1190, PR.red, wc, { rot: -0.07, size: 50, from: 1.3 });
  prBurst(400, 1190, wc, { n: 18, cols: [PR.red, PR.mustard, '#ffffff'] });
  clawd(CLAWD_X, CLAWD_Y, { hat: 'guard', key: 'c4', s: 0.9, look: -16, mood: TT > wc ? 'wow' : 'flat' });
  prSign('OUT OF THE BOX: 0 / 20', { kicker: KICK });
  prChips();
};

// ---- 5. fine-tuned in its training scene
SCENES.tuned = () => {
  const t0 = E0, w1 = VW('tuned', '50').t0, w2 = VW('tuned', '50', 1).t0, wd = VW('tuned', '251').t0;
  prWall(PR.slate, { streak: 300 });
  prFloor(1560, PR.floor, { studs: true });
  prBoard(60, 1070, 520, 300, 'b5');
  chalk('251 demos', 100, 1170, 50, ev(wd - 0.1, 0.6));
  chalk('LoRA, 10k steps', 100, 1260, 40, ev(wd + 0.4, 0.6));
  pp(pE(690, 1380, 34, 30), PR.red, { key: 'apple5', sb: 8, sy: 4 }); pp(pR(686, 1340, 8, 18, 2), PR.woodD, { shadow: false, tex: false });
  rlMonitor('m5', 'π0.5 FINE-TUNED · 2x', (x, y, w, h) => rlClip('tuned', x, y, w, h, t0, 4.9));
  prCounter('NOMINAL', countUp(w1, Math.max(0.5, w2 - w1), 0, 50), 838, 640, { w: 230, h: 180, big: 90, rot: -0.04, s: popS(t0 + 0.1) });
  prStamp('50 / 50', 838, 860, PR.green, w2, { rot: 0.06, size: 48, from: 1.3 });
  clawd(CLAWD_X, CLAWD_Y, { hat: 'mortar', key: 'c5', s: 0.9, look: -12, mood: TT > w2 ? 'happy' : 'neutral' });
  prSign('FINE-TUNED: 50 / 50', { kicker: KICK });
  prChips();
};

// ---- 6. the fine-tuned column fills in: 72 / 73 / 43 %
SCENES.table = () => {
  const cells = { zs0: -1, zs1: -1, zs2: -1, zs3: -1, ft0: -1, ft1: VW('table', '72').t0, ft2: VW('table', '73').t0, ft3: VW('table', '43').t0 };
  prWall(PR.teal, { panel: 'rgba(255,255,255,0.06)', streak: 380 });
  prFloor(1560, PR.woodD, { trim: false });
  rlTable({ cells, hi: [3, VW('table', 'heavy').t0] });
  clawd(CLAWD_X + 30, CLAWD_Y + 80, { hat: 'cap', key: 'c6', s: 0.72, look: -14, mood: TT > VW('table', '43').t0 ? 'flat' : 'neutral' });
  prSign('OFF ITS TRAINING SCENE', { kicker: KICK });
  prChips();
};

// ---- 7. the expert column: the ceiling
SCENES.expert = () => {
  const wp = VW('expert', 'perfect').t0, w57 = VW('expert', '57').t0;
  const cells = { zs0: -1, zs1: -1, zs2: -1, zs3: -1, ft0: -1, ft1: -1, ft2: -1, ft3: -1, ex0: wp, ex1: wp + 0.15, ex2: wp + 0.3, ex3: w57 };
  prWall('#c79a5b', { panel: 'rgba(0,0,0,0.05)', streak: 540 });
  prFloor(1560, PR.floorD, { trim: false });
  rlTable({ cells, hi: [3, VW('expert', 'heavy').t0] });
  clawd(CLAWD_X + 30, CLAWD_Y + 80, { hat: 'guard', key: 'c7', s: 0.72, look: -14 });
  prSign('THE CEILING: A SCRIPTED EXPERT', { kicker: KICK });
  prChips();
};

// ---- 8. where the 39 misses happen: 33 at contact
SCENES.contact = () => {
  const wc = VW('contact', 'contact').t0, w33 = VW('contact', '33').t0;
  prWall('#583656', { panel: 'rgba(0,0,0,0.10)', gap: 120, streak: 420 });
  prFloor(1560, PR.woodD, { trim: false });
  rlFailTable(E0 + 0.2, w33);
  prStamp('33 OF 39 AT CONTACT', 420, 1350, PR.red, wc, { rot: -0.04, size: 46, from: 1.3 });
  clawd(CLAWD_X + 40, CLAWD_Y + 90, { hat: 'ranger', key: 'c8', s: 0.7, look: -14, mood: TT > wc ? 'wow' : 'neutral' });
  prSign('WHERE THE 39 MISSES HAPPEN', { kicker: KICK });
  prChips();
};

// ---- 9. push-in beside a thick book
SCENES.push = () => {
  const t0 = E0, wp = VW('push', 'push').t0, wn = VW('push', 'never').t0;
  rlInvestigation('i9');
  rlMonitor('m9', 'TOP VIEW · 45 mm BOOK', (x, y, w, h) => rlClip('push', x, y, w, h, t0 + 0.6, 3.2));
  rlArrow(MON.x + MON.w * 0.50, MON.y + MON.h * 0.80, MON.x + MON.w * 0.50, MON.y + MON.h * 0.30, ev(wp, 0.5));
  prStamp('PUSHED 56 mm DEEPER', 400, 1150, PR.red, wp + 0.3, { rot: -0.06, size: 44, from: 1.3 });
  prStamp('PADS: 0 N', 300, 1290, PR.navy, wn, { rot: 0.05, size: 44, from: 1.3 });
  clawd(CLAWD_X, CLAWD_Y, { hat: 'ranger', key: 'c9', s: 0.9, look: -14, mood: 'flat' });
  prSign('PUSH-IN: PADS NEVER TOUCH', { kicker: KICK });
  prChips();
};

// ---- 10. narrow gaps: shoves a neighbour
SCENES.shove = () => {
  const t0 = E0, wn = VW('shove', 'neighbors').t0;
  rlInvestigation('i10');
  rlMonitor('m10', 'TOP VIEW · NARROW GAPS', (x, y, w, h) => rlClip('shove', x, y, w, h, t0, 3.6));
  rlArrow(MON.x + MON.w * 0.30, MON.y + MON.h * 0.62, MON.x + MON.w * 0.30, MON.y + MON.h * 0.22, ev(VW('shove', 'shoves').t0, 0.5));
  prStamp('NEIGHBOR SHOVED 56 mm', 400, 1190, PR.red, wn, { rot: 0.05, size: 42, from: 1.3 });
  clawd(CLAWD_X, CLAWD_Y, { hat: 'ranger', key: 'c10', s: 0.9, look: -14, mood: 'wow' });
  prSign('NARROW GAPS: SHOVING', { kicker: KICK });
  prChips();
};

// ---- 11. heavy books slip, for the policy and (split on "expert") the expert
function rlGym(key) {
  prWall('#c79a5b', { panel: 'rgba(0,0,0,0.05)', streak: 540 });
  prFloor(1560, PR.floorD, { trim: false });
  for (let i = 0; i < 3; i++) pp(pR(80 + i * 26, 1300 - i * 6, 18, 230, 4), PR.char, { key: key + 'rack' + i, sb: 6, sy: 3 });
  pp(pR(60, 1300, 140, 18, 4), PR.char, { key: key + 'rackb', sb: 6, sy: 3 });
  rlKettle(470, 1420, '2.35 kg', key + 'kb');
}
SCENES.slip = () => {
  const t0 = E0;
  rlGym('g11');
  rlMonitor('m11', 'π0.5 FINE-TUNED · 4x', (x, y, w, h) => rlClip('slip', x, y, w, h, t0, 2.9));
  prCounter('π0.5 TUNED', '13/30', 838, 640, { w: 230, h: 170, big: 64, rot: -0.04, s: popS(t0 + 0.1) });
  prStamp('SLIPS', 250, 1150, PR.red, VW('slip', 'slip').t0, { rot: -0.08, size: 56, from: 1.3 });
  clawd(CLAWD_X, CLAWD_Y, { hat: 'phones', key: 'c11', s: 0.9, look: -12, mood: 'flat' });
  prSign('HEAVY BOOKS SLIP', { kicker: KICK });
  prChips();
};
SCENES.slip2 = () => {
  const t0 = E0;
  rlGym('g12');
  rlMonitor('m12', 'EXPERT · SAME BOOK · 4x', (x, y, w, h) => rlClip('slipx', x, y, w, h, t0, 1.9));
  prCounter('EXPERT', '17/30', 838, 640, { w: 230, h: 170, big: 64, rot: 0.03, s: popS(t0 + 0.05), fill: PR.tealL });
  prStamp('SLIPS TOO', 280, 1150, PR.red, t0 + 0.4, { rot: -0.06, size: 52, from: 1.3 });
  clawd(CLAWD_X, CLAWD_Y, { hat: 'phones', key: 'c11', s: 0.9, look: -12, mood: 'wow' });
  prSign('HEAVY BOOKS SLIP', { kicker: KICK });
  prChips();
};

// ---- 12. the catch: what this does not show
SCENES.catch = () => {
  const ws = VW('catch', 'simulation').t0, wr = VW('catch', 'training').t0;
  prWall('#4a4566', { panel: 'rgba(0,0,0,0.08)', streak: 300 });
  prFloor(1560, '#3b3855', { trim: false });
  rlCaveats(60, 470, 780, ['simulation only', 'scripted demos', 'one training run per setting'], ws, Math.max(0.3, (wr - ws) / 2));
  pp(pE(940, 520, 42, 42), PR.creamL, { key: 'moon', sb: 10, sy: 4, tex: 0.5 }); prStar(980, 640, 16); prStar(900, 700, 10);
  clawd(CLAWD_X, CLAWD_Y, { hat: 'beret', key: 'c12', s: 0.9, look: -14, mood: 'flat' });
  prSign('THE CATCH', { kicker: KICK });
  prChips();
};

// ---- 13. what touch adds; (split on "That's") the same table with an empty touch column
SCENES.next = () => {
  const t0 = E0, wc = VW('next', "can't").t0, ws = VW('next', 'slip').t0, wk = VW('next', 'skin').t0;
  prWall(PR.slate, { streak: 300 });
  prFloor(1520, PR.wood, { trim: false });
  pp(pR(40, 1180, 1000, 40, 6), PR.woodL, { key: 'bench', sb: 12, sy: 6 });
  prBunting(-10, undefined, 'bun13');
  withT(250, 700, -0.05, popS(t0 + 0.05), () => {
    pp(pR(-150, -95, 300, 190, 18), PR.char, { key: 'cam', sb: 16, sy: 8, tex: 0.4 });
    pp(pR(-110, -125, 90, 40, 8), PR.char, { key: 'camtop', sb: 6, sy: 3, tex: 0.4 });
    pp(pE(10, 0, 70, 70), PR.greyD, { key: 'camlens', sb: 8, sy: 4 }); pp(pE(10, 0, 44, 44), '#1d2a3a', { key: 'camg', shadow: false, tex: 0.3 });
  });
  prCross(380, 610, wc + 0.1, 50);
  rlSkin(700, 760, wk, ws);
  clawd(CLAWD_X + 20, CLAWD_Y + 40, { hat: 'beret', key: 'c13', s: 0.82, look: -12, mood: TT > wk ? 'happy' : 'neutral' });
  prSign("CAMERAS CAN'T FEEL SLIP", { kicker: KICK });
  prChips();
};
SCENES.next2 = () => {
  const cells = {};
  for (const c of ['zs', 'ft', 'ex']) for (let i = 0; i < 4; i++) cells[c + i] = -1;
  prWall(PR.teal, { panel: 'rgba(255,255,255,0.06)', streak: 380 });
  prFloor(1560, PR.woodD, { trim: false });
  rlTable({ cells, skin: E0 + 0.15, hi: [3, E0 + 0.5] });
  clawd(CLAWD_X + 30, CLAWD_Y + 80, { hat: 'beret', key: 'c14', s: 0.72, look: -14, mood: 'happy' });
  prSign('NEXT: SAME TABLE + TOUCH', { kicker: KICK });
  prChips();
};

// =====================================================================
//  SHOTS + TIMELINE — one shot per line (in script order); SPLITS = [[lineId, word, n, sceneKey]] cut again on a word
// =====================================================================
const SPLITS = [['slip', 'expert', 0, 'slip2'], ['next', "that's", 0, 'next2']];
function SFX_CUES() {
  const last = (id) => VL(id).words[VL(id).words.length - 1].t0;
  const c = [{ k: 'type', t: 0.35, n: 22, d: 1.2 }, { k: 'pop', t: VW('hook', 'named').t0 }];
  c.push({ k: 'type', t: VL('setup').t0 + 0.3, n: 18, d: 1.1 }, { k: 'pop', t: VW('setup', 'four').t0 });
  for (const w of ['training', 'shuffled', 'narrower', 'heavy']) c.push({ k: 'pop', t: VW('suites', w).t0 });
  c.push({ k: 'pop', t: VW('zero', '0').t0 + 0.1 }, { k: 'thud', t: VW('zero', 'crashed').t0 + 0.16 });
  for (let i = 1; i < 7; i++) c.push({ k: 'tick', t: VW('tuned', '50').t0 + 0.09 * i });
  c.push({ k: 'thud', t: VW('tuned', '50', 1).t0 + 0.16 });
  for (const w of ['72', '73', '43']) c.push({ k: 'pop', t: VW('table', w).t0 }, { k: 'tick', t: VW('table', w).t0 + 0.2 });
  const wp = VW('expert', 'perfect').t0;
  c.push({ k: 'pop', t: wp }, { k: 'pop', t: wp + 0.15 }, { k: 'pop', t: wp + 0.3 }, { k: 'pop', t: VW('expert', '57').t0 });
  for (let i = 0; i < 8; i++) c.push({ k: 'tick', t: VL('contact').t0 + 0.1 + 0.12 * i });
  c.push({ k: 'thud', t: VW('contact', 'contact').t0 + 0.16 });
  c.push({ k: 'thud', t: VW('push', 'push').t0 + 0.46 }, { k: 'thud', t: VW('push', 'never').t0 + 0.16 });
  c.push({ k: 'thud', t: VW('shove', 'neighbors').t0 + 0.16 }, { k: 'thud', t: VW('slip', 'slip').t0 + 0.16 });
  c.push({ k: 'thud', t: VW('slip', 'expert').t0 + 0.38 });
  const ws = VW('catch', 'simulation').t0, wr = VW('catch', 'training').t0;
  for (let i = 0; i < 3; i++) c.push({ k: 'pop', t: ws + i * Math.max(0.3, (wr - ws) / 2) });
  c.push({ k: 'pop', t: VW('next', "can't").t0 + 0.1 }, { k: 'chime', t: last('next') });
  return c;
}
const SHOT_LIST = VOICE.lines.map((l, i) => [i === 0 ? 0 : CUT(l.id), l.id])
  .concat(SPLITS.map(([id, w, n, key]) => [VW(id, w, n).t0 - 0.08, key])).sort((a, b) => a[0] - b[0]);
const ERA_LIST = SHOT_LIST.map(([t0, sc], i) => [t0, i + 1 < SHOT_LIST.length ? SHOT_LIST[i + 1][0] : DURATION, () => SCENES[sc]()]);
const TIMELINE = {
  width: W, height: H, fps: FPS, duration: DURATION, frames: NFRAMES, loop: LOOP_T, safe: SAFE, format: 'voice-cut',
  shots: SHOT_LIST.map(([t0, sc], i) => ({ id: i + 1, key: sc, era: i, t0, t1: ERA_LIST[i][1], title: sc })),
  cuts: SHOT_LIST.slice(1).map(([t0]) => t0), morphs: [], cues: {},
  acts: [{ name: 'reel', role: 'journey', t0: 0, t1: DURATION }],
  board: SHOT_LIST.map(([t0, sc], i) => ({ t: (t0 + ERA_LIST[i][1]) / 2 + 0.3, title: sc, sound: '', next: 'cut' })),
  narration: VOICE.lines.map((l) => ({ t0: l.t0, t1: l.t1, text: l.text })),
};
window.TIMELINE = TIMELINE;
