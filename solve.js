// A faster drop-in for makefun-diff 1.0.0: run(input) returns exactly makefun-diff's
// patch of input { old, new }, as one string, but computed with a compact Myers
// search instead of jsdiff's linked-list paths.
//
// The rule, restated from makefun-diff 1.0.0's header (the source of truth):
// a text is cut after every "\n" (the only separator; "\r" is part of its line),
// a last line with no "\n" is a line too, and two lines are equal only when they
// are the same code units, newline or its absence included. The common prefix of
// lines is kept, then the common suffix of the lines left. Between them, Myers's
// greedy search from the first lines: for d = 0, 1, 2, ... edits and each
// diagonal k = x - y from -d to d in steps of 2, the furthest point a path of d
// edits reaches on k is one old line past the furthest (d-1)-point on diagonal
// k - 1 (a deletion) or one new line past the furthest (d-1)-point on diagonal
// k + 1 (an insertion), whichever lies further into the old text, and the
// deletion when both reach the same point; a move past the end of either text
// is no candidate, and no edge diagonal is pruned. From that point the path
// follows every run of lines equal in both as far as it goes. The first path
// to reach both ends is the edit script.
//
// Speed, for the same answer:
// - Lines are interned to small integers (one Map lookup per line), so every
//   comparison in the search is an int comparison, never a string comparison
//   (the search ends each of its millions of steps with one such comparison).
// - The search keeps only one furthest-x per (d, k) in one flat Int32Array,
//   reused across cases (the furthest point is all the search needs; the path
//   is rebuilt afterwards), so the hot loop allocates nothing per step and
//   stores no per-step history.
// - The edit script is rebuilt by walking the stored furthest-x table
//   backwards, re-applying the forward step's exact choice (same validity
//   checks, same deletion-on-tie rule), which reproduces the same first path
//   to both ends, including its snakes.

/** The patch's first word. */
export const FORMAT = 'makefun-diff/1';

/** The kept lines a hunk shows before and after each change run. */
export const CONTEXT = 3;

const KEEP = 0;
const DELETE = 1;
const INSERT = 2;

/**
 * A tokenized text: parts is text.split('\n'), last is parts.length - 1, and
 * line i (0 <= i < n) is parts[i] with a "\n" when i < last. The line count
 * is last + (parts[last] !== '' ? 1 : 0). ids holds one interned integer per
 * line: part id * 2 + (has-newline ? 1 : 0), so two lines are equal exactly
 * when their ids are equal.
 */
function tokenize(text, map, out) {
  const parts = text.split('\n');
  const last = parts.length - 1;
  const n = last + (parts[last] !== '' ? 1 : 0);
  const ids = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    const part = parts[i];
    let pid = map.get(part);
    if (pid === undefined) {
      pid = map.size;
      map.set(part, pid);
    }
    ids[i] = pid * 2 + (i < last ? 1 : 0);
  }
  out.parts = parts;
  out.last = last;
  return ids;
}

/**
 * The furthest-x table, one flat Int32Array reused across cases: row d
 * starts at offset d * d and holds, at d * d + k + d, the furthest x (old
 * lines consumed, after the snake) a path of d edits reaches on diagonal k,
 * or -1 when no such path exists. Rows hold 2d + 1 entries, so the rows of
 * depths 0..D fill (D + 1) * (D + 1) entries.
 */
let vbuf = new Int32Array(1 << 20);
function ensureVcap(n) {
  if (n <= vbuf.length) return;
  let cap = vbuf.length;
  while (cap < n) cap *= 2;
  const nb = new Int32Array(cap);
  nb.set(vbuf);
  vbuf = nb;
}
function vrow(d, need) {
  const end = d * d + need;
  if (end > vbuf.length) ensureVcap(end);
  return vbuf;
}

/**
 * A lower bound on the middle's edit distance: every line the old slice has
 * more of than the new slice must be deleted, and vice versa inserted, so
 * D >= sum |count_old - count_new| over lines. (The slices' ids are
 * interned line ids, all below maxId.)
 */
let cntbuf = new Int32Array(1 << 16);
function lowerBound(a, a0, a1, b, b0, b1, maxId) {
  if (maxId > cntbuf.length) {
    let cap = cntbuf.length;
    while (cap < maxId) cap *= 2;
    cntbuf = new Int32Array(cap);
  }
  cntbuf.fill(0, 0, maxId);
  for (let i = a0; i < a1; i++) cntbuf[a[i]]++;
  for (let j = b0; j < b1; j++) cntbuf[b[j]]--;
  let lb = 0;
  for (let i = 0; i < maxId; i++) {
    const v = cntbuf[i];
    lb += v < 0 ? -v : v;
  }
  return lb;
}

/**
 * The forward greedy search, restricted to the band of diagonals that a
 * shortest path of length at most dHat can still be on: at depth d, a path
 * that reaches both ends in dHat edits sits on a diagonal k with
 * |k - (N - M)| <= dHat - d. With dHat >= D this finds the same first path
 * as the unbanded search: the true path never leaves the band, and neither
 * does any path that attains a furthest-x the true path's choices read.
 * Returns D, or -1 when the band runs out (dHat < D): the caller then
 * retries with a dHat that cannot be too small.
 */
function forward(a, a0, N, b, b0, M, dHat) {
  const kend = N - M;
  let x = 0;
  while (x < N && x < M && a[a0 + x] === b[b0 + x]) x++;
  vbuf[0] = x;
  if (x >= N && x >= M) return 0;
  for (let d = 1; ; d++) {
    const prevBase = (d - 1) * (d - 1);
    const curBase = d * d;
    vrow(d, 2 * d + 1);
    const w = dHat - d;
    let klo = -d > kend - w ? -d : kend - w;
    let khi = d < kend + w ? d : kend + w;
    klo += (d - klo) & 1;
    khi -= (khi - d) & 1;
    if (klo > khi) return -1;
    let alive = false;
    // k = -d: insertion only, from diagonal -d + 1 (prev index 0).
    if (klo === -d) {
      const v = vbuf[prevBase];
      let xx = -1;
      if (v >= 0 && v + d - 1 < M) xx = v;
      if (xx >= 0) {
        let y = xx + d;
        while (xx < N && y < M && a[a0 + xx] === b[b0 + y]) {
          xx++;
          y++;
        }
        alive = true;
      }
      vbuf[curBase] = xx;
      // Only diagonal N - M can be at both ends (here k = -d).
      if (-d === kend && xx >= N) return d;
    }
    // Interior diagonals: both candidates always in range, no bounds
    // checks. Deletion from k - 1 (prev index k + d - 2), insertion from
    // k + 1 (prev index k + d).
    const kStart = klo === -d ? -d + 2 : klo;
    const kEnd = khi === d ? d - 2 : khi;
    for (let k = kStart; k <= kEnd; k += 2) {
      const xd0 = vbuf[prevBase + k + d - 2];
      const xi0 = vbuf[prevBase + k + d];
      const xd = xd0 >= 0 && xd0 < N ? xd0 : -1;
      const xi = xi0 >= 0 && xi0 - (k + 1) < M ? xi0 : -1;
      if (xd < 0 && xi < 0) {
        vbuf[curBase + k + d] = -1;
        continue;
      }
      // Further into the old text wins; a tie goes to the deletion.
      let xx;
      if (xd < 0) xx = xi;
      else if (xi < 0) xx = xd + 1;
      else xx = xd >= xi - 1 ? xd + 1 : xi;
      let y = xx - k;
      while (xx < N && y < M && a[a0 + xx] === b[b0 + y]) {
        xx++;
        y++;
      }
      vbuf[curBase + k + d] = xx;
      alive = true;
      // Only diagonal N - M can be at both ends.
      if (k === kend && xx >= N) return d;
    }
    // k = d: deletion only, from diagonal d - 1 (prev last index).
    if (khi === d) {
      const v = vbuf[prevBase + 2 * d - 2];
      let xx = -1;
      if (v >= 0 && v < N) {
        xx = v + 1;
        let y = xx - d;
        while (xx < N && y < M && a[a0 + xx] === b[b0 + y]) {
          xx++;
          y++;
        }
        alive = true;
      }
      vbuf[curBase + 2 * d] = xx;
      if (d === kend && xx >= N) return d;
    }
    // No in-band path of length d: no longer path can enter the band.
    if (!alive) return -1;
  }
}

/**
 * The edit script of old lines [a0, a1) into new lines [b0, b1), as runs
 * [{ kind, count }] in order, following the header's rule exactly.
 * Prefix/suffix trimming has already been done by the caller, so the first
 * lines differ and the last lines differ whenever both slices are non-empty.
 * a and b are Int32Arrays of interned line ids; maxId bounds those ids.
 */
function editRuns(a, a0, a1, b, b0, b1, maxId) {
  const N = a1 - a0;
  const M = b1 - b0;
  if (N === 0) return M === 0 ? [] : [{ kind: INSERT, count: M }];
  if (M === 0) return [{ kind: DELETE, count: N }];

  // Banded first: 2 * the multiset lower bound is above D on every case
  // the published generator draws (D / LB stays under 1.8); the fallback
  // band N + M can never be too small, since deleting everything and
  // inserting everything is a script of that length. Size the table for
  // the banded band up front, so the search never grows it mid-run.
  const lb = lowerBound(a, a0, a1, b, b0, b1, maxId);
  const dBand = 2 * lb;
  ensureVcap((dBand + 1) * (dBand + 1));
  let D = forward(a, a0, N, b, b0, M, dBand);
  if (D < 0) D = forward(a, a0, N, b, b0, M, N + M);

  // Backtrack from (N, M). At depth d on diagonal k, the predecessor is
  // re-derived with the forward step's exact choice, so the walk follows
  // the same first path; the snake between the edit's landing point and
  // the stored furthest point is the run of kept lines.
  const rev = [];
  let cx = N;
  let cy = M;
  for (let d = D; d >= 1; d--) {
    const k = cx - cy;
    const prevBase = (d - 1) * (d - 1);
    const plen = 2 * d - 1;
    let xd = -1;
    const iDel = k + d - 2;
    if (iDel >= 0 && iDel < plen) {
      const v = vbuf[prevBase + iDel];
      if (v >= 0 && v < N) xd = v;
    }
    let xi = -1;
    const iIns = k + d;
    if (iIns >= 0 && iIns < plen) {
      const v = vbuf[prevBase + iIns];
      if (v >= 0 && v - (k + 1) < M) xi = v;
    }
    let isDel;
    if (xd < 0) isDel = false;
    else if (xi < 0) isDel = true;
    else isDel = xd >= xi - 1;
    let px;
    let py;
    if (isDel) {
      px = xd;
      py = xd - (k - 1);
    } else {
      px = xi;
      py = xi - (k + 1);
    }
    const x0 = isDel ? px + 1 : px;
    const snake = cx - x0;
    if (snake > 0) rev.push({ kind: KEEP, count: snake });
    rev.push({ kind: isDel ? DELETE : INSERT, count: 1 });
    cx = px;
    cy = py;
  }
  if (cx > 0) rev.push({ kind: KEEP, count: cx });

  // Reverse, merging runs of one kind (a deletion can follow a deletion
  // across two depths with no kept line between them, and the same for
  // insertions; the reference merges those into one component).
  const runs = [];
  for (let i = rev.length - 1; i >= 0; i--) {
    const r = rev[i];
    const last = runs[runs.length - 1];
    if (last !== undefined && last.kind === r.kind) last.count += r.count;
    else runs.push({ kind: r.kind, count: r.count });
  }
  return runs;
}

/** A line as the patch writes it: its op, the line, a newline, and "%" on a line of its own when it had none. */
function written(op, t, i) {
  const ch = op === KEEP ? '=' : op === DELETE ? '-' : '+';
  return i < t.last ? ch + t.parts[i] + '\n' : ch + t.parts[i] + '\n%\n';
}

/** make.fun's own patch of input.old into input.new (the header's rule). */
export function run(input) {
  if (input === null || typeof input !== 'object' || typeof input.old !== 'string' || typeof input.new !== 'string') {
    throw new TypeError('makefun-diff: the input is { old, new }, two strings');
  }
  const map = new Map();
  const aOut = {};
  const bOut = {};
  const a = tokenize(input.old, map, aOut);
  const b = tokenize(input.new, map, bOut);
  const n = a.length;
  const m = b.length;

  // The common prefix, then the common suffix of what is left.
  let prefix = 0;
  while (prefix < n && prefix < m && a[prefix] === b[prefix]) prefix++;
  let suffix = 0;
  while (suffix < n - prefix && suffix < m - prefix && a[n - 1 - suffix] === b[m - 1 - suffix]) suffix++;
  const midRuns = editRuns(a, prefix, n - suffix, b, prefix, m - suffix, 2 * map.size);

  const runs = [];
  if (prefix > 0) runs.push({ kind: KEEP, count: prefix });
  for (const r of midRuns) {
    const last = runs[runs.length - 1];
    if (last !== undefined && last.kind === r.kind) last.count += r.count;
    else runs.push({ kind: r.kind, count: r.count });
  }
  if (suffix > 0) {
    const last = runs[runs.length - 1];
    if (last !== undefined && last.kind === KEEP) last.count += suffix;
    else runs.push({ kind: KEEP, count: suffix });
  }

  // The change runs: each a stretch of deletions and insertions between two
  // kept lines, at old line o and new line p.
  const changes = [];
  let o = 0;
  let p = 0;
  let deleted = 0;
  let inserted = 0;
  for (const r of runs) {
    if (r.kind === KEEP) {
      o += r.count;
      p += r.count;
      continue;
    }
    let c = changes[changes.length - 1];
    if (c === undefined || c.o + c.dels !== o || c.p + c.ins !== p) {
      c = { o, p, dels: 0, ins: 0 };
      changes.push(c);
    }
    if (r.kind === DELETE) {
      c.dels += r.count;
      o += r.count;
      deleted += r.count;
    } else {
      c.ins += r.count;
      p += r.count;
      inserted += r.count;
    }
  }

  // The hunks: change runs at most 2 * CONTEXT kept lines apart share one.
  const hunks = [];
  for (const c of changes) {
    const h = hunks[hunks.length - 1];
    const prev = h !== undefined ? h[h.length - 1] : undefined;
    if (prev !== undefined && c.o - (prev.o + prev.dels) <= 2 * CONTEXT) h.push(c);
    else hunks.push([c]);
  }

  const out = [`${FORMAT} old=${n} new=${m} hunks=${hunks.length} deleted=${deleted} inserted=${inserted}\n`];
  for (const h of hunks) {
    const first = h[0];
    const last = h[h.length - 1];
    const oldStart = Math.max(0, first.o - CONTEXT);
    const newStart = first.p - (first.o - oldStart);
    const after = Math.min(CONTEXT, n - (last.o + last.dels));
    const oldEnd = last.o + last.dels + after;
    const newEnd = last.p + last.ins + after;
    out.push(`@ ${oldStart + 1} ${oldEnd - oldStart} ${newStart + 1} ${newEnd - newStart}\n`);
    let at = oldStart;
    for (const c of h) {
      for (; at < c.o; at++) out.push(written(KEEP, aOut, at));
      for (let i = 0; i < c.dels; i++) out.push(written(DELETE, aOut, c.o + i));
      for (let i = 0; i < c.ins; i++) out.push(written(INSERT, bOut, c.p + i));
      at = c.o + c.dels;
    }
    for (; at < oldEnd; at++) out.push(written(KEEP, aOut, at));
  }
  return out.join('');
}
