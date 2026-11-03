# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Paul Owen
"""Split metered gas into heating, hot water and hob, per 30-minute meter slot.

Inputs are state histories: the cumulative gas meter, the thermostat's hvac_action/preset_mode,
humidity in hot-water rooms (showers and taps) and the kitchen (hob), and optionally 5-minute electricity means.
A humidity value of None marks the sensor unavailable. A gas reading may carry a third value, when the meter
last reported it unchanged.
"""
import bisect
import datetime as dt
from dataclasses import dataclass, field
from zoneinfo import ZoneInfo

SLOT = 1800  # the meter books gas in half hours, a few seconds after each one ends

# Event detection: a rise of `rise` points within `within` seconds starts an event; it ends at the peak
# once the value has not beaten it for `settle` seconds or has dropped `drop` points.
SHOWER = dict(rise=5, within=300, settle=300, drop=3)
HOB = dict(rise=6, within=900, settle=300, drop=3)
HOB_AFTER_SHOWER = 1800  # bathroom steam reaching the kitchen is not cooking
OVEN_KW = 1.5            # an electric load above this for most of a kitchen event = oven, not hob
MIN_EVENT = 180          # shortest event rates are learned from
MIN_HOB = 120            # shortest kitchen rise taken as a hob session: a kettle-filled pan steams within minutes
SENSOR_GAP = 600         # a humidity sensor away this long (battery, re-pair) reads wrong when it comes back:
SENSOR_SETTLE = 1800     # warmed by handling, its humidity climbs back for an hour, so skip its first half hour
MIN_SHOWER = 60          # brief hot-water draws in a shower room still claim their share of a slot
SHOWER_RISE = 20         # a shower always steams the bathroom (+33..42 points); basin draws lift it +6..12
MAX_KW = {'dhw': 35.0, 'tap': 35.0, 'hob': 3.0}  # evidence above these is something else (a domestic burner tops out ~3 kW)

# Pans filled from the kettle: a short burst (5-minute max >= 2.5 kW for at most two buckets, then the load
# drops) with the boiler idle stands in for a hob session when the kitchen humidity shows none.
KETTLE_KW = 2.5
KETTLE_BUCKETS = 2
KETTLE_AFTER_KW = 0.8
KETTLE_HOB = 1200
KETTLE_GAS = 0.1  # the slot after the kettle must have its own reading of at least this, unexplained
KETTLE_STEAM = 300  # a kitchen rise that starts while the kettle boils and peaks this soon after it stops is its steam

# Priors (kW) and their weight in hours of evidence. A tap runs the burner flat out with no mixing down,
# and a short draw also pays for heating the exchanger and pipes: 25 Oct basin fill 0.647 kWh in 140 s.
PRIOR_KW = {'heat': (17.0, 1.0), 'dhw': (14.0, 0.25), 'tap': (17.0, 0.1), 'hob': (1.3, 0.3)}
RATE = {'shower': 'dhw', 'tap': 'tap', 'hob': 'hob'}  # event kind -> rate key
STAT_USES = ('heating', 'hot_water', 'hob', 'boost_heating')  # uses kept as hourly statistics


@dataclass
class Event:
    kind: str    # 'shower', 'tap' or 'hob'
    start: float
    end: float
    rise: float
    inferred: bool = False  # from a kettle cue: the timing is a guess, so no rates are learned from it


@dataclass
class Slot:
    start: float             # UTC timestamp
    gas: float = 0.0
    heat: float = 0.0        # seconds calling for heat outside showers (a combi serves hot water first)
    boost_heat: float = 0.0  # of which with boost on
    shower: float = 0.0
    tap: float = 0.0
    hob: float = 0.0
    lumped: bool = False     # gas booked late across several slots and not explained by activity
    parts: dict = field(default_factory=dict)


def settled(samples: list[tuple[float, float | None]]) -> list[list[tuple[float, float]]]:
    """Runs of available readings, each starting once the sensor has settled after a long absence."""
    runs, run, gone, until = [], [], None, float('-inf')
    for t, v in samples:
        if v is None:
            if gone is None:
                gone = t
            continue
        if gone is not None:
            if run:
                runs.append(run)
                run = []
            if t - gone >= SENSOR_GAP:
                until = t + SENSOR_SETTLE
            gone = None
        if t >= until:
            run.append((t, v))
    if run:
        runs.append(run)
    return runs


def detect(samples: list[tuple[float, float | None]], kind: str, rise: float, within: float, settle: float,
           drop: float, min_len: float = MIN_EVENT) -> list[Event]:
    """Humidity rises fast enough to be a shower or cooking."""
    return [e for run in settled(samples) for e in _detect(run, kind, rise, within, settle, drop, min_len)]


def _detect(samples: list[tuple[float, float]], kind: str, rise: float, within: float, settle: float,
            drop: float, min_len: float) -> list[Event]:
    times = [t for t, _ in samples]
    out, ev, floor = [], None, float('-inf')
    for j, (t, v) in enumerate(samples):
        if ev is None:
            # the value at the window start, then every sample inside it; never reaching back into the last event
            i0 = max(bisect.bisect_left(times, t - within, 0, j) - 1, bisect.bisect_left(times, floor, 0, j), 0)
            if i0 >= j:
                continue
            win = range(i0, j)
            lo = min(samples[i][1] for i in win)
            if v - lo >= rise:
                last_lo = max(i for i in win if samples[i][1] == lo)
                ev = Event(kind, samples[last_lo + 1][0], t, v - lo)
                peak, peak_t, base = v, t, lo
        else:
            if v > peak:
                peak, peak_t = v, t
                ev.rise = peak - base
            elif t - peak_t >= settle or v <= peak - drop:
                ev.end = floor = peak_t
                out.append(ev)
                ev = None
    if ev is not None:
        ev.end = peak_t
        out.append(ev)
    return [e for e in out if e.end - e.start >= min_len]


def kettles(power: list[tuple]) -> list[tuple[float, float]]:
    """Short high bursts in 5-minute (start, mean, max) buckets: a kettle, not an oven."""
    rows = sorted(p for p in power if len(p) > 2 and p[2] is not None)
    out, run = [], []
    for i, (t, m, mx) in enumerate(rows):
        if mx >= KETTLE_KW and (not run or t - run[-1][0] <= 300):
            run.append((t, m, mx))
            continue
        if run:
            if len(run) <= KETTLE_BUCKETS and (t - run[-1][0] > 300 or m < KETTLE_AFTER_KW):
                out.append((run[0][0], run[-1][0] + 300))
            run = [(t, m, mx)] if mx >= KETTLE_KW else []
    if run and len(run) <= KETTLE_BUCKETS:
        out.append((run[0][0], run[-1][0] + 300))
    return out


def kettle_off(power: list[tuple], end: float) -> float:
    """When a kettle burst ending in the bucket before `end` switched off, from that bucket's mean."""
    rows = {p[0]: p for p in power if len(p) > 2 and p[2] is not None}
    last, nxt = rows.get(end - 300), rows.get(end)
    if not last:
        return end
    base = nxt[1] if nxt else 0.0
    on = (last[1] - base) / (last[2] - base) if last[2] > base else 1.0
    return end - 300 + 300 * min(max(on, 0.0), 1.0)


def events(showers: list[list], hobs: list[list], power: list[tuple] | None,
           climate: list[tuple[float, str, str]] | None = None) -> list[Event]:
    evs = [e for s in showers for e in detect(s, 'shower', **SHOWER, min_len=MIN_SHOWER)]
    for e in evs:
        if e.rise < SHOWER_RISE:
            e.kind = 'tap'
    after_shower = lambda t: any(0 <= t - x.end <= HOB_AFTER_SHOWER or x.start <= t <= x.end
                                 for x in evs if x.kind == 'shower')
    bursts = kettles(power) if power else []
    boils = [(ks, kettle_off(power, ke)) for ks, ke in bursts]
    for s in hobs:
        for e in detect(s, 'hob', **HOB, min_len=MIN_HOB):
            if after_shower(e.start):
                continue
            if any(ks <= e.start <= off and e.end <= off + KETTLE_STEAM for ks, off in boils):
                continue
            if power:
                inside = [p[1] for p in power if e.start - 300 < p[0] < e.end]
                if inside and sum(m > OVEN_KW for m in inside) > len(inside) / 2:
                    continue
            evs.append(e)
    if power and hobs:
        heat = _intervals([(t, a == 'heating') for t, a, _ in climate], float('-inf'), float('inf')) if climate else []
        for ks, ke in bursts:
            if after_shower(ke) or any(ks - 600 <= e.start <= ke + KETTLE_HOB for e in evs if e.kind == 'hob'):
                continue
            if any(a < ke + KETTLE_HOB and b > ks for a, b in heat):
                continue
            evs.append(Event('hob', ke, ke + KETTLE_HOB, 0.0, inferred=True))
    return sorted(evs, key=lambda e: e.start)


def _intervals(changes: list[tuple[float, bool]], t0: float, t1: float) -> list[tuple[float, float]]:
    """On-intervals from (time, on) changes, clipped to [t0, t1)."""
    out, on_since = [], None
    for t, on in changes:
        t = max(t, t0)
        if on and on_since is None:
            on_since = t
        elif not on and on_since is not None:
            if t > on_since:
                out.append((on_since, min(t, t1)))
            on_since = None
    if on_since is not None and t1 > on_since:
        out.append((on_since, t1))
    return [(a, b) for a, b in out if b > a and a < t1]


def _add(slots: dict, a: float, b: float, attr: str) -> None:
    while a < b:
        k = int(a // SLOT) * SLOT
        e = min(b, k + SLOT)
        if k in slots:
            setattr(slots[k], attr, getattr(slots[k], attr) + e - a)
        a = e


def _overlap(xs, ys):
    out = []
    for a, b in xs:
        for c, d in ys:
            lo, hi = max(a, c), min(b, d)
            if hi > lo:
                out.append((lo, hi))
    return out


def _minus(xs, ys):
    """xs with ys removed."""
    out = []
    for a, b in xs:
        pieces = [(a, b)]
        for c, d in ys:
            nxt = []
            for p, q in pieces:
                if d <= p or c >= q:
                    nxt.append((p, q))
                else:
                    if c > p:
                        nxt.append((p, c))
                    if d < q:
                        nxt.append((d, q))
            pieces = nxt
        out += pieces
    return out


def build_slots(t0: float, t1: float, gas: list[tuple], climate: list[tuple[float, str, str]],
                evs: list[Event], kw: dict[str, float]) -> dict[float, Slot]:
    """Slots in [t0, t1) with activity seconds and the gas booked for them."""
    t0 = int(t0 // SLOT) * SLOT
    slots = {k: Slot(k) for k in range(int(t0), int(t1), SLOT)}
    heat = _intervals([(t, a == 'heating') for t, a, _ in climate], t0, t1)
    boost = _intervals([(t, p == 'boost') for t, _, p in climate], t0, t1)
    draws = [(e.start, e.end) for e in evs if e.kind in ('shower', 'tap')]
    for a, b in _minus(heat, draws):
        _add(slots, a, b, 'heat')
    for a, b in _overlap(_minus(heat, draws), boost):
        _add(slots, a, b, 'boost_heat')
    for e in evs:
        if not e.inferred:
            _add(slots, e.start, e.end, e.kind)

    def expected(s: Slot) -> float:
        return (kw['heat'] * s.heat + kw['dhw'] * s.shower + kw['tap'] * s.tap + kw['hob'] * s.hob) / 3600

    # Each reading books the slots since the meter last reported the previous value: a flat meter burnt nothing.
    # It always books at least the half hour before it, which may have ended just before an unchanged report.
    # Without report times the gas may belong to any slot since the previous reading.
    single = set()
    for (p, v0, *seen), (u, v1, *_) in zip(gas, gas[1:]):
        delta = v1 - v0
        if delta <= 0:
            continue
        if seen and seen[0] is not None:
            p = max(p, min(seen[0], u - SLOT))
        covered = [k for k in range(int(p // SLOT) * SLOT, int(u // SLOT) * SLOT, SLOT) if k in slots]
        if not covered:
            continue
        if len(covered) == 1:
            single.add(covered[0])
        w = [expected(slots[k]) for k in covered]
        if sum(w) > 0:
            for k, wk in zip(covered, w):
                slots[k].gas += delta * wk / sum(w)
            lumped = len(covered) > 1 and delta > 2 * sum(w) + 1.0
        else:
            slots[covered[-1]].gas += delta
            lumped = len(covered) > 2 and delta > 1.0
        if lumped:
            for k in covered:
                slots[k].lumped = True

    # A kettle cue claims only gas booked for its own slot with nothing else to explain it
    for e in evs:
        if not e.inferred:
            continue
        for k in range(int(e.start // SLOT) * SLOT, int(e.end), SLOT):
            s = slots.get(k)
            if s and k in single and s.gas >= KETTLE_GAS and s.heat == 0 and s.shower == 0 and s.tap == 0 and s.hob == 0:
                s.hob = min(e.end, k + SLOT) - max(e.start, k)
    return slots


def claimed(slots: dict[float, Slot], e: Event) -> bool:
    """Whether a kettle cue took gas; one that didn't was tea."""
    return any(slots[k].hob > 0 for k in range(int(e.start // SLOT) * SLOT, int(e.end), SLOT) if k in slots)


def split_slot(s: Slot, kw: dict[str, float]) -> dict[str, float]:
    """Share the slot's gas in proportion to the expected draw of each use; hot water = shower + taps."""
    w = {'heating': kw['heat'] * s.heat, 'shower': kw['dhw'] * s.shower, 'taps': kw['tap'] * s.tap,
         'hob': kw['hob'] * s.hob}
    tot = sum(w.values())
    if tot == 0:
        # gas with nothing seen: hot taps elsewhere (washing up, the kitchen sink)
        w = {'heating': 0.0, 'shower': 0.0, 'taps': 1.0, 'hob': 0.0}
        tot = 1.0
    parts = {k: s.gas * v / tot for k, v in w.items()}
    parts['hot_water'] = parts['shower'] + parts['taps']
    return parts


def evidence(slots: dict[float, Slot], evs: list[Event], tz: ZoneInfo) -> dict[dt.date, dict]:
    """Clean kWh and hours per use and local day: heating-only slots, showers, taps and hob sessions on their own."""
    out: dict[dt.date, dict] = {}

    def add(t, key, g, h):
        e = out.setdefault(dt.datetime.fromtimestamp(t, tz).date(), {})
        e.setdefault(key, [0.0, 0.0])
        e[key][0] += g
        e[key][1] += h

    for s in slots.values():
        if not s.lumped and s.heat >= 300 and s.shower == 0 and s.tap == 0 and s.hob == 0 and s.gas > 0:
            add(s.start, 'heat', s.gas, s.heat / 3600)
    for e in evs:
        if e.inferred or e.end - e.start < (MIN_SHOWER if e.kind == 'tap' else MIN_EVENT):
            continue
        ks = [k for k in slots if k < e.end and k + SLOT > e.start]
        # the slot after the event may hold the tail
        nxt = int(e.end // SLOT) * SLOT + SLOT
        if nxt in slots and e.end - int(e.end // SLOT) * SLOT > SLOT - 300:
            ks.append(nxt)
        group = [slots[k] for k in ks]
        if not group or any(g.lumped or g.heat > 0 for g in group):
            continue
        if any(getattr(g, k) > 0 for g in group for k in RATE if k != e.kind):
            continue
        # another event sharing these slots would be counted against this one's hours
        if any(x is not e and not x.inferred and x.start < max(ks) + SLOT and x.end > min(ks) for x in evs):
            continue
        g, h = sum(x.gas for x in group), (e.end - e.start) / 3600
        key = RATE[e.kind]
        if 0 < g <= MAX_KW[key] * h:
            add(e.start, key, g, h)
    return out


def kw_from(days: list[dict], prior: dict = PRIOR_KW) -> dict[str, float]:
    """Pooled kW over days of evidence, shrunk towards the priors."""
    gas = {k: p[0] * p[1] for k, p in prior.items()}
    hours = {k: p[1] for k, p in prior.items()}
    for d in days:
        for k, (g, h) in d.items():
            gas[k] += g
            hours[k] += h
    return {k: gas[k] / hours[k] for k in gas}


def learn_kw(slots: dict[float, Slot], evs: list[Event], tz: ZoneInfo, prior: dict = PRIOR_KW) -> dict[str, float]:
    return kw_from(list(evidence(slots, evs, tz).values()), prior)


def daily(slots: dict[float, Slot], kw: dict[str, float], tz: ZoneInfo) -> dict[dt.date, dict]:
    """Per local day (of each slot's start): heating, hot water (shower + taps), hob, boost heating and quality flags."""
    out: dict[dt.date, dict] = {}
    for s in sorted(slots.values(), key=lambda s: s.start):
        d = dt.datetime.fromtimestamp(s.start, tz).date()
        day = out.setdefault(d, {'heating': 0.0, 'hot_water': 0.0, 'shower': 0.0, 'taps': 0.0, 'hob': 0.0,
                                 'boost_heating': 0.0, 'gas': 0.0, 'lumped': False, 'slots': 0})
        parts = split_slot(s, kw)
        parts['boost_heating'] = parts['heating'] * s.boost_heat / s.heat if s.heat else 0.0
        s.parts = parts
        for k, v in parts.items():
            day[k] += v
        day['gas'] += s.gas
        day['lumped'] |= s.lumped
        day['slots'] += 1
    return out


def hourly(slots: dict[float, Slot], until: float) -> dict[int, dict[str, float]]:
    """Gas per use and clock hour (UTC start) for hours ending by `until`, from slots split by daily()."""
    out: dict[int, dict[str, float]] = {}
    for s in slots.values():
        h = int(s.start // 3600) * 3600
        if h + 3600 <= until:
            row = out.setdefault(h, dict.fromkeys(STAT_USES, 0.0))
            for k in STAT_USES:
                row[k] += s.parts.get(k, 0.0)
    return out


def cumulative(hours: dict[int, dict[str, float]], base: dict[str, float],
               tz: ZoneInfo) -> tuple[dict[str, list[tuple[int, float]]], dict[dt.date, dict[str, float]]]:
    """Running sums per use from `base`, and each local day's sums at its last hour."""
    run = {k: base.get(k, 0.0) for k in STAT_USES}
    rows: dict[str, list[tuple[int, float]]] = {k: [] for k in STAT_USES}
    ends: dict[dt.date, dict[str, float]] = {}
    for h in sorted(hours):
        for k in STAT_USES:
            run[k] += hours[h][k]
            rows[k].append((h, run[k]))
        ends[dt.datetime.fromtimestamp(h, tz).date()] = dict(run)
    return rows, ends
