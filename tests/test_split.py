# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Paul Owen
import datetime as dt

import pytest

from core import split
from fixtures_evenings import (BATHROOM_24, CLIMATE_24, GAS_22, GAS_24, KITCHEN_22, KITCHEN_24, POWER_22,
                               POWER_24, TZ, ts)

KW = {'heat': 17.0, 'dhw': 14.0, 'tap': 17.0, 'hob': 1.3}


def test_shower_detected_from_bathroom_humidity():
    evs = split.detect(BATHROOM_24, 'shower', **split.SHOWER)
    assert len(evs) == 1
    e = evs[0]
    assert e.start == ts('2026-10-24T23:15:05') and e.end == ts('2026-10-24T23:31:28')
    assert e.rise == 42


def test_hob_detected_through_a_kettle():
    evs = split.events([], [KITCHEN_22], POWER_22)
    assert [(e.kind, e.start, e.end) for e in evs] == [('hob', ts('2026-10-22T20:04:52'), ts('2026-10-22T20:24:19'))]


def test_kettle_steam_and_shower_drift_are_not_hob():
    evs = split.events([BATHROOM_24], [KITCHEN_24], POWER_24, CLIMATE_24)
    assert [e.kind for e in evs] == ['shower']


def test_oven_is_not_hob():
    oven = [(t, 2.5, 3.0) for t, *_ in POWER_22]
    assert split.events([], [KITCHEN_22], oven) == []


def test_evening_24_split():
    evs = split.events([BATHROOM_24], [KITCHEN_24], POWER_24, CLIMATE_24)
    slots = split.build_slots(ts('2026-10-24T16:30'), ts('2026-10-25T00:30'), GAS_24, CLIMATE_24, evs, KW)
    s17, s19 = slots[ts('2026-10-24T17:00')], slots[ts('2026-10-24T19:00')]
    assert s17.gas == pytest.approx(5.492) and s19.gas == pytest.approx(5.136)
    assert s19.boost_heat == s19.heat == pytest.approx(ts('2026-10-24T19:21:32') - ts('2026-10-24T19:03:27'))
    # 19:30-23:30 has no reading until 23:30; it all belongs to the shower slot
    assert slots[ts('2026-10-24T23:00')].gas == pytest.approx(3.907)
    assert slots[ts('2026-10-24T23:30')].gas == pytest.approx(0.112)
    assert not any(s.lumped for s in slots.values())
    days = split.daily(slots, KW, TZ)
    d24 = days[dt.date(2026, 10, 24)]
    assert d24['heating'] == pytest.approx(5.492 + 5.136)
    assert d24['boost_heating'] == pytest.approx(5.136)
    assert d24['hot_water'] == d24['shower'] == pytest.approx(4.019)
    assert d24['taps'] == 0
    assert d24['hob'] == 0


def test_learned_rates_from_the_two_evenings():
    evs = split.events([BATHROOM_24], [KITCHEN_24], POWER_24, CLIMATE_24)
    s24 = split.build_slots(ts('2026-10-24T16:30'), ts('2026-10-25T00:30'), GAS_24, CLIMATE_24, evs, KW)
    ev22 = split.events([], [KITCHEN_22], POWER_22)
    s22 = split.build_slots(ts('2026-10-22T19:00'), ts('2026-10-22T21:00'), GAS_22, [(ts('2026-10-22T19:00'), 'idle', 'none')], ev22, KW)
    kw = split.learn_kw({**s22, **s24}, ev22 + evs, TZ)
    # clean evidence: heat 10.63 kWh / 0.6 h, shower 4.02 kWh / 0.27 h, hob 0.446 kWh / 0.32 h
    assert 17 < kw['heat'] < 18
    assert 14 < kw['dhw'] < 15
    assert 1.3 < kw['hob'] < 1.4


def test_late_lump_is_flagged():
    t = ts('2026-10-20T00:00')
    gas = [(t, 0.0), (t + 10 * 3600 + 20, 8.0)]  # a day's gas booked at once, nothing seen
    slots = split.build_slots(t, t + 12 * 3600, gas, [(t, 'idle', 'none')], [], KW)
    assert all(s.lumped for k, s in slots.items() if k < t + 10 * 3600)


def test_heating_during_shower_counts_as_hot_water():
    t = ts('2026-10-26T07:00')
    clim = [(t, 'heating', 'none'), (t + 1800, 'idle', 'none')]
    ev = [split.Event('shower', t + 600, t + 1200, 30)]
    slots = split.build_slots(t, t + 1800, [(t, 0.0), (t + 1830, 7.0)], clim, ev, KW)
    s = slots[t]
    assert s.heat == 1200 and s.shower == 600
    parts = split.split_slot(s, KW)
    assert parts['hot_water'] == pytest.approx(7.0 * 14 * 600 / (14 * 600 + 17 * 1200))


def test_brief_bathroom_draw_shares_a_slot_with_the_hob():
    t = ts('2026-10-19T14:30')
    bath = [(t, 54.0), (t + 60, 56.0), (t + 100, 58.0), (t + 140, 60.0), (t + 400, 59.0), (t + 500, 56.0)]
    evs = split.events([bath], [], None)
    assert [(e.kind, e.end - e.start) for e in evs] == [('tap', 80.0)]
    hob = split.Event('hob', t + 300, t + 1500, 7)
    slots = split.build_slots(t, t + 1800, [(t, 0.0), (t + 1820, 1.0)], [(t, 'idle', 'none')], evs + [hob], KW)
    parts = split.split_slot(slots[t], KW)
    assert parts['hot_water'] > 0.4 and parts['hob'] < 0.6
    # too short to learn a rate from; and a 4 kW "hob" is not learned either
    assert split.evidence(slots, evs + [hob], TZ) == {}


def test_kettle_bursts_but_not_the_oven():
    from fixtures_evenings import POWER_21, POWER_23
    assert split.kettles(POWER_23) == [(ts('2026-10-23T19:25'), ts('2026-10-23T19:35'))]
    assert split.kettles(POWER_22) == [(ts('2026-10-22T20:00'), ts('2026-10-22T20:10'))]
    assert split.kettles(POWER_21) == []


def test_pasta_without_steam_found_from_the_kettle():
    from fixtures_evenings import CLIMATE_23, GAS_23, KITCHEN_23, POWER_23
    evs = split.events([], [KITCHEN_23], POWER_23, CLIMATE_23)
    assert [(e.kind, e.inferred, e.start) for e in evs] == [('hob', True, ts('2026-10-23T19:35'))]
    slots = split.build_slots(ts('2026-10-23T18:00'), ts('2026-10-23T20:30'), GAS_23, CLIMATE_23, evs, KW)
    day = split.daily(slots, KW, TZ)[dt.date(2026, 10, 23)]
    assert day['hob'] == pytest.approx(0.436, abs=1e-3)
    assert day['hot_water'] == pytest.approx(0.045, abs=1e-3)
    assert split.evidence(slots, evs, TZ) == {}  # guessed timing: no rate learned


def test_kettle_cue_skipped_when_steam_heating_or_a_shower_explain_it():
    # 22 Oct: humidity already shows the hob
    assert [e.inferred for e in split.events([], [KITCHEN_22], POWER_22)] == [False]
    # 24 Oct: 19:05 kettle during heating, 23:45 kettle after the shower
    evs = split.events([BATHROOM_24], [KITCHEN_24], POWER_24, CLIMATE_24)
    assert [e.kind for e in evs] == ['shower']


def test_tea_kettle_without_its_own_gas_is_not_hob():
    t = ts('2026-10-16T15:00')
    power = [(t, 0.3, 0.4), (t + 300, 1.2, 3.0), (t + 600, 0.3, 0.4)]
    evs = split.events([], [[(t, 55.0)]], power, [(t - 3600, 'idle', 'none')])
    assert [e.inferred for e in evs] == [True]
    # no reading after the kettle: nothing to claim
    slots = split.build_slots(t, t + 3600, [(t, 0.0)], [(t, 'idle', 'none')], evs, KW)
    assert all(s.hob == 0 for s in slots.values())
    # a late reading lumping several slots (a hot tap hours earlier) is not claimed either
    gas = [(t - 3 * 3600, 0.0), (t + 1820, 0.6)]
    slots = split.build_slots(t - 3 * 3600, t + 3600, gas, [(t - 3 * 3600, 'idle', 'none')], evs, KW)
    assert all(s.hob == 0 for s in slots.values())
    assert split.daily(slots, KW, TZ)[dt.date(2026, 10, 16)]['hot_water'] == pytest.approx(0.6)


def test_basin_fill_is_a_tap_and_teaches_the_tap_rate():
    from fixtures_evenings import BATHROOM_25, CLIMATE_25, GAS_25
    evs = split.events([BATHROOM_25], [], None, CLIMATE_25)
    assert [(e.kind, e.start, e.end) for e in evs] == [('tap', ts('2026-10-25T17:46:50'), ts('2026-10-25T17:49:10'))]
    slots = split.build_slots(ts('2026-10-25T08:30'), ts('2026-10-25T18:30'), GAS_25, CLIMATE_25, evs, KW)
    # the reading books every slot since 08:30; only the tap's slot can explain it
    assert slots[ts('2026-10-25T17:30')].gas == pytest.approx(0.647)
    assert not any(s.lumped for s in slots.values())
    day = split.daily(slots, KW, TZ)[dt.date(2026, 10, 25)]
    assert day['taps'] == day['hot_water'] == pytest.approx(0.647)
    assert day['shower'] == 0
    ev = split.evidence(slots, evs, TZ)[dt.date(2026, 10, 25)]
    assert list(ev) == ['tap'] and ev['tap'] == pytest.approx([0.647, 140 / 3600])
    kw = split.kw_from([ev])
    assert 16.6 < kw['tap'] < 17 and kw['dhw'] == 14.0


def test_unseen_gas_counts_as_taps():
    t = ts('2026-10-20T12:00')
    slots = split.build_slots(t, t + 1800, [(t, 0.0), (t + 1820, 0.3)], [(t, 'idle', 'none')], [], KW)
    parts = split.split_slot(slots[t], KW)
    assert parts['taps'] == parts['hot_water'] == pytest.approx(0.3) and parts['shower'] == 0


def test_tap_during_heating_is_not_heating_evidence():
    t = ts('2026-10-20T07:00')
    tap = split.Event('tap', t + 600, t + 740, 6)
    slots = split.build_slots(t, t + 1800, [(t, 0.0), (t + 1820, 4.0)], [(t, 'heating', 'none')], [tap], KW)
    assert slots[t].heat > 300 and slots[t].tap > 0
    assert split.evidence(slots, [tap], TZ) == {}


def test_kettle_steam_while_heating_is_not_hob_and_the_poached_egg_is():
    from fixtures_evenings import CLIMATE_26, GAS_26, KITCHEN_26, POWER_26
    evs = split.events([], [KITCHEN_26], POWER_26, CLIMATE_26)
    real = [e for e in evs if not e.inferred]
    assert [(e.kind, e.start, e.end) for e in real] == [('hob', ts('2026-10-26T20:37:39'), ts('2026-10-26T20:40:30'))]
    slots = split.build_slots(ts('2026-10-26T16:30'), ts('2026-10-26T21:30'), GAS_26, CLIMATE_26, evs, KW)
    # the 18:55 tea kettle claims nothing
    assert [split.claimed(slots, e) for e in evs if e.inferred] == [False]
    day = split.daily(slots, KW, TZ)[dt.date(2026, 10, 26)]
    assert day['hob'] == pytest.approx(0.088, abs=1e-3)
    assert day['taps'] == 0
    assert day['heating'] == pytest.approx(3.54, abs=1e-3)


def test_kettle_off_from_the_last_bucket():
    from fixtures_evenings import POWER_26
    assert split.kettle_off(POWER_26, ts('2026-10-26T17:10')) == pytest.approx(ts('2026-10-26T17:05:18'), abs=2)


def test_poached_egg_without_the_grill_is_still_hob():
    from fixtures_evenings import CLIMATE_26, KITCHEN_26
    t = ts('2026-10-26T20:20')
    # kettle 20:33-20:36 seen on its own, nothing after it
    power = [(t, 0.32, 0.35), (t + 300, 0.32, 0.35), (t + 600, 1.35, 2.9), (t + 900, 0.84, 2.9),
             (t + 1200, 0.32, 0.35), (t + 1500, 0.32, 0.35)]
    assert split.kettles(power) == [(t + 600, t + 1200)]
    evs = [e for e in split.events([], [KITCHEN_26], power, CLIMATE_26) if e.start > t]
    assert [(e.kind, e.inferred, e.start) for e in evs] == [('hob', False, ts('2026-10-26T20:37:39'))]


def test_hourly_statistics_add_up_to_the_days():
    evs = split.events([BATHROOM_24], [KITCHEN_24], POWER_24, CLIMATE_24)
    slots = split.build_slots(ts('2026-10-24T16:30'), ts('2026-10-25T00:30'), GAS_24, CLIMATE_24, evs, KW)
    days = split.daily(slots, KW, TZ)
    hours = split.hourly(slots, ts('2026-10-25T00:30'))
    # the 00:00 hour has not ended: left for the next refresh
    assert max(hours) == ts('2026-10-24T23:00') and ts('2026-10-24T16:00') in hours
    rows, ends = split.cumulative(hours, {'heating': 100.0}, TZ)
    d24 = days[dt.date(2026, 10, 24)]
    assert rows['heating'][-1][1] == pytest.approx(100 + d24['heating'])
    assert rows['boost_heating'][-1][1] == pytest.approx(d24['boost_heating'])
    assert rows['hot_water'][-1][1] == pytest.approx(d24['hot_water'])
    assert ends[dt.date(2026, 10, 24)]['heating'] == pytest.approx(100 + d24['heating'])
    # the 19:00 slot's boost lands in the 19:00 hour
    h19 = dict(rows['boost_heating'])
    assert h19[ts('2026-10-24T19:00')] - h19[ts('2026-10-24T18:00')] == pytest.approx(5.136)


def test_sensor_back_from_a_long_gap_is_not_hob():
    from fixtures_evenings import BATHROOM_3, KITCHEN_3
    evs = split.events([BATHROOM_3], [KITCHEN_3], None)
    assert [(e.kind, e.start) for e in evs] == [('tap', ts('2026-11-03T13:31:31'))]
    # the same readings with no gap would pass for cooking
    joined = [(t, v) for t, v in KITCHEN_3 if v is not None]
    assert [e.kind for e in split.detect(joined, 'hob', **split.HOB, min_len=split.MIN_HOB)] == ['hob']


def test_short_gap_is_not_settled():
    t = ts('2026-11-03T12:00')
    rows = [(t, 50.0), (t + 60, None), (t + 120, 50.0), (t + 300, 54.0), (t + 500, 58.0), (t + 700, 60.0),
            (t + 1100, 60.0)]
    assert [e.kind for e in split.detect(rows, 'hob', **split.HOB, min_len=split.MIN_HOB)] == ['hob']


def test_basin_draw_takes_the_reading_after_a_flat_meter():
    from fixtures_evenings import BATHROOM_3, CLIMATE_3, GAS_3, KITCHEN_3
    evs = split.events([BATHROOM_3], [KITCHEN_3], None, CLIMATE_3)
    slots = split.build_slots(ts('2026-11-03T09:00'), ts('2026-11-03T15:00'), GAS_3, CLIMATE_3, evs, KW)
    assert {k for k, s in slots.items() if s.gas} == {ts('2026-11-03T13:30')}
    day = split.daily(slots, KW, TZ)[dt.date(2026, 11, 3)]
    assert day['hob'] == 0 and day['heating'] == 0
    assert day['hot_water'] == pytest.approx(0.647)


def test_unchanged_report_just_after_the_half_hour_keeps_its_slot():
    t = ts('2026-11-03T13:30')
    gas = [(t - 3 * 3600, 0.0, t + 1810), (t + 1827, 0.6)]
    slots = split.build_slots(t - 3 * 3600, t + 3600, gas, [(t - 3 * 3600, 'idle', 'none')], [], KW)
    assert {k for k, s in slots.items() if s.gas} == {t}
