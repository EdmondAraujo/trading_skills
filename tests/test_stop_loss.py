# ABOUTME: Unit tests for stop-loss analytics functions.
# ABOUTME: All tests run without IBKR dependency — pure calculation coverage.

import pytest

from trading_skills.broker.stop_loss import (
    build_position_analysis,
    calc_short_premium_decay_pct,
    calc_stop_basis,
    calc_stop_price,
    detect_orphan_orders,
    identify_positions,
    summarize_all_conditional_orders,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _opt(symbol, quantity, avg_cost, strike, expiry, right="C", account="U123"):
    return {
        "account": account,
        "symbol": symbol,
        "sec_type": "OPT",
        "quantity": quantity,
        "avg_cost": avg_cost,
        "strike": strike,
        "expiry": expiry,
        "right": right,
    }


def _stk(symbol, quantity=100, avg_cost=150.0, account="U123"):
    return {
        "account": account,
        "symbol": symbol,
        "sec_type": "STK",
        "quantity": quantity,
        "avg_cost": avg_cost,
        "strike": None,
        "expiry": None,
        "right": None,
    }


# ---------------------------------------------------------------------------
# calc_stop_basis
# ---------------------------------------------------------------------------


def test_stop_basis_normal_uses_market_when_higher():
    assert calc_stop_basis(40.0, 30.0, forced=False) == pytest.approx(40.0)


def test_stop_basis_normal_uses_avg_cost_when_market_lower():
    assert calc_stop_basis(25.0, 35.0, forced=False) == pytest.approx(35.0)


def test_stop_basis_normal_falls_back_to_avg_cost_when_no_market():
    assert calc_stop_basis(None, 35.0, forced=False) == pytest.approx(35.0)


def test_stop_basis_normal_falls_back_when_market_zero():
    assert calc_stop_basis(0.0, 35.0, forced=False) == pytest.approx(35.0)


def test_stop_basis_forced_uses_current_price():
    assert calc_stop_basis(25.0, 35.0, forced=True) == pytest.approx(25.0)


def test_stop_basis_forced_falls_back_to_avg_cost_when_no_market():
    assert calc_stop_basis(None, 35.0, forced=True) == pytest.approx(35.0)


# ---------------------------------------------------------------------------
# calc_stop_price
# ---------------------------------------------------------------------------


def test_stop_price_50pct_normal():
    # basis = max(40, 30) = 40; stop = 40 * 0.5 = 20
    assert calc_stop_price(40.0, 30.0, stop_pct=50.0) == pytest.approx(20.0)


def test_stop_price_50pct_forced():
    # basis = current_mid = 25; stop = 25 * 0.5 = 12.5
    assert calc_stop_price(25.0, 35.0, stop_pct=50.0, forced=True) == pytest.approx(12.5)


def test_stop_price_custom_pct():
    # basis = 40; stop = 40 * 0.75 = 30
    assert calc_stop_price(40.0, 30.0, stop_pct=25.0) == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# calc_short_premium_decay_pct
# ---------------------------------------------------------------------------


def test_short_decay_pct_fully_intact():
    assert calc_short_premium_decay_pct(5.0, 5.0) == pytest.approx(0.0)


def test_short_decay_pct_fully_captured():
    assert calc_short_premium_decay_pct(5.0, 0.0) == pytest.approx(100.0)


def test_short_decay_pct_90pct():
    assert calc_short_premium_decay_pct(5.0, 0.50) == pytest.approx(90.0)


def test_short_decay_pct_zero_premium():
    assert calc_short_premium_decay_pct(0.0, 1.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# identify_positions
# ---------------------------------------------------------------------------


def test_identify_pmcc_basic():
    normalized = [
        _opt("NVDA", 3, 44.27, 200.0, "20270115"),  # long LEAPS
        _opt("NVDA", -3, -0.61, 235.0, "20260515"),  # short
    ]
    result = identify_positions(normalized)
    assert len(result) == 1
    pos = result[0]
    assert pos["type"] == "pmcc"
    assert pos["symbol"] == "NVDA"
    assert pos["qty"] == 3
    assert pos["leaps"]["strike"] == 200.0
    assert len(pos["shorts"]) == 1
    assert pos["shorts"][0]["strike"] == 235.0


def test_identify_naked_leaps():
    normalized = [_opt("NVDA", 3, 44.27, 200.0, "20270115")]
    result = identify_positions(normalized)
    assert len(result) == 1
    assert result[0]["type"] == "leaps"
    assert result[0]["leaps"]["strike"] == 200.0


def test_identify_stock():
    normalized = [_stk("AAPL", quantity=100, avg_cost=175.0)]
    result = identify_positions(normalized)
    assert len(result) == 1
    assert result[0]["type"] == "stock"
    assert result[0]["symbol"] == "AAPL"
    assert result[0]["qty"] == 100
    assert result[0]["avg_cost"] == 175.0


def test_identify_multiple_shorts_same_symbol():
    normalized = [
        _opt("IWM", 15, 21.28, 260.0, "20260918"),  # long LEAPS
        _opt("IWM", -15, -5.0, 280.0, "20260618"),  # short 1
        _opt("IWM", -10, -3.0, 285.0, "20260718"),  # short 2
    ]
    result = identify_positions(normalized)
    assert len(result) == 1
    assert result[0]["type"] == "pmcc"
    assert len(result[0]["shorts"]) == 2


def test_identify_mixed_portfolio():
    normalized = [
        _opt("NVDA", 3, 44.27, 200.0, "20270115"),
        _opt("NVDA", -3, -0.61, 235.0, "20260515"),
        _opt("SOLO", 2, 10.0, 50.0, "20260918"),  # naked LEAPS
        _stk("AAPL"),
    ]
    result = identify_positions(normalized)
    types = {p["type"] for p in result}
    assert types == {"pmcc", "leaps", "stock"}


def test_identify_ignores_short_stock_positions():
    # Short stock (negative qty) should be ignored
    normalized = [_stk("AAPL", quantity=-100)]
    result = identify_positions(normalized)
    assert result == []


# ---------------------------------------------------------------------------
# build_position_analysis
# ---------------------------------------------------------------------------


def _pmcc_pos(
    symbol="NVDA",
    qty=3,
    leaps_cost=44.27,
    leaps_strike=200.0,
    leaps_expiry="20270115",
    short_cost=0.61,
    short_strike=235.0,
    short_expiry="20260515",
    account="U123",
):
    return {
        "type": "pmcc",
        "symbol": symbol,
        "account": account,
        "qty": qty,
        "leaps": {
            "strike": leaps_strike,
            "expiry": leaps_expiry,
            "right": "C",
            "avg_cost": leaps_cost,
        },
        "shorts": [
            {
                "strike": short_strike,
                "expiry": short_expiry,
                "right": "C",
                "premium_received": short_cost,
                "qty": qty,
            }
        ],
    }


def _leaps_pos(symbol="SOLO", qty=2, avg_cost=10.0, strike=50.0, expiry="20260918", account="U123"):
    return {
        "type": "leaps",
        "symbol": symbol,
        "account": account,
        "qty": qty,
        "leaps": {"strike": strike, "expiry": expiry, "right": "C", "avg_cost": avg_cost},
    }


def _stock_pos(symbol="AAPL", qty=100, avg_cost=175.0, account="U123"):
    return {"type": "stock", "symbol": symbol, "account": account, "qty": qty, "avg_cost": avg_cost}


def test_build_pmcc_no_alert():
    pos = _pmcc_pos()
    result = build_position_analysis(
        position=pos,
        underlying_price=219.05,
        current_mid=44.23,
        short_mids=[0.56],
        existing_stop=None,
        stop_pct=50.0,
        forced=False,
    )
    assert result["type"] == "pmcc"
    assert result["stop_loss"]["action"] == "place_new"
    assert result["alert_soon"] is False
    assert result["leaps"]["stop_basis"] == pytest.approx(44.27)  # max(44.23, 44.27)
    assert result["leaps"]["stop_price"] == pytest.approx(22.14)


def test_build_pmcc_alert_soon():
    # LEAPS down 46% from basis — past early_warning_pct (25%)
    pos = _pmcc_pos(leaps_cost=2.93)
    result = build_position_analysis(
        position=pos,
        underlying_price=26.12,
        current_mid=1.58,
        short_mids=[0.24],
        existing_stop=None,
        stop_pct=50.0,
        forced=False,
    )
    assert result["alert_soon"] is True
    types = [a["type"] for a in result["alerts"]]
    assert "leaps_early_warning" in types


def test_build_stock_analysis():
    pos = _stock_pos()
    result = build_position_analysis(
        position=pos,
        underlying_price=189.50,
        current_mid=189.50,
        short_mids=[],
        existing_stop=None,
        stop_pct=50.0,
        forced=False,
    )
    assert result["type"] == "stock"
    # basis = max(189.50, 175.0) = 189.50; stop = 189.50 * 0.5 = 94.75
    assert result["stock"]["stop_basis"] == pytest.approx(189.50)
    assert result["stop_loss"]["stop_price"] == pytest.approx(94.75)
    assert result["alert_soon"] is False


def test_build_preserve_existing_stop():
    pos = _pmcc_pos()
    result = build_position_analysis(
        position=pos,
        underlying_price=219.05,
        current_mid=44.23,
        short_mids=[0.56],
        existing_stop=25.0,  # existing stop is higher (more protective) than new ~22
        stop_pct=50.0,
        forced=False,
    )
    assert result["stop_loss"]["action"] == "preserve_existing"


def test_build_overwrite_with_forced():
    pos = _pmcc_pos()
    result = build_position_analysis(
        position=pos,
        underlying_price=219.05,
        current_mid=44.23,
        short_mids=[0.56],
        existing_stop=25.0,
        stop_pct=50.0,
        forced=True,
    )
    assert result["stop_loss"]["action"] == "overwrite"


def test_build_forced_uses_current_mid_as_basis():
    pos = _pmcc_pos(leaps_cost=44.27)
    result = build_position_analysis(
        position=pos,
        underlying_price=219.05,
        current_mid=30.0,  # lower than avg_cost
        short_mids=[0.56],
        existing_stop=None,
        stop_pct=50.0,
        forced=True,  # forced: basis = current_mid = 30.0
    )
    assert result["leaps"]["stop_basis"] == pytest.approx(30.0)
    assert result["stop_loss"]["stop_price"] == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# detect_orphan_orders
# ---------------------------------------------------------------------------


def _sl_order(order_ref, symbol="NVDA", order_id=1):
    return {
        "order_ref": order_ref,
        "symbol": symbol,
        "order_id": order_id,
        "conditions": [{"price": 20.0, "is_more": False}],
    }


def test_detect_orphan_no_orphans():
    positions = [_pmcc_pos()]  # NVDA 200.0 20270115
    orders = [_sl_order("SL_FALL_NVDA_200.0_20270115")]
    assert detect_orphan_orders(orders, positions) == []


def test_detect_orphan_detects_closed_position():
    positions = []
    orders = [_sl_order("SL_FALL_NVDA_200.0_20270115")]
    orphans = detect_orphan_orders(orders, positions)
    assert len(orphans) == 1
    assert orphans[0]["order_ref"] == "SL_FALL_NVDA_200.0_20270115"


def test_detect_orphan_ignores_non_sl_orders():
    positions = []
    orders = [_sl_order("MANUAL_ORDER")]
    assert detect_orphan_orders(orders, positions) == []


def test_detect_orphan_stock_position():
    positions = [_stock_pos()]  # AAPL stock
    orders = [
        _sl_order("SL_FALL_AAPL_STK", symbol="AAPL"),
        _sl_order("SL_FALL_NVDA_200.0_20270115", symbol="NVDA"),  # orphan
    ]
    orphans = detect_orphan_orders(orders, positions)
    assert len(orphans) == 1
    assert orphans[0]["order_ref"] == "SL_FALL_NVDA_200.0_20270115"


# ---------------------------------------------------------------------------
# summarize_all_conditional_orders
# ---------------------------------------------------------------------------


def _cond_order(order_ref, conditions=None):
    return {
        "order_ref": order_ref,
        "conditions": conditions or [],
        "symbol": "NVDA",
        "order_id": 1,
        "action": "BUY",
        "qty": 1,
    }


def test_all_conditional_orders_splits_module_and_manual():
    orders = [
        _cond_order("SL_FALL_NVDA_200.0_20270115", [{"price": 22.0, "is_more": False}]),
        _cond_order("MANUAL_COND", [{"price": 200.0, "is_more": False}]),
    ]
    result = summarize_all_conditional_orders(orders)
    assert len(result["module"]) == 1
    assert len(result["manual"]) == 1


def test_all_conditional_orders_excludes_no_conditions():
    orders = [
        _cond_order("SL_FALL_NVDA_200.0_20270115", []),
        _cond_order("MANUAL_COND", []),
    ]
    result = summarize_all_conditional_orders(orders)
    assert result == {"module": [], "manual": []}


def test_all_conditional_orders_empty():
    assert summarize_all_conditional_orders([]) == {"module": [], "manual": []}
