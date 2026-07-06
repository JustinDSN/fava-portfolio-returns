"""Tests for transactions shared by several investments of one group.

Beangrow extracts and categorizes transactions once per investment, so a
transaction touching several selected investments used to be double counted:
its cash legs appeared once per investment, and each member's asset legs
leaked into the other members' flows as boundary-crossing transfers.
"""

import datetime
import unittest

import numpy as np
from beancount.core import prices
from beancount.core.amount import Amount
from beancount.core.number import D

from fava_portfolio_returns._vendor.beangrow.investments import CashFlow
from fava_portfolio_returns._vendor.beangrow.returns import Pricer
from fava_portfolio_returns._vendor.beangrow.returns import bracketed_irr
from fava_portfolio_returns._vendor.beangrow.returns import compute_irr
from fava_portfolio_returns.metrics.pnl import TotalPNL
from fava_portfolio_returns.test.test import BEANGROW_CONFIG_CORPAB
from fava_portfolio_returns.test.test import approx2
from fava_portfolio_returns.test.test import load_portfolio_str

LEDGER_HEADER = """
plugin "beancount.plugins.auto_accounts"
plugin "beancount.plugins.implicit_prices"

2020-01-01 commodity CORPA
2020-01-01 commodity CORPB
"""


class TestSharedTransactions(unittest.TestCase):
    def test_multi_bucket_purchase_counts_cash_once(self):
        p = load_portfolio_str(
            LEDGER_HEADER
            + """
2020-01-02 * "one statement line buys into two buckets"
  Assets:CORPA   5 CORPA {12 USD}
  Assets:CORPB   5 CORPB {12 USD}
  Assets:Cash  -120 USD
""",
            BEANGROW_CONFIG_CORPAB,
            investment_filter=["g_CORP"],
        )
        flows = p.cash_flows()
        assert [(f.date, f.amount, f.source) for f in flows] == [
            (datetime.date(2020, 1, 2), Amount(D("-120"), "USD"), "cash"),
        ]

    def test_cross_lot_sale_counts_proceeds_once(self):
        p = load_portfolio_str(
            LEDGER_HEADER
            + """
2020-01-02 * "buy lot A"
  Assets:CORPA   10 CORPA {10 USD}
  Assets:Cash  -100 USD

2020-01-02 * "buy lot B"
  Assets:CORPB   10 CORPB {10 USD}
  Assets:Cash  -100 USD

2020-06-01 * "sell both lots in one order"
  Assets:CORPA  -10 CORPA {10 USD} @ 15 USD
  Assets:CORPB  -10 CORPB {10 USD} @ 15 USD
  Assets:Cash   300 USD
  Income:PnL   -100 USD
""",
            BEANGROW_CONFIG_CORPAB,
            investment_filter=["g_CORP"],
        )
        flows = p.cash_flows()
        assert [float(f.amount.number) for f in flows] == [-100.0, -100.0, 300.0]
        assert TotalPNL().single(p, datetime.date(2020, 1, 1), datetime.date(2020, 12, 31)) == approx2(100)

    def test_internal_exchange_produces_no_boundary_flows(self):
        p = load_portfolio_str(
            LEDGER_HEADER
            + """
2020-01-02 * "buy lot A"
  Assets:CORPA   10 CORPA {10 USD}
  Assets:Cash  -100 USD

2020-06-01 * "exchange A into B at market in one event"
  Assets:CORPA  -10 CORPA {10 USD}
  Assets:CORPB    6 CORPB {25 USD}
  Income:Gains  -50 USD

2020-12-31 price CORPB 25 USD
""",
            BEANGROW_CONFIG_CORPAB,
            investment_filter=["g_CORP"],
        )
        flows = p.cash_flows()
        assert [(f.date, f.amount, f.source) for f in flows] == [
            (datetime.date(2020, 1, 2), Amount(D("-100"), "USD"), "cash"),
        ]
        # the exchange gain stays inside the group: value 150, invested 100
        assert TotalPNL().single(p, datetime.date(2020, 1, 1), datetime.date(2020, 12, 31)) == approx2(50)


class TestBracketedIRR(unittest.TestCase):
    def test_bracketing_finds_known_root(self):
        cash_flows = np.array([-100.0, 150.0])
        years = np.array([-1.0, 0.0])
        assert bracketed_irr(cash_flows, years) == approx2(0.5)

    def test_vest_then_liquidate_pattern_is_not_a_total_loss(self):
        """Monthly vest-in/sell-out cycles: the solver drifts into the
        NPV collapse zone at irr -> -1 and the profitable pattern used to be
        reported as a -100% loss. Its only true root is at an implausible
        annualized rate, so the IRR is undefined and reported as 0."""
        end_date = datetime.date(2020, 7, 6)
        amounts = [
            (datetime.date(2020, 3, 16), "-100000"),
            (datetime.date(2020, 3, 16), "43000"),
            (datetime.date(2020, 3, 18), "61000"),
            (datetime.date(2020, 4, 16), "-33000"),
            (datetime.date(2020, 4, 16), "16500"),
            (datetime.date(2020, 4, 16), "17800"),
            (datetime.date(2020, 5, 16), "-32500"),
            (datetime.date(2020, 5, 16), "15700"),
            (datetime.date(2020, 5, 16), "16700"),
            (datetime.date(2020, 6, 16), "-31600"),
            (datetime.date(2020, 6, 16), "15000"),
            (datetime.date(2020, 6, 16), "16000"),
        ]
        flows = [
            CashFlow(date, Amount(D(number), "USD"), False, "cash", "Assets:CORPA", None)
            for date, number in amounts
        ]
        pricer = Pricer(prices.build_price_map([]))

        assert compute_irr(flows, pricer, "USD", end_date) == 0.0

    def test_total_loss_still_reports_minus_one(self):
        end_date = datetime.date(2021, 1, 1)
        flows = [
            CashFlow(datetime.date(2020, 1, 1), Amount(D("-1000"), "USD"), False, "cash", "Assets:CORPA", None),
            CashFlow(datetime.date(2020, 6, 1), Amount(D("-500"), "USD"), False, "cash", "Assets:CORPA", None),
        ]
        pricer = Pricer(prices.build_price_map([]))

        assert compute_irr(flows, pricer, "USD", end_date) == -1
