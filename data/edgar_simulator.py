"""
SEC EDGAR API Simulator.

Returns realistic 10-K excerpts (risk factors, MD&A, financial statements)
for 10 fictional companies.  No real API key or network access required.

Companies:
  APEX    — Apex Capital Holdings (investment banking)
  BRKR    — Brookfield Regional Bank (community banking)
  CSTL    — Castle Wealth Management (asset management)
  DVRT    — Divert Financial Services (consumer lending)
  ENCR    — Encore Insurance Group (P&C insurance)
  FSTM    — Firstmark Digital Payments (fintech/payments)
  GLDN    — Golden Gate Bancorp (mortgage origination)
  HRBN    — Harbor Equity Partners (private equity)
  IVXP    — InvestX Platform (robo-advisory)
  JVLN    — Javelin Credit Union (credit union)

Usage:
    from data.edgar_simulator import EdgarSimulator

    sim = EdgarSimulator()
    filing = sim.get_filing("APEX", section="risk_factors")
    print(filing["text"])

    all_companies = sim.list_companies()
    all_filings = sim.get_all_filings()
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Company registry
# ---------------------------------------------------------------------------

COMPANY_REGISTRY: Dict[str, Dict[str, str]] = {
    "APEX": {
        "name": "Apex Capital Holdings, Inc.",
        "sector": "Investment Banking",
        "fiscal_year_end": "December 31, 2023",
        "total_assets": "$148.3 billion",
        "cik": "0001234567",
    },
    "BRKR": {
        "name": "Brookfield Regional Bank",
        "sector": "Community Banking",
        "fiscal_year_end": "December 31, 2023",
        "total_assets": "$8.7 billion",
        "cik": "0002345678",
    },
    "CSTL": {
        "name": "Castle Wealth Management Group",
        "sector": "Asset Management",
        "fiscal_year_end": "December 31, 2023",
        "total_assets": "$42.1 billion AUM",
        "cik": "0003456789",
    },
    "DVRT": {
        "name": "Divert Financial Services Corp.",
        "sector": "Consumer Lending",
        "fiscal_year_end": "December 31, 2023",
        "total_assets": "$19.4 billion",
        "cik": "0004567890",
    },
    "ENCR": {
        "name": "Encore Insurance Group",
        "sector": "Property & Casualty Insurance",
        "fiscal_year_end": "December 31, 2023",
        "total_assets": "$31.8 billion",
        "cik": "0005678901",
    },
    "FSTM": {
        "name": "Firstmark Digital Payments, Inc.",
        "sector": "Fintech / Payments",
        "fiscal_year_end": "December 31, 2023",
        "total_assets": "$5.2 billion",
        "cik": "0006789012",
    },
    "GLDN": {
        "name": "Golden Gate Bancorp",
        "sector": "Mortgage Banking",
        "fiscal_year_end": "December 31, 2023",
        "total_assets": "$24.6 billion",
        "cik": "0007890123",
    },
    "HRBN": {
        "name": "Harbor Equity Partners, L.P.",
        "sector": "Private Equity",
        "fiscal_year_end": "December 31, 2023",
        "total_assets": "$87.5 billion AUM",
        "cik": "0008901234",
    },
    "IVXP": {
        "name": "InvestX Platform Corp.",
        "sector": "Robo-Advisory / WealthTech",
        "fiscal_year_end": "December 31, 2023",
        "total_assets": "$2.1 billion AUA",
        "cik": "0009012345",
    },
    "JVLN": {
        "name": "Javelin Credit Union",
        "sector": "Credit Union",
        "fiscal_year_end": "December 31, 2023",
        "total_assets": "$6.3 billion",
        "cik": "0009123456",
    },
}

# ---------------------------------------------------------------------------
# Filing content templates (realistic 10-K excerpts)
# ---------------------------------------------------------------------------

_FILINGS: Dict[str, Dict[str, str]] = {
    # ------------------------------------------------------------------
    "APEX": {
        "risk_factors": (
            "RISK FACTORS — APEX CAPITAL HOLDINGS\n\n"
            "Market and Credit Risk. The Company's investment banking revenues are highly "
            "sensitive to global capital market conditions. A 10% decline in transaction "
            "volumes could reduce advisory fees by approximately $340 million. Credit "
            "spreads widening by 50 basis points across the leveraged-loan book would "
            "increase expected credit losses by an estimated $120 million.\n\n"
            "Regulatory Capital Requirements. Apex Capital is subject to Basel III capital "
            "requirements as administered by the Federal Reserve. The Company maintains a "
            "CET1 ratio of 14.2%, providing a 780-basis-point buffer above the 6.5% "
            "well-capitalized threshold. A severe stress scenario consistent with DFAST "
            "assumptions could reduce the CET1 ratio by up to 320 basis points.\n\n"
            "Liquidity Risk. The Company's liquidity coverage ratio (LCR) was 141% as of "
            "December 31, 2023. High-quality liquid assets (HQLA) totaled $18.4 billion "
            "against projected 30-day net outflows of $13.1 billion."
        ),
        "mda": (
            "MANAGEMENT'S DISCUSSION AND ANALYSIS — APEX CAPITAL HOLDINGS\n\n"
            "Financial Highlights. For fiscal year 2023, Apex Capital reported net revenue "
            "of $7.84 billion, a decrease of 8.3% from $8.55 billion in fiscal 2022, "
            "reflecting lower M&A advisory volumes partially offset by record Fixed Income "
            "Markets revenue of $2.12 billion.\n\n"
            "Advisory Services generated $2.94 billion (37.5% of total), compared to "
            "$3.68 billion in 2022. The decline reflects a 22% reduction in completed "
            "M&A transaction volume to $312 billion, consistent with the broader market "
            "slowdown in leveraged buyout activity.\n\n"
            "Equity Markets contributed $1.78 billion, up 4.2% year-over-year, driven by "
            "a 31% increase in equity underwriting and strong prime brokerage net revenues. "
            "Fixed Income, Currencies and Commodities (FICC) revenue of $2.12 billion "
            "reflected strong client activity in rates and credit."
        ),
        "financial_statements": (
            "CONSOLIDATED STATEMENTS OF INCOME — APEX CAPITAL HOLDINGS\n"
            "Year Ended December 31, 2023 (in millions)\n\n"
            "Net revenues:                                    $ 7,842\n"
            "Compensation and benefits:                       $ 3,921\n"
            "Non-compensation operating expenses:             $ 1,568\n"
            "Pre-tax income:                                  $ 2,353\n"
            "Income tax expense:                              $   588\n"
            "Net income:                                      $ 1,765\n\n"
            "Earnings per diluted share:                      $  8.43\n"
            "Return on equity:                                  12.4%\n"
            "Book value per share:                            $ 68.91\n\n"
            "CONSOLIDATED BALANCE SHEET HIGHLIGHTS\n"
            "Total assets:                                    $148,320 million\n"
            "Total equity:                                    $ 14,230 million\n"
            "Tier 1 capital:                                  $ 12,840 million\n"
            "Risk-weighted assets:                            $ 90,420 million\n"
            "CET1 ratio:                                         14.2%\n"
            "Leverage ratio:                                      8.3%"
        ),
    },

    # ------------------------------------------------------------------
    "BRKR": {
        "risk_factors": (
            "RISK FACTORS — BROOKFIELD REGIONAL BANK\n\n"
            "Interest Rate Risk. As a community bank, Brookfield's net interest income "
            "is sensitive to changes in interest rates. A parallel shift of +200 basis "
            "points in market rates would increase net interest income by an estimated "
            "$34.2 million over a 12-month horizon, while a -100 bps shift would reduce "
            "NII by $18.7 million.\n\n"
            "Concentration Risk. Commercial real estate loans represent 312% of risk-based "
            "capital, above the 300% informal supervisory guidance. The Bank has implemented "
            "enhanced monitoring protocols for the $2.1 billion CRE portfolio.\n\n"
            "Cybersecurity. The Bank invested $12.4 million in cybersecurity in fiscal 2023. "
            "The Bank experienced zero material security breaches during fiscal 2023."
        ),
        "mda": (
            "MANAGEMENT'S DISCUSSION AND ANALYSIS — BROOKFIELD REGIONAL BANK\n\n"
            "Financial Overview. Net interest income increased 9.8% to $312.4 million for "
            "fiscal 2023, driven by a 28-basis-point expansion in net interest margin to "
            "3.82%. Loans held for investment grew 6.3% to $6.84 billion, while total "
            "deposits reached $7.21 billion.\n\n"
            "Net Interest Margin. The net interest margin of 3.82% compares favorably to "
            "the community bank peer median of 3.54%. Loan yields expanded 48 basis points "
            "to 5.14%, partially offset by a 26-basis-point increase in deposit costs to "
            "1.42% as the cumulative deposit beta reached 38% by year-end.\n\n"
            "Credit Quality. Non-performing loans declined to 0.48% of total loans from "
            "0.61% at year-end 2022. Net charge-offs were 0.18%, well below the peer "
            "median of 0.38%. The allowance for credit losses was $82.4 million, or "
            "1.21% of loans, providing 2.5x coverage of NPLs."
        ),
        "financial_statements": (
            "CONSOLIDATED STATEMENTS OF INCOME — BROOKFIELD REGIONAL BANK\n"
            "Year Ended December 31, 2023 (in millions)\n\n"
            "Net interest income:                             $  312.4\n"
            "Non-interest income:                             $   48.6\n"
            "Total net revenue:                               $  361.0\n"
            "Provision for credit losses:                     $   14.2\n"
            "Non-interest expense:                            $  218.4\n"
            "Pre-tax income:                                  $  128.4\n"
            "Net income:                                      $   96.3\n\n"
            "Earnings per diluted share:                      $   4.82\n"
            "Return on assets:                                   1.11%\n"
            "Return on equity:                                  13.8%\n"
            "Net interest margin:                                3.82%\n"
            "Efficiency ratio:                                  60.5%"
        ),
    },

    # ------------------------------------------------------------------
    "CSTL": {
        "risk_factors": (
            "RISK FACTORS — CASTLE WEALTH MANAGEMENT GROUP\n\n"
            "Market Risk. Assets under management decreased 4.2% to $42.1 billion during "
            "fiscal 2023, reflecting net client inflows of $1.84 billion offset by negative "
            "market returns of $3.62 billion. A sustained 20% market decline would reduce "
            "AUM-based management fees by approximately $67 million annually.\n\n"
            "Client Concentration. The 10 largest institutional clients represent 38.4% "
            "of total AUM. Loss of any two of these clients could materially impact revenues.\n\n"
            "Regulatory and Fiduciary Risk. As a registered investment adviser, Castle is "
            "subject to SEC examination. The most recent examination (2022) resulted in "
            "no material findings."
        ),
        "mda": (
            "MANAGEMENT'S DISCUSSION AND ANALYSIS — CASTLE WEALTH MANAGEMENT GROUP\n\n"
            "Revenue Overview. Total revenues of $312.6 million for fiscal 2023 declined "
            "3.8% from $325.0 million in 2022, reflecting lower average AUM of $43.8 billion "
            "versus $46.2 billion in 2022. Management fees of $261.4 million (83.6% of total) "
            "carry a weighted average fee rate of 0.60%, compressed by 3 basis points "
            "year-over-year due to institutional pricing pressure.\n\n"
            "Performance fees of $28.4 million were earned on $6.2 billion of performance-fee "
            "eligible AUM. The primary equity strategies outperformed benchmarks by 142 basis "
            "points on average during fiscal 2023."
        ),
        "financial_statements": (
            "CONSOLIDATED STATEMENTS OF INCOME — CASTLE WEALTH MANAGEMENT GROUP\n"
            "Year Ended December 31, 2023 (in millions)\n\n"
            "Management fees:                                 $  261.4\n"
            "Performance fees:                                $   28.4\n"
            "Distribution and other fees:                     $   22.8\n"
            "Total revenues:                                  $  312.6\n"
            "Operating expenses:                              $  198.4\n"
            "Pre-tax income:                                  $  114.2\n"
            "Net income:                                      $   85.7\n\n"
            "AUM at period-end (billions):                      $ 42.1\n"
            "Net flows (billions):                              $  1.84\n"
            "Fee rate (bps):                                       60.0\n"
            "Operating margin:                                   36.5%"
        ),
    },

    # ------------------------------------------------------------------
    "DVRT": {
        "risk_factors": (
            "RISK FACTORS — DIVERT FINANCIAL SERVICES CORP.\n\n"
            "Consumer Credit Risk. The personal loan portfolio of $14.2 billion is "
            "concentrated in unsecured installment loans to near-prime borrowers "
            "(FICO 620–720). Net charge-offs increased to 3.84% in fiscal 2023 from "
            "2.91% in 2022, reflecting normalization of consumer credit following the "
            "pandemic stimulus period. The allowance for loan losses is $681.6 million, "
            "or 4.80% of outstanding balances.\n\n"
            "Funding Risk. Divert relies on securitization markets for 62% of funding. "
            "Sustained disruption to ABS markets could impair origination capacity and "
            "require reliance on more expensive warehouse lines."
        ),
        "mda": (
            "MANAGEMENT'S DISCUSSION AND ANALYSIS — DIVERT FINANCIAL SERVICES CORP.\n\n"
            "Origination Volume. Personal loan originations increased 11.4% to $7.84 billion "
            "in fiscal 2023. The average loan balance is $8,420 with an average term of "
            "42 months and weighted average APR of 18.4%. The 30-day delinquency rate "
            "was 4.21% at December 31, 2023, compared to 3.68% at December 31, 2022.\n\n"
            "Revenue. Net interest income of $682.4 million reflects a net yield on "
            "earning assets of 12.8%, partially offset by cost of funds of 4.2% including "
            "securitization costs. Non-interest income of $94.2 million includes origination "
            "fees ($62.1M) and servicing income ($32.1M)."
        ),
        "financial_statements": (
            "CONSOLIDATED STATEMENTS OF INCOME — DIVERT FINANCIAL SERVICES CORP.\n"
            "Year Ended December 31, 2023 (in millions)\n\n"
            "Net interest income:                             $  682.4\n"
            "Non-interest income:                             $   94.2\n"
            "Total net revenue:                               $  776.6\n"
            "Provision for loan losses:                       $  542.8\n"
            "Operating expenses:                              $  284.2\n"
            "Pre-tax income (loss):                           $  (50.4)\n"
            "Net loss:                                        $  (38.3)\n\n"
            "Net charge-off rate:                                3.84%\n"
            "30-day delinquency rate:                            4.21%\n"
            "Allowance / loans ratio:                            4.80%\n"
            "Return on assets:                                  (0.20)%"
        ),
    },

    # ------------------------------------------------------------------
    "ENCR": {
        "risk_factors": (
            "RISK FACTORS — ENCORE INSURANCE GROUP\n\n"
            "Catastrophe Exposure. Encore's property reinsurance portfolio carries estimated "
            "probable maximum loss of $2.84 billion for a 1-in-100-year Atlantic hurricane event. "
            "The Company purchased $3.1 billion in catastrophe reinsurance protection with "
            "attachment points beginning at $400 million per event.\n\n"
            "Reserve Adequacy. P&C loss reserves of $14.2 billion represent management's "
            "best estimate of ultimate claims. Actuarial analysis indicates a range of "
            "$13.4 billion to $15.1 billion, with the booked reserve at the 55th percentile.\n\n"
            "Investment Portfolio Risk. The $24.8 billion investment portfolio is 94.2% "
            "investment-grade fixed income. A 100-basis-point rise in rates would reduce "
            "the portfolio's fair value by approximately $1.84 billion."
        ),
        "mda": (
            "MANAGEMENT'S DISCUSSION AND ANALYSIS — ENCORE INSURANCE GROUP\n\n"
            "Underwriting Results. The combined ratio was 94.8% for fiscal 2023, including "
            "8.4 points of catastrophe losses (3.2 points from Hurricane Elara; 2.8 points "
            "from Texas hailstorms; 2.4 points from California wildfires). The ex-cat "
            "combined ratio improved to 86.4% from 88.2% in 2022.\n\n"
            "Net written premiums grew 9.4% to $8.42 billion, driven by rate increases "
            "averaging 11.2% in Personal Lines and 8.4% in Commercial Lines. The expense "
            "ratio of 28.4% reflects continued investment in technology and claims automation."
        ),
        "financial_statements": (
            "CONSOLIDATED STATEMENTS OF INCOME — ENCORE INSURANCE GROUP\n"
            "Year Ended December 31, 2023 (in millions)\n\n"
            "Net written premiums:                            $ 8,421\n"
            "Net earned premiums:                             $ 8,184\n"
            "Net investment income:                           $   842\n"
            "Net realized gains (losses):                     $   (84)\n"
            "Total revenues:                                  $ 8,942\n"
            "Losses and loss adjustment expenses:             $ 5,431\n"
            "Underwriting expenses:                           $ 2,322\n"
            "Pre-tax income:                                  $ 1,189\n"
            "Net income:                                      $   941\n\n"
            "Combined ratio:                                    94.8%\n"
            "Return on equity:                                  14.2%\n"
            "Book value per share:                            $ 84.20"
        ),
    },

    # ------------------------------------------------------------------
    "FSTM": {
        "risk_factors": (
            "RISK FACTORS — FIRSTMARK DIGITAL PAYMENTS, INC.\n\n"
            "Payment Processing Risk. Firstmark processes $284 billion in annual payment "
            "volume. A major platform outage lasting 24 hours could result in estimated "
            "revenue impact of $12–18 million and potential regulatory penalties. "
            "The Company maintains 99.97% platform availability.\n\n"
            "Fraud and Chargeback Exposure. Fraudulent transaction losses were $84.2 million "
            "(0.030% of processing volume) in fiscal 2023, compared to $72.4 million in 2022. "
            "AI-based fraud detection prevents an estimated $340 million in annual fraud.\n\n"
            "Regulatory Risk. Firstmark is subject to regulation as a money services business "
            "in 48 states and is pursuing a federal fintech charter."
        ),
        "mda": (
            "MANAGEMENT'S DISCUSSION AND ANALYSIS — FIRSTMARK DIGITAL PAYMENTS, INC.\n\n"
            "Business Overview. Firstmark processed $284 billion in total payment volume "
            "for fiscal 2023, a 24.6% increase from $228 billion in 2022. Net revenue of "
            "$1.84 billion reflects a net take rate of 0.65%, compressed 4 basis points "
            "year-over-year due to volume-based pricing discounts to large enterprise clients.\n\n"
            "Active merchant accounts grew 18.4% to 2.84 million. Consumer wallet accounts "
            "reached 42.1 million, with monthly active users of 24.6 million. "
            "Buy-Now-Pay-Later (BNPL) volume of $8.42 billion grew 84% year-over-year, "
            "representing 3.0% of total payment volume."
        ),
        "financial_statements": (
            "CONSOLIDATED STATEMENTS OF INCOME — FIRSTMARK DIGITAL PAYMENTS, INC.\n"
            "Year Ended December 31, 2023 (in millions)\n\n"
            "Transaction fees:                                $ 1,642\n"
            "Subscription and platform fees:                  $   142\n"
            "Interest and finance income:                     $    56\n"
            "Total net revenue:                               $ 1,840\n"
            "Transaction-related costs:                       $   689\n"
            "Technology and development:                      $   284\n"
            "Sales and marketing:                             $   312\n"
            "General and administrative:                      $   184\n"
            "Pre-tax income:                                  $   371\n"
            "Net income:                                      $   285\n\n"
            "Total payment volume ($B):                         $284.0\n"
            "Net take rate:                                      0.65%\n"
            "EBITDA margin:                                     28.4%"
        ),
    },

    # ------------------------------------------------------------------
    "GLDN": {
        "risk_factors": (
            "RISK FACTORS — GOLDEN GATE BANCORP\n\n"
            "Mortgage Market Risk. Origination volumes are highly sensitive to the "
            "30-year fixed mortgage rate. Volumes declined 42% to $14.2 billion in fiscal "
            "2023 as the 30-year rate peaked at 7.84%. A 100-basis-point decline in rates "
            "is estimated to increase annual origination volumes by $6–8 billion.\n\n"
            "Servicing Asset Valuation. The mortgage servicing rights (MSR) asset is "
            "$842 million as of December 31, 2023. A 25-basis-point decline in rates would "
            "reduce the MSR fair value by approximately $121 million due to prepayment risk."
        ),
        "mda": (
            "MANAGEMENT'S DISCUSSION AND ANALYSIS — GOLDEN GATE BANCORP\n\n"
            "Origination. Mortgage loan originations of $14.2 billion in fiscal 2023 "
            "declined 42% from $24.4 billion in 2022 as rising rates curtailed refinance "
            "activity. Purchase volume of $10.8 billion (76% of total) partially offset "
            "refinance declines of $8.6 billion.\n\n"
            "Gain on Sale. Gain-on-sale margin compressed to 1.84% from 2.12% in 2022 "
            "due to market competition. Net gain on mortgage loan sales of $261.3 million "
            "declined from $517.3 million in 2022."
        ),
        "financial_statements": (
            "CONSOLIDATED STATEMENTS OF INCOME — GOLDEN GATE BANCORP\n"
            "Year Ended December 31, 2023 (in millions)\n\n"
            "Net interest income:                             $   421.4\n"
            "Gain on mortgage loan sales:                     $   261.3\n"
            "Servicing income (net):                          $    84.2\n"
            "Total net revenue:                               $   766.9\n"
            "Operating expenses:                              $   612.4\n"
            "Pre-tax income:                                  $   154.5\n"
            "Net income:                                      $   116.2\n\n"
            "Origination volume ($B):                           $ 14.2\n"
            "Gain-on-sale margin:                                1.84%\n"
            "MSR fair value ($M):                               $ 842.0\n"
            "Return on equity:                                   9.4%"
        ),
    },

    # ------------------------------------------------------------------
    "HRBN": {
        "risk_factors": (
            "RISK FACTORS — HARBOR EQUITY PARTNERS, L.P.\n\n"
            "Valuation Risk. The fair value of portfolio investments is determined using "
            "significant unobservable inputs (Level 3). A 10% decline in implied EV/EBITDA "
            "multiples would reduce total AUM fair value by approximately $8.75 billion "
            "and management fees by $43.8 million annually.\n\n"
            "Realization Risk. Distributions to fund investors depend on successful exits "
            "via IPO, secondary sale, or strategic acquisition. The current portfolio has "
            "an average hold period of 5.2 years. IPO market conditions deteriorated in "
            "fiscal 2023, limiting exit opportunities."
        ),
        "mda": (
            "MANAGEMENT'S DISCUSSION AND ANALYSIS — HARBOR EQUITY PARTNERS, L.P.\n\n"
            "Fund Performance. Fund VI achieved a gross IRR of 18.4% and 2.1x MOIC since "
            "inception. Fund VII (vintage 2020) is tracking at a gross IRR of 24.2% and "
            "1.6x MOIC on invested capital of $12.4 billion.\n\n"
            "Fee-Earning AUM grew 8.4% to $62.4 billion, generating management fees of "
            "$624 million (1.0% average rate). Carried interest distributions of $284 million "
            "were recognized in fiscal 2023 upon realization of Fund V portfolio companies "
            "at a 28.4% gross IRR."
        ),
        "financial_statements": (
            "CONSOLIDATED STATEMENTS OF INCOME — HARBOR EQUITY PARTNERS, L.P.\n"
            "Year Ended December 31, 2023 (in millions)\n\n"
            "Management fees:                                 $   624.0\n"
            "Carried interest:                                $   284.0\n"
            "Investment income:                               $   142.0\n"
            "Total revenues:                                  $ 1,050.0\n"
            "Operating expenses:                              $   312.0\n"
            "Pre-tax income:                                  $   738.0\n"
            "Net income (attributable to GP):                 $   184.5\n\n"
            "Fee-earning AUM ($B):                              $  62.4\n"
            "Total AUM ($B):                                    $  87.5\n"
            "Management fee rate:                                 1.00%\n"
            "Distribution yield:                                  6.84%"
        ),
    },

    # ------------------------------------------------------------------
    "IVXP": {
        "risk_factors": (
            "RISK FACTORS — INVESTX PLATFORM CORP.\n\n"
            "Revenue Concentration. InvestX generates 94.2% of revenue from AUA-based "
            "advisory fees at a rate of 0.25% annually. Total AUA of $2.1 billion is "
            "managed across 184,200 active accounts with an average balance of $11,400. "
            "A 20% market decline would reduce AUA to $1.68 billion and annualized revenue "
            "by approximately $1.05 million.\n\n"
            "Technology Risk. The platform relies on third-party custodians and API "
            "integrations. A custodian outage lasting more than 4 hours would prevent "
            "trade execution and rebalancing for all client accounts."
        ),
        "mda": (
            "MANAGEMENT'S DISCUSSION AND ANALYSIS — INVESTX PLATFORM CORP.\n\n"
            "Growth Metrics. Active accounts grew 42.4% to 184,200 at December 31, 2023. "
            "Average account balance of $11,400 declined from $13,200 in 2022 due to "
            "market depreciation. Net new accounts of 54,600 in fiscal 2023 reflected "
            "strong organic growth through social media and employer benefit partnerships.\n\n"
            "Unit Economics. Customer acquisition cost declined to $84 from $112 in 2022 "
            "as referral and viral channels scaled. Lifetime value of $320 per account "
            "implies an LTV/CAC ratio of 3.8x."
        ),
        "financial_statements": (
            "CONSOLIDATED STATEMENTS OF INCOME — INVESTX PLATFORM CORP.\n"
            "Year Ended December 31, 2023 (in millions)\n\n"
            "Advisory fees (0.25% of AUA):                   $     5.2\n"
            "Premium subscription revenue:                   $     2.8\n"
            "Other revenue:                                   $     0.4\n"
            "Total revenue:                                   $     8.4\n"
            "Technology and infrastructure:                   $     3.2\n"
            "Sales and marketing:                             $     4.6\n"
            "General and administrative:                      $     2.1\n"
            "Pre-tax loss:                                    $    (1.5)\n"
            "Net loss:                                        $    (1.2)\n\n"
            "AUA at period-end ($B):                          $     2.1\n"
            "Active accounts (thousands):                         184.2\n"
            "Monthly active users (thousands):                    124.8\n"
            "ARPU (annualized):                               $    45.6"
        ),
    },

    # ------------------------------------------------------------------
    "JVLN": {
        "risk_factors": (
            "RISK FACTORS — JAVELIN CREDIT UNION\n\n"
            "Interest Rate Risk. As a non-profit cooperative, Javelin Credit Union is "
            "exempt from federal income tax. Net interest income of $184.2 million for "
            "fiscal 2023 is sensitive to rate changes. A +200 basis-point parallel shift "
            "would increase NII by $14.8 million while a -100 bps shift would reduce NII "
            "by $8.4 million.\n\n"
            "Concentration in Auto Lending. Vehicle loans of $2.84 billion represent 45.1% "
            "of total loans. Used vehicle values declined 12.4% in fiscal 2023, increasing "
            "LTV ratios on the existing portfolio. Net charge-offs on auto loans were "
            "0.42%, above the NCUA peer median of 0.31%."
        ),
        "mda": (
            "MANAGEMENT'S DISCUSSION AND ANALYSIS — JAVELIN CREDIT UNION\n\n"
            "Membership and Growth. Membership grew 5.4% to 324,600 members at December 31, "
            "2023. Total assets of $6.3 billion reflect loan growth of 8.2% to $6.3 billion "
            "and share (deposit) growth of 4.1% to $5.42 billion.\n\n"
            "Capital Position. Net worth ratio of 12.4% exceeds the 7.0% well-capitalized "
            "threshold established by NCUA regulations. The credit union targets a net worth "
            "ratio above 10.0% to absorb potential stress losses."
        ),
        "financial_statements": (
            "STATEMENTS OF INCOME — JAVELIN CREDIT UNION\n"
            "Year Ended December 31, 2023 (in millions)\n\n"
            "Interest income:                                 $   242.6\n"
            "Interest expense:                                $    58.4\n"
            "Net interest income:                             $   184.2\n"
            "Non-interest income:                             $    28.4\n"
            "Total operating income:                          $   212.6\n"
            "Non-interest expense:                            $   148.4\n"
            "Provision for loan losses:                       $    26.8\n"
            "Net income (surplus):                            $    37.4\n\n"
            "Net worth ratio:                                   12.4%\n"
            "Net interest margin:                                3.24%\n"
            "Return on assets:                                   0.59%\n"
            "Total loans ($B):                                  $ 6.30"
        ),
    },
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EdgarFiling:
    """A simulated SEC EDGAR 10-K filing excerpt."""
    ticker: str
    company_name: str
    sector: str
    fiscal_year_end: str
    section: str          # "risk_factors" | "mda" | "financial_statements"
    text: str
    cik: str
    doc_id: str
    retrieved_at: str


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class EdgarSimulator:
    """
    Simulates SEC EDGAR API responses for 10 fictional companies.

    Usage:
        sim = EdgarSimulator()

        # Get a specific section
        filing = sim.get_filing("APEX", section="risk_factors")

        # Get all sections for a company
        all_sections = sim.get_company_filings("BRKR")

        # Get all filings across all companies
        all_filings = sim.get_all_filings()

        # List available companies
        companies = sim.list_companies()
    """

    SECTIONS = ("risk_factors", "mda", "financial_statements")

    def get_filing(
        self,
        ticker: str,
        section: str = "mda",
    ) -> Optional[EdgarFiling]:
        """
        Retrieve a single 10-K section for a given ticker.

        Args:
            ticker: Company ticker symbol (e.g., "APEX").
            section: One of "risk_factors", "mda", "financial_statements".

        Returns:
            EdgarFiling or None if ticker/section not found.
        """
        ticker = ticker.upper()
        if ticker not in COMPANY_REGISTRY:
            return None
        if section not in self.SECTIONS:
            return None

        company = COMPANY_REGISTRY[ticker]
        text = _FILINGS.get(ticker, {}).get(section, "")

        # Stable doc_id derived from content hash
        doc_id = "EDGAR-" + hashlib.md5(f"{ticker}-{section}".encode()).hexdigest()[:8].upper()

        return EdgarFiling(
            ticker=ticker,
            company_name=company["name"],
            sector=company["sector"],
            fiscal_year_end=company["fiscal_year_end"],
            section=section,
            text=text,
            cik=company["cik"],
            doc_id=doc_id,
            retrieved_at=datetime.utcnow().isoformat() + "Z",
        )

    def get_company_filings(self, ticker: str) -> List[EdgarFiling]:
        """Return all three sections for a given company."""
        filings = []
        for section in self.SECTIONS:
            filing = self.get_filing(ticker, section)
            if filing:
                filings.append(filing)
        return filings

    def get_all_filings(self) -> List[EdgarFiling]:
        """Return all 30 filings (10 companies × 3 sections)."""
        filings = []
        for ticker in sorted(COMPANY_REGISTRY.keys()):
            filings.extend(self.get_company_filings(ticker))
        return filings

    def list_companies(self) -> List[Dict[str, str]]:
        """Return a list of all registered companies."""
        return [
            {
                "ticker": ticker,
                "name": info["name"],
                "sector": info["sector"],
                "fiscal_year_end": info["fiscal_year_end"],
                "total_assets": info["total_assets"],
                "cik": info["cik"],
            }
            for ticker, info in sorted(COMPANY_REGISTRY.items())
        ]

    def search_filings(self, keywords: List[str]) -> List[EdgarFiling]:
        """
        Search all filings for keywords.

        Returns filings where at least one keyword appears in the text.
        """
        results = []
        kw_lower = [kw.lower() for kw in keywords]

        for filing in self.get_all_filings():
            text_lower = filing.text.lower()
            if any(kw in text_lower for kw in kw_lower):
                results.append(filing)

        return results

    def to_document_chunks(self, ticker: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Convert filings to document chunks compatible with the ingestion pipeline.

        Args:
            ticker: If provided, only return chunks for this company.

        Returns:
            List of dicts with keys: chunk_id, doc_name, doc_type, text.
        """
        filings = (
            self.get_company_filings(ticker.upper()) if ticker
            else self.get_all_filings()
        )

        chunks = []
        for filing in filings:
            chunk_id = f"{filing.ticker}_{filing.section}_0001"
            chunks.append({
                "chunk_id": chunk_id,
                "doc_name": f"{filing.ticker}_10K_2023",
                "doc_type": "10k",
                "text": filing.text,
                "metadata": {
                    "ticker": filing.ticker,
                    "company": filing.company_name,
                    "section": filing.section,
                    "cik": filing.cik,
                    "doc_id": filing.doc_id,
                },
            })

        return chunks


if __name__ == "__main__":
    sim = EdgarSimulator()

    print("=== Available Companies ===")
    for co in sim.list_companies():
        print(f"  {co['ticker']:<6} {co['name']:<40} {co['sector']}")

    print("\n=== Sample Filing: APEX risk_factors ===")
    filing = sim.get_filing("APEX", "risk_factors")
    print(filing.text[:400], "...")

    print("\n=== Keyword Search: 'LCR' ===")
    hits = sim.search_filings(["LCR", "liquidity coverage"])
    print(f"Found {len(hits)} filings mentioning LCR")
    for h in hits:
        print(f"  {h.ticker} — {h.section}")

    print("\n=== Document Chunks (JVLN) ===")
    chunks = sim.to_document_chunks("JVLN")
    print(f"  {len(chunks)} chunks for JVLN")
