"""
Comprehensive Market Universe - Self-Contained Data Source Configuration

This module provides a hard-coded, market-spanning universe of financial
instruments across all major asset classes. Each symbol is paired with its
exchange, currency, and data-source identifier for seamless pipeline
integration without further configuration.
"""

import logging
from typing import NamedTuple

import pandas as pd
import ray
import yfinance as yf

logger = logging.getLogger(__name__)


@ray.remote
def fetch_data(
    symbol: str, start_date: str, end_date: str, interval: str = '1d'
) -> pd.DataFrame:
    """Fetches OHLCV data for a single symbol."""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(
            start=start_date,
            end=end_date,
            interval=interval
        )
        if not data.empty:
            data['Symbol'] = symbol
            return data.reset_index()
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()


def get_sp500_symbols() -> list[str]:
    """Get S&P 500 stock symbols from the MarketUniverse.

    Returns:
        List of S&P 500 stock symbols
    """
    # Get all equity instruments from the market universe
    equities = MarketUniverse.get_equities_universe()

    # Extract symbols (first element of each tuple)
    sp500_symbols = [equity.symbol for equity in equities]

    logger.info(f"Retrieved {len(sp500_symbols)} S&P 500 symbols")
    return sp500_symbols


class Instrument(NamedTuple):
    """Represents a financial instrument with complete metadata."""
    symbol: str
    name: str
    currency: str
    data_source: str
    asset_class: str
    exchange: str = "Global"
    sector: str = ""


class MarketUniverse:
    """Comprehensive market universe spanning all major asset classes."""

    @staticmethod
    def get_equities_universe() -> list[Instrument]:
        """40 large and mid-cap equities (20 large + 20 mid) from each GICS sector."""
        equities = {
            "Information Technology": [
                ("AAPL", "Apple Inc.", "NASDAQ", "USD", "yahoo", "equity"),
                ("MSFT", "Microsoft Corp.", "NASDAQ",
                 "USD", "yahoo", "equity", "Information Technology"),
                ("GOOGL", "Alphabet Inc. Class A", "NASDAQ",
                 "USD", "yahoo", "equity", "Information Technology"),
                ("NVDA", "NVIDIA Corporation", "NASDAQ",
                 "USD", "yahoo", "equity", "Information Technology"),
                ("META", "Meta Platforms Inc.", "NASDAQ",
                 "USD", "yahoo", "equity", "Information Technology"),
                ("AVGO", "Broadcom Inc.", "NASDAQ",
                 "USD", "yahoo", "equity", "Information Technology"),
                ("ORCL", "Oracle Corporation", "NYSE",
                 "USD", "yahoo", "equity", "Information Technology"),
                ("CRM", "Salesforce Inc.", "NYSE",
                 "USD", "yahoo", "equity", "Information Technology"),
                ("ADBE", "Adobe Inc.", "NASDAQ",
                 "USD", "yahoo", "equity", "Information Technology"),
                ("INTC", "Intel Corporation", "NASDAQ",
                 "USD", "yahoo", "equity", "Information Technology"),
                ("CSCO", "Cisco Systems Inc.", "NASDAQ",
                 "USD", "yahoo", "equity", "Information Technology"),
                ("ACN", "Accenture plc", "NYSE",
                 "USD", "yahoo", "equity", "Information Technology"),
                ("TXN", "Texas Instruments Inc.", "NASDAQ",
                 "USD", "yahoo", "equity", "Information Technology"),
                ("QCOM", "Qualcomm Inc.", "NASDAQ",
                 "USD", "yahoo", "equity", "Information Technology"),
                ("AMAT", "Applied Materials Inc.", "NASDAQ",
                 "USD", "yahoo", "equity", "Information Technology"),
                ("AMD", "Advanced Micro Devices", "NASDAQ",
                 "USD", "yahoo", "equity", "Information Technology"),
                ("NOW", "ServiceNow Inc.", "NYSE",
                 "USD", "yahoo", "equity", "Information Technology"),
                ("IBM", "International Business", "NYSE",
                 "USD", "yahoo", "equity", "Information Technology"),
                ("MU", "Micron Technology Inc.", "NASDAQ",
                 "USD", "yahoo", "equity", "Information Technology"),
                ("LRCX", "Lam Research Corp.", "NASDAQ",
                 "USD", "yahoo", "equity", "Information Technology"),
                ("ADI", "Analog Devices Inc.", "NASDAQ",
                 "USD", "yahoo", "equity", "Information Technology"),
                ("QCOM", "Qualcomm Inc.", "NASDAQ",
                 "USD", "yahoo", "equity", "Information Technology"),
                ("TXN", "Texas Instruments Inc.", "NASDAQ",
                 "USD", "yahoo", "equity", "Information Technology"),
                ("CDNS", "Cadence Design Systems", "NASDAQ",
                 "USD", "yahoo", "equity", "Information Technology")
               ],
            "Healthcare": [
                ("UNH", "UnitedHealth Group Inc.", "NYSE",
                 "USD", "yahoo", "equity", "Healthcare"),
                ("JNJ", "Johnson & Johnson", "NYSE",
                 "USD", "yahoo", "equity", "Healthcare"),
                ("PFE", "Pfizer Inc.", "NYSE",
                 "USD", "yahoo", "equity", "Healthcare"),
                ("ABBV", "AbbVie Inc.", "NYSE",
                 "USD", "yahoo", "equity", "Healthcare"),
                ("TMO", "Thermo Fisher Sci. Inc.",
                 "NYSE", "USD", "yahoo", "equity", "Healthcare"),
                ("ABT", "Abbott Laboratories", "NYSE",
                 "USD", "yahoo", "equity", "Healthcare"),
                ("LLY", "Eli Lilly and Company", "NYSE",
                 "USD", "yahoo", "equity", "Healthcare"),
                ("MRK", "Merck & Co. Inc.", "NYSE",
                 "USD", "yahoo", "equity", "Healthcare"),
                ("DHR", "Danaher Corporation", "NYSE",
                 "USD", "yahoo", "equity", "Healthcare"),
                ("BMY", "Bristol-Myers Squibb", "NYSE",
                 "USD", "yahoo", "equity", "Healthcare"),
                ("CVS", "CVS Health Corp.", "NYSE",
                 "USD", "yahoo", "equity", "Healthcare"),
                ("MDT", "Medtronic plc", "NYSE",
                 "USD", "yahoo", "equity", "Healthcare"),
                ("AMGN", "Amgen Inc.", "NASDAQ",
                 "USD", "yahoo", "equity", "Healthcare"),
                ("ISRG", "Intuitive Surgical Inc.",
                 "NASDAQ", "USD", "yahoo", "equity", "Healthcare"),
                ("VRTX", "Vertex Pharmaceuticals", "NASDAQ",
                 "USD", "yahoo", "equity", "Healthcare"),
                ("GILD", "Gilead Sciences Inc.", "NASDAQ",
                 "USD", "yahoo", "equity", "Healthcare"),
                ("CI", "Cigna Group", "NYSE",
                 "USD", "yahoo", "equity", "Healthcare"),
                ("ELV", "Elevance Health Inc.", "NYSE",
                 "USD", "yahoo", "equity", "Healthcare"),
                ("REGN", "Regeneron Pharma. Inc.",
                 "NASDAQ", "USD", "yahoo", "equity", "Healthcare"),
                ("BIIB", "Biogen Inc.", "NASDAQ",
                 "USD", "yahoo", "equity", "Healthcare")
            ],
            "Financials": [
                ("BRK-B", "Berkshire Hathaway Cl B", "NYSE",
                 "USD", "yahoo", "equity", "Financials"),
                ("JPM", "JPMorgan Chase & Co.", "NYSE",
                 "USD", "yahoo", "equity", "Financials"),
                ("V", "Visa Inc.", "NYSE",
                 "USD", "yahoo", "equity", "Financials"),
                ("MA", "Mastercard Inc.", "NYSE",
                 "USD", "yahoo", "equity", "Financials"),
                ("BAC", "Bank of America Corp.", "NYSE",
                 "USD", "yahoo", "equity", "Financials"),
                ("WFC", "Wells Fargo & Co.", "NYSE",
                 "USD", "yahoo", "equity", "Financials"),
                ("GS", "Goldman Sachs Group Inc.", "NYSE",
                 "USD", "yahoo", "equity", "Financials"),
                ("MS", "Morgan Stanley", "NYSE",
                 "USD", "yahoo", "equity", "Financials"),
                ("AXP", "American Express Co.", "NYSE",
                 "USD", "yahoo", "equity", "Financials"),
                ("C", "Citigroup Inc.", "NYSE",
                 "USD", "yahoo", "equity", "Financials"),
                ("USB", "U.S. Bancorp", "NYSE",
                 "USD", "yahoo", "equity", "Financials"),
                ("TFC", "Truist Financial Corp.",
                 "NYSE", "USD", "yahoo", "equity", "Financials"),
                ("PNC", "PNC Financial Services", "NYSE",
                 "USD", "yahoo", "equity", "Financials"),
                ("COF", "Capital One Financial", "NYSE",
                 "USD", "yahoo", "equity", "Financials"),
                ("SCHW", "Charles Schwab Corp.", "NYSE",
                 "USD", "yahoo", "equity", "Financials"),
                ("CB", "Chubb Limited", "NYSE",
                 "USD", "yahoo", "equity", "Financials"),
                ("BLK", "BlackRock Inc.", "NYSE",
                 "USD", "yahoo", "equity", "Financials"),
                ("AON", "Aon plc", "NYSE",
                 "USD", "yahoo", "equity", "Financials"),
                ("SPGI", "S&P Global Inc.", "NYSE",
                 "USD", "yahoo", "equity", "Financials"),
                ("MCO", "Moody's Corp.", "NYSE",
                 "USD", "yahoo", "equity", "Financials")
            ],
            "Consumer Discretionary": [
                ("AMZN", "Amazon.com Inc.", "NASDAQ",
                 "USD", "yahoo", "equity", "Consumer Discretionary"),
                ("TSLA", "Tesla Inc.", "NASDAQ",
                 "USD", "yahoo", "equity", "Consumer Discretionary"),
                ("HD", "Home Depot Inc.", "NYSE",
                 "USD", "yahoo", "equity", "Consumer Discretionary"),
                ("MCD", "McDonald's Corp.", "NYSE",
                 "USD", "yahoo", "equity", "Consumer Discretionary"),
                ("NKE", "Nike Inc. Class B", "NYSE",
                 "USD", "yahoo", "equity", "Consumer Discretionary"),
                ("SBUX", "Starbucks Corp.", "NASDAQ",
                 "USD", "yahoo", "equity", "Consumer Discretionary"),
                ("LOW", "Lowe's Companies Inc.", "NYSE",
                 "USD", "yahoo", "equity", "Consumer Discretionary"),
                ("TJX", "TJX Companies Inc.", "NYSE",
                 "USD", "yahoo", "equity", "Consumer Discretionary"),
                ("BKNG", "Booking Holdings Inc.",
                 "NASDAQ", "USD", "yahoo", "equity", "Consumer Discretionary"),
                ("EL", "Estée Lauder Cos.", "NYSE",
                 "USD", "yahoo", "equity", "Consumer Discretionary"),
                ("GM", "General Motors Co.", "NYSE",
                 "USD", "yahoo", "equity", "Consumer Discretionary"),
                ("F", "Ford Motor Company", "NYSE",
                 "USD", "yahoo", "equity", "Consumer Discretionary"),
                ("MAR", "Marriott Intl. Inc.", "NASDAQ",
                 "USD", "yahoo", "equity", "Consumer Discretionary"),
                ("HLT", "Hilton Worldwide Inc.", "NYSE",
                 "USD", "yahoo", "equity", "Consumer Discretionary"),
                ("ABNB", "Airbnb Inc.", "NASDAQ",
                 "USD", "yahoo", "equity", "Consumer Discretionary"),
                ("RCL", "Royal Caribbean", "NYSE",
                 "USD", "yahoo", "equity", "Consumer Discretionary"),
                ("CCL", "Carnival Corp.", "NYSE",
                 "USD", "yahoo", "equity", "Consumer Discretionary"),
                ("NCLH", "Norwegian Cruise Line",
                 "NYSE", "USD", "yahoo", "equity", "Consumer Discretionary"),
                ("LULU", "Lululemon Athletica", "NASDAQ",
                 "USD", "yahoo", "equity", "Consumer Discretionary"),
                ("ORLY", "O'Reilly Automotive Inc.", "NASDAQ",
                 "USD", "yahoo", "equity", "Consumer Discretionary")
            ],
            "Communication Services": [
                ("GOOGL", "Alphabet Inc. Class A", "NASDAQ",
                 "USD", "yahoo", "equity", "Communication Services"),
                ("META", "Meta Platforms Inc.", "NASDAQ",
                 "USD", "yahoo", "equity", "Communication Services"),
                ("NFLX", "Netflix Inc.", "NASDAQ",
                 "USD", "yahoo", "equity", "Communication Services"),
                ("DIS", "Walt Disney Company", "NYSE",
                 "USD", "yahoo", "equity", "Communication Services"),
                ("CMCSA", "Comcast Corp.", "NASDAQ",
                 "USD", "yahoo", "equity", "Communication Services"),
                ("VZ", "Verizon Communications", "NYSE",
                 "USD", "yahoo", "equity", "Communication Services"),
                ("T", "AT&T Inc.", "NYSE",
                 "USD", "yahoo", "equity", "Communication Services"),
                ("CHTR", "Charter Communications",
                 "NASDAQ", "USD", "yahoo", "equity", "Communication Services"),
                ("TMUS", "T-Mobile US Inc.", "NASDAQ",
                 "USD", "yahoo", "equity", "Communication Services"),
                ("PARA", "Paramount Global", "NASDAQ",
                 "USD", "yahoo", "equity", "Communication Services"),
                ("WBD", "Warner Bros. Disc.", "NASDAQ",
                 "USD", "yahoo", "equity", "Communication Services"),
                ("DISH", "DISH Network Corp.", "NASDAQ",
                 "USD", "yahoo", "equity", "Communication Services"),
                ("FOXA", "Fox Corp. Class A", "NASDAQ",
                 "USD", "yahoo", "equity", "Communication Services"),
                ("FOX", "Fox Corp. Class B", "NASDAQ",
                 "USD", "yahoo", "equity", "Communication Services"),
                ("LUMN", "Lumen Technologies", "NYSE",
                 "USD", "yahoo", "equity", "Communication Services"),
                ("SIRI", "Sirius XM Holdings", "NASDAQ",
                 "USD", "yahoo", "equity", "Communication Services"),
                ("OMC", "Omnicom Group Inc.", "NYSE",
                 "USD", "yahoo", "equity", "Communication Services"),
                ("IPG", "Interpublic Group", "NYSE",
                 "USD", "yahoo", "equity", "Communication Services"),
                ("WPP", "WPP plc", "NYSE",
                 "USD", "yahoo", "equity", "Communication Services"),
                ("DASH", "DoorDash Inc.", "NYSE",
                 "USD", "yahoo", "equity", "Communication Services")
            ],
            "Industrials": [
                ("HON", "Honeywell Intl. Inc.", "NASDAQ",
                 "USD", "yahoo", "equity", "Industrials"),
                ("UPS", "United Parcel Svc.", "NYSE",
                 "USD", "yahoo", "equity", "Industrials"),
                ("CAT", "Caterpillar Inc.", "NYSE",
                 "USD", "yahoo", "equity", "Industrials"),
                ("RTX", "RTX Corporation", "NYSE",
                 "USD", "yahoo", "equity", "Industrials"),
                ("LMT", "Lockheed Martin Corp.", "NYSE",
                 "USD", "yahoo", "equity", "Industrials"),
                ("GE", "General Electric Co.", "NYSE",
                 "USD", "yahoo", "equity", "Industrials"),
                ("MMM", "3M Company", "NYSE",
                 "USD", "yahoo", "equity", "Industrials"),
                ("FDX", "FedEx Corp.", "NYSE",
                 "USD", "yahoo", "equity", "Industrials"),
                ("NOC", "Northrop Grumman", "NYSE",
                 "USD", "yahoo", "equity", "Industrials"),
                ("BA", "Boeing Company", "NYSE",
                 "USD", "yahoo", "equity", "Industrials"),
                ("DE", "Deere & Company", "NYSE",
                 "USD", "yahoo", "equity", "Industrials"),
                ("WM", "Waste Management Inc.",
                 "NYSE", "USD", "yahoo", "equity", "Industrials"),
                ("ETN", "Eaton Corporation plc", "NYSE",
                 "USD", "yahoo", "equity", "Industrials"),
                ("ITW", "Illinois Tool Works", "NYSE",
                 "USD", "yahoo", "equity", "Industrials"),
                ("GD", "General Dynamics", "NYSE",
                 "USD", "yahoo", "equity", "Industrials"),
                ("EMR", "Emerson Electric", "NYSE",
                 "USD", "yahoo", "equity", "Industrials"),
                ("PH", "Parker-Hannifin", "NYSE",
                 "USD", "yahoo", "equity", "Industrials"),
                ("ROK", "Rockwell Automation", "NYSE",
                 "USD", "yahoo", "equity", "Industrials"),
                ("CTAS", "Cintas Corporation", "NASDAQ",
                 "USD", "yahoo", "equity", "Industrials"),
                ("NSC", "Norfolk Southern", "NYSE",
                 "USD", "yahoo", "equity", "Industrials")
            ],
            "Consumer Staples": [
                ("PG", "Procter & Gamble Co.", "NYSE",
                 "USD", "yahoo", "equity", "Consumer Staples"),
                ("KO", "Coca-Cola Company", "NYSE",
                 "USD", "yahoo", "equity", "Consumer Staples"),
                ("PEP", "PepsiCo Inc.", "NASDAQ",
                 "USD", "yahoo", "equity", "Consumer Staples"),
                ("WMT", "Walmart Inc.", "NYSE",
                 "USD", "yahoo", "equity", "Consumer Staples"),
                ("COST", "Costco Wholesale Corp.", "NASDAQ",
                 "USD", "yahoo", "equity", "Consumer Staples"),
                ("CL", "Colgate-Palmolive", "NYSE",
                 "USD", "yahoo", "equity", "Consumer Staples"),
                ("KMB", "Kimberly-Clark Corp.", "NYSE",
                 "USD", "yahoo", "equity", "Consumer Staples"),
                ("GIS", "General Mills Inc.", "NYSE",
                 "USD", "yahoo", "equity", "Consumer Staples"),
                ("KHC", "Kraft Heinz Co.", "NASDAQ",
                 "USD", "yahoo", "equity", "Consumer Staples"),
                ("HSY", "Hershey Company", "NYSE",
                 "USD", "yahoo", "equity", "Consumer Staples"),
                ("MNST", "Monster Beverage", "NASDAQ",
                 "USD", "yahoo", "equity", "Consumer Staples"),
                ("TSN", "Tyson Foods Inc. Cl A",
                 "NYSE", "USD", "yahoo", "equity", "Consumer Staples"),
                ("K", "Kellanova", "NYSE",
                 "USD", "yahoo", "equity", "Consumer Staples"),
                ("CAG", "Conagra Brands Inc.", "NYSE",
                 "USD", "yahoo", "equity", "Consumer Staples"),
                ("CPB", "Campbell Soup Co.", "NYSE",
                 "USD", "yahoo", "equity", "Consumer Staples"),
                ("SJM", "J.M. Smucker", "NYSE",
                 "USD", "yahoo", "equity", "Consumer Staples"),
                ("MDLZ", "Mondelez Intl. Inc.", "NASDAQ",
                 "USD", "yahoo", "equity", "Consumer Staples"),
                ("STZ", "Constellation Brands", "NYSE",
                 "USD", "yahoo", "equity", "Consumer Staples"),
                ("TAP", "Molson Coors Bev.", "NYSE",
                 "USD", "yahoo", "equity", "Consumer Staples"),
                ("KR", "Kroger Company", "NYSE",
                 "USD", "yahoo", "equity", "Consumer Staples")
            ],
            "Energy": [
                ("XOM", "Exxon Mobil Corp.", "NYSE",
                 "USD", "yahoo", "equity", "Energy"),
                ("CVX", "Chevron Corp.", "NYSE",
                 "USD", "yahoo", "equity", "Energy"),
                ("COP", "ConocoPhillips", "NYSE",
                 "USD", "yahoo", "equity", "Energy"),
                ("EOG", "EOG Resources Inc.", "NYSE",
                 "USD", "yahoo", "equity", "Energy"),
                ("SLB", "Schlumberger N.V.", "NYSE",
                 "USD", "yahoo", "equity", "Energy"),
                ("PXD", "Pioneer Natural Res.", "NYSE",
                 "USD", "yahoo", "equity", "Energy"),
                ("KMI", "Kinder Morgan Inc.", "NYSE",
                 "USD", "yahoo", "equity", "Energy"),
                ("OXY", "Occidental Petroleum", "NYSE",
                 "USD", "yahoo", "equity", "Energy"),
                ("WMB", "Williams Companies", "NYSE",
                 "USD", "yahoo", "equity", "Energy"),
                ("MPC", "Marathon Petroleum",
                 "NYSE", "USD", "yahoo", "equity", "Energy"),
                ("VLO", "Valero Energy Corp.", "NYSE",
                 "USD", "yahoo", "equity", "Energy"),
                ("PSX", "Phillips 66", "NYSE",
                 "USD", "yahoo", "equity", "Energy"),
                ("HES", "Hess Corporation", "NYSE",
                 "USD", "yahoo", "equity", "Energy"),
                ("DVN", "Devon Energy Corp.", "NYSE",
                 "USD", "yahoo", "equity", "Energy"),
                ("FANG", "Diamondback Energy", "NASDAQ",
                 "USD", "yahoo", "equity", "Energy"),
                ("BKR", "Baker Hughes Co.", "NASDAQ",
                 "USD", "yahoo", "equity", "Energy"),
                ("HAL", "Halliburton Co.", "NYSE",
                 "USD", "yahoo", "equity", "Energy"),
                ("TRGP", "Targa Resources", "NYSE",
                 "USD", "yahoo", "equity", "Energy"),
                ("OKE", "ONEOK Inc.", "NYSE",
                 "USD", "yahoo", "equity", "Energy"),
                ("CTRA", "Coterra Energy Inc.", "NYSE",
                 "USD", "yahoo", "equity", "Energy")
            ],
            "Utilities": [
                ("NEE", "NextEra Energy Inc.", "NYSE",
                 "USD", "yahoo", "equity", "Utilities"),
                ("DUK", "Duke Energy Corp.", "NYSE",
                 "USD", "yahoo", "equity", "Utilities"),
                ("SO", "Southern Company", "NYSE",
                 "USD", "yahoo", "equity", "Utilities"),
                ("AEP", "American Electric Pwr.",
                 "NASDAQ", "USD", "yahoo", "equity", "Utilities"),
                ("XEL", "Xcel Energy Inc.", "NASDAQ",
                 "USD", "yahoo", "equity", "Utilities"),
                ("EXC", "Exelon Corp.", "NASDAQ",
                 "USD", "yahoo", "equity", "Utilities"),
                ("D", "Dominion Energy Inc.", "NYSE",
                 "USD", "yahoo", "equity", "Utilities"),
                ("PEG", "Public Service Ent. Gp", "NYSE",
                 "USD", "yahoo", "equity", "Utilities"),
                ("SRE", "Sempra Energy", "NYSE",
                 "USD", "yahoo", "equity", "Utilities"),
                ("ES", "Eversource Energy", "NYSE",
                 "USD", "yahoo", "equity", "Utilities"),
                ("AWK", "American Water Works", "NYSE",
                 "USD", "yahoo", "equity", "Utilities"),
                ("PPL", "PPL Corp.", "NYSE",
                 "USD", "yahoo", "equity", "Utilities"),
                ("ATO", "Atmos Energy Corp.", "NYSE",
                 "USD", "yahoo", "equity", "Utilities"),
                ("WEC", "WEC Energy Group", "NYSE",
                 "USD", "yahoo", "equity", "Utilities"),
                ("DTE", "DTE Energy Co.", "NYSE",
                 "USD", "yahoo", "equity", "Utilities"),
                ("ETR", "Entergy Corporation",
                 "NYSE", "USD", "yahoo", "equity", "Utilities"),
                ("FE", "FirstEnergy Corp.", "NYSE",
                 "USD", "yahoo", "equity", "Utilities"),
                ("EIX", "Edison Intnl.", "NYSE",
                 "USD", "yahoo", "equity", "Utilities"),
                ("AEE", "Ameren Corp.", "NYSE",
                 "USD", "yahoo", "equity", "Utilities"),
                ("CMS", "CMS Energy Corp.", "NYSE",
                 "USD", "yahoo", "equity", "Utilities")
            ],
            "Real Estate": [
                ("AMT", "American Tower Corp.", "NYSE",
                 "USD", "yahoo", "equity", "Real Estate"),
                ("PLD", "Prologis Inc.", "NYSE",
                 "USD", "yahoo", "equity", "Real Estate"),
                ("CCI", "Crown Castle Inc.", "NYSE",
                 "USD", "yahoo", "equity", "Real Estate"),
                ("EQIX", "Equinix Inc.", "NASDAQ",
                 "USD", "yahoo", "equity", "Real Estate"),
                ("PSA", "Public Storage", "NYSE",
                 "USD", "yahoo", "equity", "Real Estate"),
                ("SPG", "Simon Property Grp.", "NYSE",
                 "USD", "yahoo", "equity", "Real Estate"),
                ("DLR", "Digital Realty Trust",
                 "NYSE", "USD", "yahoo", "equity", "Real Estate"),
                ("O", "Realty Income Corp.", "NYSE",
                 "USD", "yahoo", "equity", "Real Estate"),
                ("WELL", "Welltower Inc.", "NYSE",
                 "USD", "yahoo", "equity", "Real Estate"),
                ("EQR", "Equity Residential", "NYSE",
                 "USD", "yahoo", "equity", "Real Estate"),
                ("AVB", "AvalonBay Cmties.", "NYSE",
                 "USD", "yahoo", "equity", "Real Estate"),
                ("ESS", "Essex Property Trust", "NYSE",
                 "USD", "yahoo", "equity", "Real Estate"),
                ("MAA", "Mid-America Apt Cmt.", "NYSE",
                 "USD", "yahoo", "equity", "Real Estate"),
                ("VTR", "Ventas Inc.", "NYSE",
                 "USD", "yahoo", "equity", "Real Estate"),
                ("ARE", "Alexandria Real Estate", "NYSE",
                 "USD", "yahoo", "equity", "Real Estate"),
                ("BXP", "Boston Properties", "NYSE",
                 "USD", "yahoo", "equity", "Real Estate"),
                ("HST", "Host Hotels & Resorts",
                 "NASDAQ", "USD", "yahoo", "equity", "Real Estate"),
                ("REG", "Regency Centers", "NASDAQ",
                 "USD", "yahoo", "equity", "Real Estate"),
                ("UDR", "UDR Inc.", "NYSE",
                 "USD", "yahoo", "equity", "Real Estate"),
                ("CPT", "Camden Property Trust", "NYSE",
                 "USD", "yahoo", "equity", "Real Estate")
            ],
            "Materials": [
                ("LIN", "Linde plc", "NASDAQ",
                 "USD", "yahoo", "equity", "Materials"),
                ("APD", "Air Products & Chems.", "NYSE",
                 "USD", "yahoo", "equity", "Materials"),
                ("SHW", "Sherwin-Williams", "NYSE",
                 "USD", "yahoo", "equity", "Materials"),
                ("ECL", "Ecolab Inc.", "NYSE",
                 "USD", "yahoo", "equity", "Materials"),
                ("FCX", "Freeport-McMoRan", "NYSE",
                 "USD", "yahoo", "equity", "Materials"),
                ("NUE", "Nucor Corp.", "NYSE",
                 "USD", "yahoo", "equity", "Materials"),
                ("DOW", "Dow Inc.", "NYSE",
                 "USD", "yahoo", "equity", "Materials"),
                ("NEM", "Newmont Corp.", "NYSE",
                 "USD", "yahoo", "equity", "Materials"),
                ("VMC", "Vulcan Materials", "NYSE",
                 "USD", "yahoo", "equity", "Materials"),
                ("MLM", "Martin Marietta", "NYSE",
                 "USD", "yahoo", "equity", "Materials"),
                ("ALB", "Albemarle Corp.", "NYSE",
                 "USD", "yahoo", "equity", "Materials"),
                ("FMC", "FMC Corp.", "NYSE",
                 "USD", "yahoo", "equity", "Materials"),
                ("CF", "CF Industries", "NYSE",
                 "USD", "yahoo", "equity", "Materials"),
                ("MOS", "Mosaic Company", "NYSE",
                 "USD", "yahoo", "equity", "Materials"),
                ("PPG", "PPG Industries", "NYSE",
                 "USD", "yahoo", "equity", "Materials"),
                ("RPM", "RPM International", "NYSE",
                 "USD", "yahoo", "equity", "Materials"),
                ("EMN", "Eastman Chemical", "NYSE",
                 "USD", "yahoo", "equity", "Materials"),
                ("IFF", "Intl. Flavors & Frags.",
                 "NYSE", "USD", "yahoo", "equity", "Materials"),
                ("CE", "Celanese Corp.", "NYSE",
                 "USD", "yahoo", "equity", "Materials"),
                ("LYB", "LyondellBasell", "NYSE",
                 "USD", "yahoo", "equity", "Materials")
            ]
        }

        instruments: list[Instrument] = []
        for sector, symbols in equities.items():
            for symbol_data in symbols:  # type: ignore[attr-defined]
                # Cast to tuple to help mypy understand it's iterable
                symbol_tuple: tuple[str, ...] = symbol_data  # type: ignore
                instrument = Instrument(
                    symbol=symbol_tuple[0],
                    name=symbol_tuple[1],
                    exchange=symbol_tuple[2],
                    currency=symbol_tuple[3],
                    data_source=symbol_tuple[4],
                    asset_class=symbol_tuple[5],
                    sector=sector
                )
                instruments.append(instrument)

        return instruments

    @staticmethod
    def get_etfs_universe() -> list[Instrument]:
        """30 broad-market and sector ETFs."""
        etfs = [
            ("SPY", "SPDR S&P 500 ETF", "AMEX",
             "USD", "yahoo", "etf", "US Equity"),
            ("IVV", "iShares Core S&P 500 ETF", "AMEX",
             "USD", "yahoo", "etf", "US Equity"),
            ("VOO", "Vanguard S&P 500 ETF", "AMEX",
             "USD", "yahoo", "etf", "US Equity"),
            ("QQQM", "Invesco NASDAQ 100 ETF", "NASDAQ",
             "USD", "yahoo", "etf", "US Equity"),
            ("QQQ", "Invesco QQQ Trust", "NASDAQ",
             "USD", "yahoo", "etf", "US Equity"),
            ("DIA", "SPDR Dow Jones Ind. ETF",
             "AMEX", "USD", "yahoo", "etf", "US Equity"),
            ("IWM", "iShares Russell 2000 ETF", "AMEX",
             "USD", "yahoo", "etf", "US Equity"),
            ("VTI", "Vanguard Total Stock", "AMEX",
             "USD", "yahoo", "etf", "US Equity"),
            ("VEA", "Vanguard FTSE Dev.", "AMEX",
             "USD", "yahoo", "etf", "Developed Mkts"),
            ("VWO", "Vanguard FTSE Emerg.", "AMEX",
             "USD", "yahoo", "etf", "Emerging Mkts"),

           # Sector ETFs
           ("XLE", "Energy Select Sector SPDR", "AMEX", "USD",
            "yahoo", "etf", "Energy"),
           ("XLF", "Financial Select Sector SPDR", "AMEX", "USD",
            "yahoo", "etf", "Financials"),
           ("XLV", "Health Care Select Sector SPDR", "AMEX", "USD",
            "yahoo", "etf", "Healthcare"),
           ("XLI", "Industrial Select Sector SPDR", "AMEX", "USD",
            "yahoo", "etf", "Industrials"),
           ("XLB", "Materials Select Sector SPDR", "AMEX", "USD",
            "yahoo", "etf", "Materials"),
           ("XLK", "Technology Select Sector SPDR", "AMEX", "USD",
            "yahoo", "etf", "Information Technology"),
           ("XLU", "Utilities Select Sector SPDR", "AMEX", "USD",
            "yahoo", "etf", "Utilities"),
           ("XLP", "Consumer Staples Select Sector SPDR", "AMEX", "USD",
            "yahoo", "etf", "Consumer Staples"),
           ("XLY", "Consumer Discretionary Select Sector SPDR", "AMEX", "USD",
            "yahoo", "etf", "Consumer Discretionary"),
           ("XLC", "Communication Services Select Sector SPDR", "AMEX", "USD",
            "yahoo", "etf", "Communication Services"),

           # International/Regional ETFs
           ("EWJ", "iShares MSCI Japan ETF", "AMEX", "USD",
            "yahoo", "etf", "Japan"),
           ("EWZ", "iShares MSCI Brazil ETF", "AMEX", "USD",
            "yahoo", "etf", "Brazil"),
           ("EEM", "iShares MSCI Emerging Markets ETF", "AMEX", "USD",
            "yahoo", "etf", "Emerging Markets"),
           ("FEZ", "SPDR EURO STOXX 50 ETF", "AMEX", "USD",
            "yahoo", "etf", "Europe"),
           ("FXI", "iShares China Large-Cap ETF", "AMEX", "USD",
            "yahoo", "etf", "China")
       ]

        instruments: list[Instrument] = [
            Instrument(
                symbol=symbol, name=name,
                currency=currency, data_source=data_source,
                asset_class=asset_class, sector=sector,
                exchange=exchange
            ) for symbol, name, exchange, currency,
            data_source, asset_class, sector in etfs
        ]

        return instruments

    @staticmethod
    def get_crypto_universe() -> list[Instrument]:
        """40 high-volume cryptocurrencies including altcoins and tokens."""
        cryptos = [
            ("BTC-USD", "Bitcoin", "Global", "USD", "yahoo", "crypto"),
            ("ETH-USD", "Ethereum", "Global", "USD", "yahoo", "crypto"),
            ("XRP-USD", "Ripple", "Global", "USD", "yahoo", "crypto"),
            ("ADA-USD", "Cardano", "Global", "USD", "yahoo", "crypto"),
            ("SOL-USD", "Solana", "Global", "USD", "yahoo", "crypto"),
            ("DOGE-USD", "Dogecoin", "Global", "USD", "yahoo", "crypto"),
            ("DOT-USD", "Polkadot", "Global", "USD", "yahoo", "crypto"),
            ("MATIC-USD", "Polygon", "Global", "USD", "yahoo", "crypto"),
            ("BNB-USD", "BNB", "Global", "USD", "yahoo", "crypto"),
            ("LTC-USD", "Litecoin", "Global", "USD", "yahoo", "crypto"),
            ("BCH-USD", "Bitcoin Cash", "Global", "USD", "yahoo", "crypto"),
            ("XLM-USD", "Stellar", "Global", "USD", "yahoo", "crypto"),
            ("TRX-USD", "TRON", "Global", "USD", "yahoo", "crypto"),
            ("LINK-USD", "Chainlink", "Global", "USD", "yahoo", "crypto"),
            ("ETC-USD", "Ethereum Classic", "Global",
             "USD", "yahoo", "crypto"),
            ("UNI-USD", "Uniswap", "Global", "USD", "yahoo", "crypto"),
            ("AVAX-USD", "Avalanche", "Global", "USD", "yahoo", "crypto"),
            ("ATOM-USD", "Cosmos", "Global", "USD", "yahoo", "crypto"),
            ("CRO-USD", "Cronos", "Global", "USD", "yahoo", "crypto"),
            ("ALGO-USD", "Algorand", "Global", "USD", "yahoo", "crypto"),
            # Additional crypto assets
            ("XMR-USD", "Monero", "Global", "USD", "yahoo", "crypto"),
            ("ZEC-USD", "Zcash", "Global", "USD", "yahoo", "crypto"),
            ("ETC-USD", "Ethereum Classic", "Global", "USD", "yahoo", "crypto"),
            ("XTZ-USD", "Tezos", "Global", "USD", "yahoo", "crypto"),
            ("NEAR-USD", "NEAR Protocol", "Global", "USD", "yahoo", "crypto"),
            ("FTM-USD", "Fantom", "Global", "USD", "yahoo", "crypto"),
            ("GRT-USD", "The Graph", "Global", "USD", "yahoo", "crypto"),
            ("SUSHI-USD", "SushiSwap", "Global", "USD", "yahoo", "crypto"),
            ("MKR-USD", "Maker", "Global", "USD", "yahoo", "crypto"),
            ("AAVE-USD", "Aave", "Global", "USD", "yahoo", "crypto"),
            ("COMP-USD", "Compound", "Global", "USD", "yahoo", "crypto"),
            ("YFI-USD", "yearn.finance", "Global", "USD", "yahoo", "crypto"),
            ("SNX-USD", "Synthetix", "Global", "USD", "yahoo", "crypto"),
            ("CRV-USD", "Curve DAO Token", "Global", "USD", "yahoo", "crypto"),
            ("KSM-USD", "Kusama", "Global", "USD", "yahoo", "crypto"),
            ("DYDX-USD", "dYdX", "Global", "USD", "yahoo", "crypto"),
            ("RUNE-USD", "THORChain", "Global", "USD", "yahoo", "crypto"),
            ("1INCH-USD", "1inch Network", "Global", "USD", "yahoo", "crypto"),
            ("OCEAN-USD", "Ocean Protocol", "Global", "USD", "yahoo", "crypto")
        ]

        instruments: list[Instrument] = [
            Instrument(
                symbol=symbol, name=name,
                currency=currency, data_source=data_source,
                asset_class=asset_class, exchange=exchange
            ) for symbol, name, exchange, currency,
            data_source, asset_class in cryptos
        ]

        return instruments

    @staticmethod
    def get_commodities_universe() -> list[Instrument]:
        """5 liquid commodity futures."""
        futures = [
            ("CL=F", "Crude Oil WTI", "CME",
             "USD", "yahoo", "future"),
            ("GC=F", "Gold Futures", "COMEX",
             "USD", "yahoo", "future"),
            ("SI=F", "Silver Futures", "COMEX",
             "USD", "yahoo", "future"),
            ("NG=F", "Natural Gas", "NYMEX",
             "USD", "yahoo", "future"),
            ("ZS=F", "Soybean Futures", "CBOT",
             "USD", "yahoo", "future")
        ]

        instruments: list[Instrument] = [
            Instrument(
                symbol=symbol, name=name, exchange=exchange,
                currency=currency, data_source=data_source,
                asset_class=asset_class
            ) for symbol, name, exchange, currency,
            data_source, asset_class in futures
        ]

        return instruments

    @staticmethod
    def get_reit_universe() -> list[Instrument]:
        """3 major REIT indices."""
        reits = [
            ("VNQ", "Vanguard Real Estate", "AMEX",
             "USD", "yahoo", "reit", "US REITs"),
            ("SCHH", "Schwab U.S. REIT ETF", "AMEX",
             "USD", "yahoo", "reit", "US REITs"),
            ("REET", "iShares Global REIT ETF",
             "AMEX", "USD", "yahoo", "reit", "Global REITs")
        ]

        instruments: list[Instrument] = [
            Instrument(
                symbol=symbol, name=name, exchange=exchange,
                currency=currency, data_source=data_source,
                asset_class=asset_class, sector=sector
            ) for symbol, name, exchange, currency,
            data_source, asset_class, sector in reits
        ]

        return instruments

    @staticmethod
    def get_volatility_universe() -> list[Instrument]:
        """2 volatility products."""
        volatility_products = [
            ("VXX", "iPath S&P 500 VIX Short-Term",
             "AMEX", "USD", "yahoo", "volatility"),
            ("BATS:VIX", "CBOE Volatility Index", "CBOE",
             "USD", "yahoo", "volatility")
        ]

        instruments: list[Instrument] = [
            Instrument(
                symbol=symbol, name=name, exchange=exchange,
                currency=currency, data_source=data_source,
                asset_class=asset_class
            ) for symbol, name, exchange, currency,
            data_source, asset_class in volatility_products
        ]

        return instruments

    @staticmethod
    def get_forex_universe() -> list[Instrument]:
        """25 forex pairs including emerging markets."""
        forex_pairs = [
            ("USDCNY=X", "USD/CNY", "FOREX", "CNY", "yahoo", "forex"),
            ("USDINR=X", "USD/INR", "FOREX", "INR", "yahoo", "forex"),
            ("USDBRL=X", "USD/BRL", "FOREX", "BRL", "yahoo", "forex"),
            # Major pairs
            ("USDCNY=X", "USD/CNY", "FOREX", "CNY", "yahoo", "forex"),
            ("USDINR=X", "USD/INR", "FOREX", "INR", "yahoo", "forex"),
            ("USDBRL=X", "USD/BRL", "FOREX", "BRL", "yahoo", "forex"),
            ("USDKRW=X", "USD/KRW", "FOREX", "KRW", "yahoo", "forex"),
            ("USDMXN=X", "USD/MXN", "FOREX", "MXN", "yahoo", "forex"),
            ("USDZAR=X", "USD/ZAR", "FOREX", "ZAR", "yahoo", "forex"),
            ("USDSGD=X", "USD/SGD", "FOREX", "SGD", "yahoo", "forex"),
            ("EURUSD=X", "Euro to US Dollar", "FOREX",
             "USD", "yahoo", "forex"),
            ("GBPUSD=X", "British Pound to US Dollar", "FOREX",
             "USD", "yahoo", "forex"),
            ("USDJPY=X", "USD/JPY", "FOREX", "JPY", "yahoo", "forex"),
            ("AUDUSD=X", "Australian Dollar to US Dollar", "FOREX",
             "USD", "yahoo", "forex"),
            ("USDCAD=X", "US Dollar to Canadian Dollar", "FOREX",
             "CAD", "yahoo", "forex"),
            ("USDCHF=X", "US Dollar to Swiss Franc", "FOREX",
             "CHF", "yahoo", "forex"),
            ("NZDUSD=X", "New Zealand Dollar to US Dollar", "FOREX",
             "USD", "yahoo", "forex"),
            ("EURGBP=X", "Euro to British Pound", "FOREX",
             "GBP", "yahoo", "forex"),
            ("EURJPY=X", "Euro to Japanese Yen", "FOREX",
             "JPY", "yahoo", "forex"),
            ("GBPJPY=X", "British Pound to Japanese Yen", "FOREX",
             "JPY", "yahoo", "forex")
        ]

        instruments: list[Instrument] = [
            Instrument(
                symbol=symbol, name=name, exchange=exchange,
                currency=currency, data_source=data_source,
                asset_class=asset_class
            ) for symbol, name, exchange, currency,
            data_source, asset_class in forex_pairs + [
                ("USDCNY=X", "US Dollar to Chinese Yuan", "FOREX", "CNY", "yahoo", "forex"),
                ("USDINR=X", "US Dollar to Indian Rupee", "FOREX", "INR", "yahoo", "forex"),
                ("USDBRL=X", "US Dollar to Brazilian Real", "FOREX", "BRL", "yahoo", "forex"),
                ("USDKRW=X", "US Dollar to South Korean Won", "FOREX", "KRW", "yahoo", "forex"),
                ("USDMXN=X", "US Dollar to Mexican Peso", "FOREX", "MXN", "yahoo", "forex"),
                ("USDZAR=X", "US Dollar to South African Rand", "FOREX", "ZAR", "yahoo", "forex"),
                ("USDSGD=X", "US Dollar to Singapore Dollar", "FOREX", "SGD", "yahoo", "forex"),
                ("USDHKD=X", "US Dollar to Hong Kong Dollar", "FOREX", "HKD", "yahoo", "forex"),
                ("USDDKK=X", "US Dollar to Danish Krone", "FOREX", "DKK", "yahoo", "forex"),
                ("USDSEK=X", "US Dollar to Swedish Krona", "FOREX", "SEK", "yahoo", "forex"),
                ("USDNOK=X", "US Dollar to Norwegian Krone", "FOREX", "NOK", "yahoo", "forex"),
                ("USDTRY=X", "US Dollar to Turkish Lira", "FOREX", "TRY", "yahoo", "forex"),
                ("USDPLN=X", "US Dollar to Polish Zloty", "FOREX", "PLN", "yahoo", "forex"),
                ("USDTHB=X", "US Dollar to Thai Baht", "FOREX", "THB", "yahoo", "forex"),
                ("USDIDR=X", "US Dollar to Indonesian Rupiah", "FOREX", "IDR", "yahoo", "forex")
            ]
        ]

        return instruments

    @staticmethod
    def get_all_instruments() -> list[Instrument]:
        """Combines all instrument universes."""
        all_instruments: list[Instrument] = []
        all_instruments.extend(MarketUniverse.get_equities_universe())
        all_instruments.extend(MarketUniverse.get_etfs_universe())
        all_instruments.extend(MarketUniverse.get_crypto_universe())
        all_instruments.extend(MarketUniverse.get_commodities_universe())
        all_instruments.extend(MarketUniverse.get_reit_universe())
        all_instruments.extend(MarketUniverse.get_volatility_universe())
        all_instruments.extend(MarketUniverse.get_forex_universe())

        logger.info(
            f"Total instruments in market universe: {len(all_instruments)}"
        )

        return all_instruments


def fetch_data_for_symbols(
    validated_symbols_only: bool = True
) -> dict[str, pd.DataFrame]:
    """Fetch data for symbols in the market universe.

    Args:
        validated_symbols_only: If True, only fetch data for validated symbols
    """
    from datetime import datetime, timedelta

    # Get all instruments
    instruments = MarketUniverse.get_all_instruments()
    symbols = [instrument.symbol for instrument in instruments]

    # Filter to only validated symbols if requested
    if validated_symbols_only:
        from .symbol_corrector import run_symbol_validation
        validation_data = run_symbol_validation()
        validation_results = validation_data.get('validation_results', {})

        # Filter to only valid symbols
        valid_symbols = []
        for symbol in symbols:
            result = validation_results.get(symbol)
            if result and hasattr(result, 'status'):
                from .validation import ValidationStatus
                if result.status == ValidationStatus.VALID:
                    valid_symbols.append(symbol)
                else:
                    logger.warning(
                        f"Skipping invalid symbol {symbol}: "
                        f"{result.status.value}"
                    )
            else:
                logger.warning(f"No validation result for symbol {symbol}")

        symbols = valid_symbols
        logger.info(
            f"Filtered to {len(symbols)} validated symbols "
            f"(skipped {len(instruments) - len(symbols)} invalid)"
        )

    # Date range for data fetching
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')

    logger.info(
        f"Fetching data for {len(symbols)} symbols "
        f"from {start_date} to {end_date}"
    )

    # Fetch data for each symbol
    symbol_data = {}

    # Use Ray to fetch data in parallel
    df_refs = []
    symbol_refs = []

    for symbol in symbols:
        df_ref = fetch_data.remote(
            symbol,
            start_date,
            end_date,
            '1d'
        )
        df_refs.append(df_ref)
        symbol_refs.append(symbol)

    # Get all results
    dfs = ray.get(df_refs)

    # Build symbol to dataframe mapping
    for symbol, df in zip(symbol_refs, dfs):
        if not df.empty:
            symbol_data[symbol] = df
        else:
            logger.warning(f"No data available for symbol: {symbol}")

    logger.info(f"Successfully fetched data for {len(symbol_data)} symbols")
    return symbol_data
