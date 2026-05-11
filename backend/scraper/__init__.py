"""
backend/scraper/__init__.py
Scraper package — Data Ingestion Layer.

Exports:
  - MarketDataFetcher : yfinance OHLCV downloader
  - NewsScraper       : BeautifulSoup4 multi-source news crawler
"""

from .market_data import MarketDataFetcher
from .news_scraper import NewsScraper

__all__ = ["MarketDataFetcher", "NewsScraper"]
