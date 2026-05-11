"""
backend/scraper/news_scraper.py
================================
News Scraper — BeautifulSoup4 multi-source Vietnamese financial news crawler.

Responsibilities:
  1. Crawl financial news from multiple Vietnamese sources concurrently.
  2. Extract title, summary (2-3 sentences), source, URL, and published date.
  3. Deduplicate articles by URL.
  4. Return a clean list of dicts ready for sentiment analysis by the LangGraph agent.

Sources Supported:
  - CafeF (cafef.vn)
  - Tin Nhanh Chứng Khoán (tinnhanhchungkhoan.vn)
  - VietStock (vietstock.vn)
  - VnEconomy (vneconomy.vn)

Design Principles:
  - OOP with per-source parser subclasses for extensibility (Strategy Pattern).
  - Async-ready via httpx with configurable timeouts and retries (tenacity).
  - try/except per article so one parse failure does not abort the whole crawl.
  - Loguru for structured logging.
"""

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import Optional
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


# ---------------------------------------------------------------------------
# Article Data Class
# ---------------------------------------------------------------------------

@dataclass
class NewsArticle:
    """Represents a single scraped news article."""

    title:        str
    summary:      str
    url:          str
    source:       str
    published_at: str = field(default_factory=lambda: str(date.today()))

    def to_dict(self) -> dict:
        return {
            "title":        self.title,
            "summary":      self.summary,
            "url":          self.url,
            "source":       self.source,
            "published_at": self.published_at,
        }


# ---------------------------------------------------------------------------
# HTTP Helper
# ---------------------------------------------------------------------------

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
}


@retry(
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=8),
    reraise=True,
)
def _get_html(url: str, timeout: int = 15) -> str:
    """
    Fetch raw HTML from a URL with retry logic.

    Args:
        url:     Target URL.
        timeout: Request timeout in seconds.

    Returns:
        Raw HTML string.

    Raises:
        httpx.HTTPStatusError: On 4xx / 5xx responses (after retries).
        httpx.TimeoutException: If all retry attempts timeout.
    """
    with httpx.Client(headers=_DEFAULT_HEADERS, follow_redirects=True, timeout=timeout) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.text


def _extract_sentences(text: str, max_sentences: int = 3) -> str:
    """Extract the first N sentences from a block of text."""
    if not text:
        return ""
    # Split on sentence-ending punctuation followed by space or newline
    sentences = re.split(r"(?<=[.!?…])\s+", text.strip())
    return " ".join(sentences[:max_sentences]).strip()


# ---------------------------------------------------------------------------
# Abstract Base Parser
# ---------------------------------------------------------------------------

class BaseNewsParser(ABC):
    """
    Abstract base class for source-specific news parsers.

    Each subclass implements:
      - SOURCE_NAME : Human-readable source identifier
      - LIST_URL    : URL of the listing/index page to crawl
      - parse_list(): Extract (title, url) tuples from listing page
      - parse_article(): Extract summary from an article's full page
    """

    SOURCE_NAME: str = "unknown"
    LIST_URL:    str = ""

    def __init__(self, max_articles: int = 10) -> None:
        self.max_articles = max_articles

    def scrape(self) -> list[NewsArticle]:
        """
        Full scrape pipeline for this source:
          1. Fetch listing page.
          2. Extract article links.
          3. Fetch each article page and extract summary.
          4. Return list of NewsArticle objects.
        """
        articles: list[NewsArticle] = []
        logger.info(f"[{self.SOURCE_NAME}] Starting scrape | max={self.max_articles}")

        try:
            listing_html = _get_html(self.LIST_URL)
        except Exception as exc:
            logger.error(f"[{self.SOURCE_NAME}] Failed to fetch listing page: {exc}")
            return articles

        links = self._parse_list(listing_html)
        logger.debug(f"[{self.SOURCE_NAME}] Found {len(links)} links on listing page")

        for title, url in links[: self.max_articles]:
            try:
                article_html = _get_html(url)
                summary      = self._parse_article(article_html)
                pub_date     = self._parse_date(article_html)
                article      = NewsArticle(
                    title        = title.strip(),
                    summary      = summary,
                    url          = url,
                    source       = self.SOURCE_NAME,
                    published_at = pub_date,
                )
                articles.append(article)
                logger.debug(f"[{self.SOURCE_NAME}] Scraped: {title[:60]}…")
                # Polite crawl delay
                time.sleep(0.8)
            except Exception as exc:
                logger.warning(f"[{self.SOURCE_NAME}] Skipped article '{url}': {exc}")
                continue

        logger.success(f"[{self.SOURCE_NAME}] Scraped {len(articles)} articles")
        return articles

    @abstractmethod
    def _parse_list(self, html: str) -> list[tuple[str, str]]:
        """Extract [(title, absolute_url)] from the listing page HTML."""
        ...

    @abstractmethod
    def _parse_article(self, html: str) -> str:
        """Extract summary text from a single article page HTML."""
        ...

    def _parse_date(self, html: str) -> str:
        """
        Attempt to extract publication date from article HTML.
        Override in subclasses for source-specific date formats.
        Falls back to today's date if parsing fails.
        """
        return str(date.today())


# ---------------------------------------------------------------------------
# Source 1: CafeF — cafef.vn/chung-khoan
# ---------------------------------------------------------------------------

class CafeFParser(BaseNewsParser):
    """Parser for cafef.vn — Vietnam's leading financial news portal."""

    SOURCE_NAME = "cafef"
    LIST_URL    = "https://cafef.vn/thi-truong-chung-khoan.chn"

    def _parse_list(self, html: str) -> list[tuple[str, str]]:
        soup  = BeautifulSoup(html, "lxml")
        items = soup.select("h3 a, h2 a, .title a")
        links: list[tuple[str, str]] = []
        for tag in items:
            title = tag.get_text(strip=True)
            href  = tag.get("href", "")
            if href and title and not href.startswith("javascript"):
                full_url = urljoin("https://cafef.vn", href)
                links.append((title, full_url))
        return links

    def _parse_article(self, html: str) -> str:
        soup = BeautifulSoup(html, "lxml")
        # CafeF stores lead paragraph in .sapo or .lead
        lead = soup.select_one(".sapo, .lead, .article-sapo, h2.sapo")
        if lead:
            return _extract_sentences(lead.get_text(separator=" ", strip=True))
        # Fallback: first paragraph of article body
        body_p = soup.select("div.content-detail p, div.detail-content p, article p")
        if body_p:
            return _extract_sentences(body_p[0].get_text(strip=True))
        return ""

    def _parse_date(self, html: str) -> str:
        soup    = BeautifulSoup(html, "lxml")
        tag     = soup.select_one("span.pdate, time[datetime]")
        if tag:
            dt = tag.get("datetime") or tag.get_text(strip=True)
            # Try to extract YYYY-MM-DD pattern
            match = re.search(r"\d{4}-\d{2}-\d{2}", dt)
            if match:
                return match.group(0)
        return str(date.today())


# ---------------------------------------------------------------------------
# Source 2: Tin Nhanh Chứng Khoán — tinnhanhchungkhoan.vn
# ---------------------------------------------------------------------------

class TinnhanhParser(BaseNewsParser):
    """Parser for tinnhanhchungkhoan.vn — specialist stock market news."""

    SOURCE_NAME = "tinnhanhchungkhoan"
    LIST_URL    = "https://tinnhanhchungkhoan.vn/chung-khoan/"

    def _parse_list(self, html: str) -> list[tuple[str, str]]:
        soup  = BeautifulSoup(html, "lxml")
        items = soup.select("article.story h2 a, .story-title a")
        links: list[tuple[str, str]] = []
        for tag in items:
            title = tag.get_text(strip=True)
            href  = tag.get("href", "")
            if href and title:
                full_url = urljoin("https://tinnhanhchungkhoan.vn", href)
                links.append((title, full_url))
        return links

    def _parse_article(self, html: str) -> str:
        soup    = BeautifulSoup(html, "lxml")
        sapo    = soup.select_one(".article-intro, .sapo, .article-sapo")
        if sapo:
            return _extract_sentences(sapo.get_text(separator=" ", strip=True))
        paragraphs = soup.select("div.article-body p")
        if paragraphs:
            return _extract_sentences(paragraphs[0].get_text(strip=True))
        return ""


# ---------------------------------------------------------------------------
# Source 3: VnEconomy — vneconomy.vn/chung-khoan
# ---------------------------------------------------------------------------

class VnEconomyParser(BaseNewsParser):
    """Parser for vneconomy.vn — Vietnam Economic Times."""

    SOURCE_NAME = "vneconomy"
    LIST_URL    = "https://vneconomy.vn/chung-khoan.htm"

    def _parse_list(self, html: str) -> list[tuple[str, str]]:
        soup  = BeautifulSoup(html, "lxml")
        items = soup.select("h3 a, h2 a, .story__heading a, .story-title a")
        links: list[tuple[str, str]] = []
        for tag in items:
            title = tag.get_text(strip=True)
            href  = tag.get("href", "")
            # Filter out category/menu links by requiring title > 5 words
            if href and title and not href.startswith("javascript") and len(title.split()) > 5:
                full_url = urljoin("https://vneconomy.vn", href)
                links.append((title, full_url))
        return links

    def _parse_article(self, html: str) -> str:
        soup = BeautifulSoup(html, "lxml")
        desc = soup.select_one(".story__summary, .article-intro, .sapo, .detail__summary")
        if desc:
            return _extract_sentences(desc.get_text(separator=" ", strip=True))
        paragraphs = soup.select("div.story__content p, div.detail__content p, article p")
        if paragraphs:
            return _extract_sentences(paragraphs[0].get_text(strip=True))
        return ""


# ---------------------------------------------------------------------------
# Orchestrator — NewsScraper
# ---------------------------------------------------------------------------

class NewsScraper:
    """
    Orchestrates multi-source news scraping and returns a deduplicated
    list of NewsArticle dicts ready for downstream sentiment analysis.

    Usage:
        scraper  = NewsScraper(max_articles_per_source=8)
        articles = scraper.scrape_all()
        # articles → list[dict] with keys: title, summary, url, source, published_at
    """

    _PARSERS: list[type[BaseNewsParser]] = [
        CafeFParser,
        TinnhanhParser,
        VnEconomyParser,
    ]

    def __init__(self, max_articles_per_source: int = 8) -> None:
        """
        Args:
            max_articles_per_source: Maximum articles to scrape per source.
                                     Tune based on Groq context window limits.
        """
        self.max_articles_per_source = max_articles_per_source
        logger.debug(f"NewsScraper init | sources={len(self._PARSERS)} | max_per_source={max_articles_per_source}")

    def scrape_all(self) -> list[dict]:
        """
        Scrape all registered sources and return deduplicated article dicts.

        Returns:
            List of article dicts, sorted newest-first by published_at.
            Each dict contains: title, summary, url, source, published_at.
        """
        all_articles: list[NewsArticle] = []
        seen_urls:    set[str]          = set()

        for parser_cls in self._PARSERS:
            parser   = parser_cls(max_articles=self.max_articles_per_source)
            articles = parser.scrape()
            for article in articles:
                if article.url not in seen_urls and article.title:
                    seen_urls.add(article.url)
                    all_articles.append(article)

        # Sort descending by published_at (lexicographic YYYY-MM-DD works)
        all_articles.sort(key=lambda a: a.published_at, reverse=True)

        logger.success(
            f"NewsScraper total: {len(all_articles)} unique articles "
            f"from {len(self._PARSERS)} sources"
        )
        return [a.to_dict() for a in all_articles]

    def scrape_source(self, source_name: str) -> list[dict]:
        """
        Scrape a single named source.

        Args:
            source_name: One of 'cafef', 'tinnhanhchungkhoan', 'vneconomy'.

        Returns:
            List of article dicts for the specified source.

        Raises:
            ValueError: If source_name is not registered.
        """
        parser_map: dict[str, type[BaseNewsParser]] = {
            cls.SOURCE_NAME: cls for cls in self._PARSERS
        }
        if source_name not in parser_map:
            raise ValueError(
                f"Unknown source '{source_name}'. "
                f"Available: {list(parser_map.keys())}"
            )
        parser = parser_map[source_name](max_articles=self.max_articles_per_source)
        return [a.to_dict() for a in parser.scrape()]
