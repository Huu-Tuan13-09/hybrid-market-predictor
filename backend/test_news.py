
from backend.scraper.news_scraper import NewsScraper
import json

def test_scrapers():
    print("--- Đang kiểm tra các nguồn tin ---")
    scraper = NewsScraper(max_articles_per_source=5)
    
    for parser_cls in scraper._PARSERS:
        source_name = parser_cls.SOURCE_NAME
        print(f"\nĐang kiểm tra nguồn: {source_name.upper()}...")
        try:
            articles = scraper.scrape_source(source_name)
            if articles:
                print(f"✅ THÀNH CÔNG: Tìm thấy {len(articles)} bài báo.")
                for i, art in enumerate(articles[:2]):
                    print(f"   {i+1}. {art['title'][:60]}...")
            else:
                print(f"❌ THẤB BẠI: Không tìm thấy bài báo nào. (Có thể giao diện web đã đổi)")
        except Exception as e:
            print(f"💥 LỖI HỆ THỐNG khi crawl {source_name}: {e}")

if __name__ == "__main__":
    test_scrapers()
