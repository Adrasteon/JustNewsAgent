#!/usr/bin/env python3
"""
Ultra-Fast BBC Crawler for 1000+ Articles/Day

This ultra-aggressive approach prioritizes speed for production scale:
- No AI analysis (too slow for 1000+ articles)
- Pure DOM extraction with heuristic filtering
- Aggressive parallel processing
- Memory-efficient batch processing
- Cookie/modal handling optimized for speed

Target: 1000+ articles/day = ~0.7 articles/second sustained
"""

import asyncio
import json
import time
import re
from datetime import datetime
from playwright.async_api import async_playwright
from typing import List, Dict, Optional, Set
import logging
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ultra_fast_bbc")

class UltraFastBBCCrawler:
    """Ultra-fast crawler optimized for 1000+ articles/day processing"""
    
    def __init__(self, concurrent_browsers: int = 3, batch_size: int = 20):
        self.concurrent_browsers = concurrent_browsers
        self.batch_size = batch_size
        self.processed_articles = []
        self.news_keywords = {
            'high_value': ['arrested', 'charged', 'court', 'police', 'sentenced', 'convicted', 
                          'investigation', 'crime', 'murder', 'theft', 'assault', 'fraud'],
            'medium_value': ['council', 'government', 'minister', 'mp', 'mayor', 'election',
                           'announced', 'confirmed', 'reports', 'statement', 'official'],
            'location_indicators': ['england', 'uk', 'britain', 'london', 'manchester', 
                                  'birmingham', 'leeds', 'liverpool', 'bristol']
        }
    
    def fast_modal_dismissal_script(self) -> str:
        """JavaScript to instantly dismiss all modals/overlays"""
        return """
        // Ultra-fast modal dismissal
        (function() {
            // Cookie consent patterns
            const cookieSelectors = [
                'button:contains("Accept")', 'button:contains("I Agree")',
                'button:contains("Continue")', '[data-testid="accept-all"]',
                '.fc-cta-consent', '.banner-actions-button'
            ];
            
            // Dismiss/close patterns
            const dismissSelectors = [
                'button:contains("Not now")', 'button:contains("Skip")',
                'button:contains("Maybe later")', '[aria-label="Dismiss"]',
                '[aria-label="Close"]', '.close-button', '.modal-close'
            ];
            
            // Try all selectors immediately
            [...cookieSelectors, ...dismissSelectors].forEach(selector => {
                try {
                    document.querySelectorAll(selector).forEach(el => {
                        if (el.offsetParent !== null) el.click();
                    });
                } catch(e) {}
            });
            
            // Remove common overlay containers
            ['.modal', '.overlay', '.popup', '.banner', '.consent'].forEach(cls => {
                document.querySelectorAll(cls).forEach(el => {
                    if (el.style.zIndex > 100) el.remove();
                });
            });
        })();
        """
    
    def calculate_news_score(self, title: str, content: str) -> float:
        """Fast heuristic news scoring (no AI needed)"""
        
        text = (title + " " + content).lower()
        score = 0.0
        
        # High-value news indicators
        for keyword in self.news_keywords['high_value']:
            if keyword in text:
                score += 0.3
        
        # Medium-value indicators
        for keyword in self.news_keywords['medium_value']:
            if keyword in text:
                score += 0.15
        
        # Location relevance
        for location in self.news_keywords['location_indicators']:
            if location in text:
                score += 0.1
        
        # Structure indicators
        if len(content) > 200:
            score += 0.2
        if re.search(r'\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2}', text):  # Dates
            score += 0.1
        if 'bbc' in text:
            score += 0.1
        
        return min(score, 1.0)
    
    async def ultra_fast_extract(self, page) -> Dict[str, str]:
        """Ultra-fast content extraction optimized for speed"""
        
        try:
            # Inject modal dismissal script immediately
            await page.evaluate(self.fast_modal_dismissal_script())
            
            # Get title (fast)
            title = await page.title()
            
            # Fast content extraction with timeout
            content = ""
            try:
                # Try main content areas with short timeout
                content_elem = await page.locator('main, [role="main"], .story-body').first.text_content(timeout=2000)
                content = content_elem[:800] if content_elem else ""
            except:
                # Fallback to paragraphs
                try:
                    paragraphs = await page.locator('p').all_text_contents(timeout=1000)
                    content = " ".join(paragraphs[:3])  # First 3 paragraphs only
                except:
                    content = ""
            
            return {
                "title": title,
                "content": content,
                "extraction_time": time.time()
            }
            
        except Exception as e:
            return {
                "title": "Error",
                "content": f"Extraction failed: {e}",
                "extraction_time": time.time()
            }
    
    async def process_url_ultra_fast(self, browser, url: str) -> Optional[Dict]:
        """Ultra-fast single URL processing"""
        
        start_time = time.time()
        
        try:
            # Fast context creation
            context = await browser.new_context(
                viewport={'width': 1024, 'height': 768},
                java_script_enabled=True
            )
            page = await context.new_page()
            
            # Navigate with aggressive timeout
            await page.goto(url, wait_until='domcontentloaded', timeout=8000)
            
            # Ultra-fast content extraction
            content_data = await self.ultra_fast_extract(page)
            
            # Close immediately
            await context.close()
            
            # Fast news scoring
            news_score = self.calculate_news_score(content_data["title"], content_data["content"])
            
            # Only keep high-quality news (threshold for speed)
            if news_score >= 0.4 and len(content_data["content"]) > 100:
                
                processing_time = time.time() - start_time
                
                return {
                    "url": url,
                    "title": content_data["title"],
                    "content": content_data["content"][:500],  # Truncate for efficiency
                    "news_score": news_score,
                    "processing_time_seconds": processing_time,
                    "timestamp": datetime.now().isoformat(),
                    "status": "success"
                }
            
            return None  # Filtered out
            
        except Exception as e:
            return {
                "url": url,
                "error": str(e),
                "processing_time_seconds": time.time() - start_time,
                "status": "error"
            }
    
    async def get_urls_ultra_fast(self, max_urls: int = 200) -> List[str]:
        """Get URLs as fast as possible"""
        
        browser = await async_playwright().start()
        browser_instance = await browser.chromium.launch(headless=True)
        
        try:
            context = await browser_instance.new_context()
            page = await context.new_page()
            
            # Navigate with timeout
            await page.goto("https://www.bbc.co.uk/news/england", timeout=10000)
            
            # Dismiss modals
            await page.evaluate(self.fast_modal_dismissal_script())
            await asyncio.sleep(1)
            
            # Extract links fast
            links = await page.evaluate("""
                () => {
                    return Array.from(document.querySelectorAll('a[href*="articles/"]'))
                        .map(a => a.href)
                        .filter(href => href.includes('articles/'))
                        .slice(0, 200);
                }
            """)
            
            await browser_instance.close()
            
            logger.info(f"⚡ Found {len(links)} URLs in record time")
            return links
            
        except Exception as e:
            logger.error(f"URL extraction failed: {e}")
            await browser_instance.close()
            return []
    
    async def process_ultra_fast_batch(self, urls: List[str]) -> List[Dict]:
        """Process batches with maximum parallelization"""
        
        # Create multiple browser instances for parallel processing
        playwright = await async_playwright().start()
        browsers = []
        
        try:
            # Launch multiple browsers
            for _ in range(self.concurrent_browsers):
                browser = await playwright.chromium.launch(headless=True)
                browsers.append(browser)
            
            logger.info(f"🚀 Processing {len(urls)} URLs with {len(browsers)} concurrent browsers")
            
            results = []
            browser_index = 0
            
            # Process in aggressive batches
            for i in range(0, len(urls), self.batch_size):
                batch_urls = urls[i:i + self.batch_size]
                batch_start = time.time()
                
                # Distribute URLs across browsers
                tasks = []
                for url in batch_urls:
                    browser = browsers[browser_index % len(browsers)]
                    browser_index += 1
                    tasks.append(self.process_url_ultra_fast(browser, url))
                
                # Process batch concurrently
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Collect successful results
                successful_in_batch = 0
                for result in batch_results:
                    if isinstance(result, dict) and result.get("status") == "success":
                        results.append(result)
                        successful_in_batch += 1
                
                batch_time = time.time() - batch_start
                rate = len(batch_urls) / batch_time if batch_time > 0 else 0
                
                logger.info(f"⚡ Batch {i//self.batch_size + 1}: {successful_in_batch}/{len(batch_urls)} success, {rate:.1f} URLs/sec")
                
                # Minimal delay between batches
                await asyncio.sleep(0.1)
            
            return results
            
        finally:
            # Close all browsers
            for browser in browsers:
                await browser.close()
    
    async def run_ultra_fast_crawl(self, target_articles: int = 200):
        """Main ultra-fast crawling function"""
        
        start_time = time.time()
        logger.info(f"🚀 Ultra-Fast BBC Crawl: Target {target_articles} articles")
        
        # Get URLs fast
        urls = await self.get_urls_ultra_fast(max_urls=target_articles * 2)  # Get extra for filtering
        
        if not urls:
            logger.error("❌ No URLs found!")
            return []
        
        # Process ultra-fast
        results = await self.process_ultra_fast_batch(urls)
        
        # Save results
        total_time = time.time() - start_time
        output_file = f"ultra_fast_bbc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        summary = {
            "ultra_fast_crawl": True,
            "target_articles": target_articles,
            "urls_processed": len(urls),
            "successful_articles": len(results),
            "total_time_seconds": total_time,
            "articles_per_second": len(results) / total_time if total_time > 0 else 0,
            "projected_daily_capacity": (len(results) / total_time) * 86400 if total_time > 0 else 0,
            "timestamp": datetime.now().isoformat(),
            "results": results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"🎉 Ultra-Fast Crawl Complete!")
        logger.info(f"📊 {len(results)} articles in {total_time:.1f}s")
        logger.info(f"⚡ Rate: {len(results) / total_time:.2f} articles/second")
        logger.info(f"📈 Daily capacity: {(len(results) / total_time) * 86400:.0f} articles/day")
        logger.info(f"💾 Results: {output_file}")
        
        return results

async def main():
    """Run ultra-fast crawler"""
    crawler = UltraFastBBCCrawler(concurrent_browsers=3, batch_size=15)
    await crawler.run_ultra_fast_crawl(target_articles=100)

if __name__ == "__main__":
    asyncio.run(main())
