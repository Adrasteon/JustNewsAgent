#!/home/adra/miniconda3/condabin/conda run -n justnews-v2-py312 python3
from common.observability import get_logger
"""
Production BBC NewsReader Crawler - Robust Implementation

This script integrates the production-grade NewsReader with BBC England crawling
to provide a robust, production-ready news content analysis pipeline.

Features:
- Production-grade NewsReader with fallback strategies
- BBC England depth-first crawling with screenshot capture
- Comprehensive error handling and monitoring
- Resource management and cleanup
"""

import asyncio
import json
import logging

from typing import List, Dict, Optional
import time
from datetime import datetime

# Production NewsReader and web crawling
import sys
sys.path.append('/home/adra/JustNewsAgentic')
from production_newsreader_fixed import ProductionNewsReader
import psycopg2
from scripts.db_dedupe import ensure_table, register_url
from playwright.async_api import async_playwright
from PIL import Image
import io

# Configure logging
logger = get_logger(__name__)

class ProductionBBCNewsReaderCrawler:
    """
    Production-grade BBC crawler with robust NewsReader integration.
    
    Features:
    - Production NewsReader with fallback strategies
    - England depth-first crawling strategy
    - Screenshot-based content capture for JavaScript handling
    - Comprehensive error handling and resource management
    """
    
    def __init__(self):
        self.newsreader = None
        self.browser = None
        self.context = None
        self.results = []
        self.failed_urls = []
        self.processed_count = 0
        
    async def initialize(self) -> bool:
        """Initialize production crawler with all components"""
        try:
            logger.info("🚀 Initializing Production BBC NewsReader Crawler...")
            
            # Initialize production NewsReader
            self.newsreader = ProductionNewsReader()
            newsreader_success = await self.newsreader.initialize_with_fallback_strategy()
            
            if not newsreader_success:
                logger.error("❌ Failed to initialize NewsReader")
                return False
            
            # Check NewsReader health
            health = await self.newsreader.health_check()
            logger.info(f"📊 NewsReader Health: {health}")
            
            # Initialize browser for screenshot capture
            playwright = await async_playwright().start()
            self.browser = await playwright.chromium.launch(headless=True)
            self.context = await self.browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            )
            
            logger.info("✅ Production crawler initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Crawler initialization failed: {e}")
            return False
    
    async def get_bbc_england_urls(self, max_urls: int = 15) -> List[str]:
        """
        Get BBC England article URLs using depth-first strategy.
        
        Args:
            max_urls: Maximum number of URLs to collect
            
        Returns:
            List[str]: England-focused BBC article URLs
        """
        try:
            logger.info("🔍 Discovering BBC England article URLs...")
            
            page = await self.context.new_page()
            
            # Start from BBC England page
            await page.goto("https://www.bbc.co.uk/news/england", wait_until='networkidle')
            await asyncio.sleep(2)  # Allow dynamic content to load
            
            urls = set()
            
            # Look for article links using correct BBC selectors
            selectors = [
                'a[href*="articles/"]',  # New BBC article format
                'h3 a[href*="/news/"]',  # Headlines in h3 tags
                'h2 a[href*="/news/"]',  # Headlines in h2 tags
            ]
            
            for selector in selectors:
                try:
                    links = await page.locator(selector).all()
                    
                    for link in links:
                        try:
                            href = await link.get_attribute('href')
                            if href:
                                # Convert relative URLs to absolute
                                if href.startswith('/'):
                                    href = f"https://www.bbc.co.uk{href}"
                                
                                # Filter for England articles and proper news content
                                if ('articles/' in href or 
                                    any(pattern in href for pattern in ['/news/uk-england-', '/news/england-'])):
                                    # Avoid duplicate and non-article URLs
                                    if not any(skip in href for skip in ['video', 'live', 'sport', 'topics', 'regions']):
                                        urls.add(href)
                        except Exception:
                            # Skip individual link processing errors
                            continue
                except Exception as e:
                    logger.warning(f"⚠️ Selector {selector} failed: {e}")
                    continue
            
            # Convert to list and limit
            england_urls = list(urls)[:max_urls]
            
            await page.close()
            
            logger.info(f"✅ Found {len(england_urls)} BBC England article URLs")
            return england_urls
            
        except Exception as e:
            logger.error(f"❌ Error discovering BBC England URLs: {e}")
            return []
    
    async def capture_page_screenshot(self, url: str) -> Optional[Image.Image]:
        """
        Capture page screenshot for NewsReader analysis.
        
        Args:
            url: URL to capture
            
        Returns:
            Optional[Image.Image]: PIL Image or None if failed
        """
        try:
            page = await self.context.new_page()
            
            # Navigate and wait for content
            await page.goto(url, wait_until='networkidle', timeout=30000)
            await asyncio.sleep(3)  # Additional wait for dynamic content
            
            # Handle cookie banners if present
            try:
                cookie_button = page.locator('button:has-text("Accept"), button:has-text("OK"), button:has-text("Agree")')
                if await cookie_button.count() > 0:
                    await cookie_button.first.click()
                    await asyncio.sleep(1)
            except Exception:
                pass  # Cookie handling is optional
            
            # Capture screenshot
            screenshot_bytes = await page.screenshot(
                full_page=True,
                type='png'
            )
            
            await page.close()
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(screenshot_bytes))
            
            logger.info(f"✅ Screenshot captured: {url}")
            return image
            
        except Exception as e:
            logger.error(f"❌ Screenshot capture failed for {url}: {e}")
            return None
    
    async def analyze_article_with_newsreader(self, url: str, image: Image.Image) -> Optional[Dict]:
        """
        Analyze article content using production NewsReader.
        
        Args:
            url: Article URL
            image: Screenshot image to analyze
            
        Returns:
            Optional[Dict]: Analysis result or None if failed
        """
        try:
            # Create analysis context
            context = f"Analyze this BBC England news article from {url}. Focus on the main news content, headline, and key information."
            
            # Analyze with production NewsReader
            analysis = await self.newsreader.analyze_news_content(image, context)
            
            if analysis:
                result = {
                    "url": url,
                    "analysis": analysis,
                    "timestamp": datetime.now().isoformat(),
                    "model_type": self.newsreader.model_type,
                    "status": "success"
                }
                
                logger.info(f"✅ Analysis complete: {url}")
                return result
            else:
                logger.warning(f"⚠️ No analysis result for: {url}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Analysis failed for {url}: {e}")
            return None
    
    async def process_article(self, url: str) -> Optional[Dict]:
        """
        Process single article with production pipeline.
        
        Args:
            url: Article URL to process
            
        Returns:
            Optional[Dict]: Processing result or None if failed
        """
        try:
            logger.info(f"🔄 Processing: {url}")
            
            # Capture screenshot
            image = await self.capture_page_screenshot(url)
            if not image:
                self.failed_urls.append({"url": url, "reason": "screenshot_failed"})
                return None
            
            # Analyze with NewsReader
            result = await self.analyze_article_with_newsreader(url, image)
            if result:
                self.processed_count += 1
                return result
            else:
                self.failed_urls.append({"url": url, "reason": "analysis_failed"})
                return None
                
        except Exception as e:
            logger.error(f"❌ Article processing failed for {url}: {e}")
            self.failed_urls.append({"url": url, "reason": str(e)})
            return None
    
    async def crawl_bbc_england_articles(self, max_articles: int = 15) -> Dict:
        """
        Main crawling method with production pipeline.
        
        Args:
            max_articles: Maximum number of articles to process
            
        Returns:
            Dict: Complete crawling results with statistics
        """
        try:
            start_time = time.time()
            
            logger.info(f"🚀 Starting BBC England crawling (max {max_articles} articles)")
            
            # Get England article URLs
            urls = await self.get_bbc_england_urls(max_articles)
            if not urls:
                return {"error": "No URLs found", "results": []}
            
            logger.info(f"📝 Processing {len(urls)} BBC England articles...")

            # Ensure dedupe table exists (DB-side gate)
            try:
                conn = psycopg2.connect(dbname='justnews', user='justnews_user', password='password123', host='localhost')
                ensure_table(conn)
            except Exception:
                conn = None

            # Process articles with production pipeline
            for i, url in enumerate(urls, 1):
                logger.info(f"📰 [{i}/{len(urls)}] Processing article...")

                # register URL; if it already exists we skip saving result
                should_save = True
                try:
                    if conn:
                        inserted = register_url(conn, url)
                        should_save = bool(inserted)
                except Exception:
                    # on error, fall back to saving
                    should_save = True

                result = await self.process_article(url)
                if result and should_save:
                    self.results.append(result)

                # Brief pause between requests
                await asyncio.sleep(1)
            
            # Compile final results
            end_time = time.time()
            duration = round(end_time - start_time, 2)
            
            # Close DB conn if used
            if conn:
                conn.close()

            # Get final NewsReader metrics
            final_metrics = await self.newsreader.get_performance_metrics()
            
            crawl_results = {
                "success": True,
                "total_urls_found": len(urls),
                "successful_analyses": len(self.results),
                "failed_analyses": len(self.failed_urls),
                "success_rate": round((len(self.results) / len(urls) * 100), 1) if urls else 0,
                "processing_time_seconds": duration,
                "newsreader_model": self.newsreader.model_type,
                "newsreader_metrics": final_metrics,
                "results": self.results,
                "failed_urls": self.failed_urls,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"✅ BBC England crawling complete: {len(self.results)}/{len(urls)} successful")
            return crawl_results
            
        except Exception as e:
            logger.error(f"❌ BBC England crawling failed: {e}")
            return {"error": str(e), "results": self.results}
    
    async def cleanup(self):
        """Production cleanup with proper resource management"""
        try:
            if self.context:
                await self.context.close()
            
            if self.browser:
                await self.browser.close()
            
            if self.newsreader:
                self.newsreader.cleanup()
            
            logger.info("✅ Production crawler cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")

async def main():
    """Production BBC NewsReader crawler execution"""
    
    crawler = ProductionBBCNewsReaderCrawler()
    
    try:
        # Initialize production crawler
        if not await crawler.initialize():
            logger.error("❌ Failed to initialize production crawler")
            return
        
        # Run BBC England crawling
        results = await crawler.crawl_bbc_england_articles(max_articles=10)
        
        # Save results
        output_file = "bbc_england_production_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Display summary
        print("\n" + "="*80)
        print("🏆 PRODUCTION BBC ENGLAND CRAWLER RESULTS")
        print("="*80)
        print(f"📊 Success Rate: {results.get('success_rate', 0)}%")
        print(f"✅ Successful: {results.get('successful_analyses', 0)}")
        print(f"❌ Failed: {results.get('failed_analyses', 0)}")
        print(f"⏱️  Duration: {results.get('processing_time_seconds', 0)}s")
        print(f"🤖 Model: {results.get('newsreader_model', 'unknown')}")
        print(f"💾 Memory: {results.get('newsreader_metrics', {}).get('memory_allocated_gb', 0)}GB")
        print(f"📄 Results saved: {output_file}")
        print("="*80)
        
    finally:
        # Always cleanup
        await crawler.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
