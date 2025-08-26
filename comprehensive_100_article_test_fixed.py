#!/usr/bin/env python3
"""
JustNews V2 Comprehensive 100-Article Pipeline Test

This script runs a complete end-to-end test of the JustNews V2 pipeline:
- Uses Scout agent production crawler to get BBC England articles
- Processes articles through NewsReader agent for content analysis  
- Stores results in Memory agent
- Tests all agents through MCP bus communication
- Uses GPU crash-resolved configuration throughout

Features:
- Batch processing for optimal performance
- Async operations for concurrent processing
- Comprehensive error handling and logging
- Performance monitoring and statistics
- Full pipeline validation

Usage:
    python comprehensive_100_article_test.py [target_articles]
    
Example:
    python comprehensive_100_article_test.py 100  # Process 100 articles
    python comprehensive_100_article_test.py 10   # Process 10 articles (test)
"""

import asyncio
import json
import requests
import time
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass
import sys

# Configure logging for comprehensive monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'comprehensive_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger("comprehensive_test")

# Configuration
MCP_BUS_URL = "http://localhost:8000"
BATCH_SIZE = 5
MAX_CONCURRENT = 3

@dataclass
class ArticleResult:
    """Result of processing a single article"""
    url: str
    success: bool
    scout_content: Optional[Dict] = None
    newsreader_analysis: Optional[Dict] = None
    memory_storage: Optional[Dict] = None
    error: Optional[str] = None
    processing_time: float = 0.0
    timestamp: str = ""

class ComprehensivePipelineTest:
    """Comprehensive pipeline test orchestrator"""
    
    def __init__(self):
        self.results: List[ArticleResult] = []
        self.start_time = None
        self.agents_health = {}
        
    async def call_agent(self, agent: str, tool: str, args: List = None, kwargs: Dict = None) -> Dict:
        """Call agent through MCP bus with proper error handling"""
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}
            
        payload = {
            "agent": agent,
            "tool": tool,
            "args": args,
            "kwargs": kwargs
        }
        
        try:
            response = requests.post(f"{MCP_BUS_URL}/call", json=payload, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"❌ Agent call failed - {agent}.{tool}: {e}")
            return {"error": str(e), "success": False}
    
    async def get_bbc_articles(self, target_articles: int) -> List[str]:
        """Get BBC England article URLs using Scout agent production crawler"""
        logger.info(f"🔍 Getting {target_articles} BBC England articles via Scout agent...")
        
        try:
            result = await self.call_agent(
                agent="scout",
                tool="production_crawl_ultra_fast",
                kwargs={"site": "bbc", "target_articles": target_articles}
            )
            
            if "error" in result:
                logger.error(f"❌ Scout agent failed: {result['error']}")
                return []
                
            # Extract URLs from the Scout agent response
            urls = []
            if "articles" in result:
                for article in result["articles"]:
                    if isinstance(article, dict) and "url" in article:
                        urls.append(article["url"])
            
            logger.info(f"✅ Scout agent returned {len(urls)} article URLs")
            return urls[:target_articles]
            
        except Exception as e:
            logger.error(f"❌ Failed to get articles from Scout: {e}")
            return []
    
    async def process_single_article(self, url: str) -> ArticleResult:
        """Process a single article through the complete pipeline"""
        start_time = time.time()
        result = ArticleResult(
            url=url,
            success=False,
            timestamp=datetime.now().isoformat()
        )
        
        try:
            logger.info(f"🔄 Processing article: {url}")
            
            # Step 1: Get article content through NewsReader
            newsreader_result = await self.call_agent(
                agent="newsreader-v2",
                tool="extract_news_from_url",
                args=[url]
            )
            
            if "error" not in newsreader_result:
                result.newsreader_analysis = newsreader_result
                logger.info(f"✅ NewsReader processed: {url}")
                
                # Step 2: Store in Memory agent
                if "content" in newsreader_result:
                    memory_result = await self.call_agent(
                        agent="memory",
                        tool="save_article",
                        kwargs={
                            "content": newsreader_result["content"],
                            "metadata": {
                                "url": url,
                                "timestamp": result.timestamp,
                                "source": "bbc_england",
                                "pipeline": "comprehensive_test"
                            }
                        }
                    )
                    
                    if "error" not in memory_result:
                        result.memory_storage = memory_result
                        result.success = True
                        logger.info(f"✅ Memory stored: {url}")
                    else:
                        result.error = f"Memory storage failed: {memory_result.get('error', 'unknown')}"
                        logger.error(f"❌ Memory failed for {url}: {result.error}")
                else:
                    result.error = "No content in NewsReader result"
                    logger.error(f"❌ No content from NewsReader: {url}")
            else:
                result.error = f"NewsReader failed: {newsreader_result.get('error', 'unknown')}"
                logger.error(f"❌ NewsReader failed for {url}: {result.error}")
                
        except Exception as e:
            result.error = str(e)
            logger.error(f"❌ Article processing failed for {url}: {e}")
        
        result.processing_time = time.time() - start_time
        return result
    
    async def process_batch(self, urls: List[str]) -> List[ArticleResult]:
        """Process articles in batches for optimal performance"""
        logger.info(f"🔄 Processing batch of {len(urls)} articles...")
        
        # Create semaphore for controlling concurrency
        semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        
        async def process_with_limit(url):
            async with semaphore:
                return await self.process_single_article(url)
        
        # Process articles concurrently within limits
        tasks = [process_with_limit(url) for url in urls]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                logger.error(f"❌ Exception processing {urls[i]}: {result}")
                processed_results.append(ArticleResult(
                    url=urls[i],
                    success=False,
                    error=str(result),
                    timestamp=datetime.now().isoformat()
                ))
            else:
                processed_results.append(result)
        
        successful = sum(1 for r in processed_results if r.success)
        logger.info(f"✅ Batch complete: {successful}/{len(urls)} successful")
        
        return processed_results
    
    async def check_agent_health(self) -> Dict[str, bool]:
        """Check agent health directly (bypass MCP bus for health checks)"""
        agents = {
            "mcp_bus": "http://localhost:8000/health",
            "scout": "http://localhost:8002/health", 
            "newsreader": "http://localhost:8009/health",
            "memory": "http://localhost:8007/health"
        }
        
        health_status = {}
        for agent, url in agents.items():
            try:
                response = requests.get(url, timeout=5)
                health_status[agent] = response.status_code == 200
            except:
                health_status[agent] = False
                
        return health_status
    
    async def run_comprehensive_test(self, target_articles: int = 100) -> Dict[str, Any]:
        """Run the complete comprehensive pipeline test"""
        self.start_time = time.time()
        logger.info(f"🚀 Starting comprehensive test for {target_articles} articles")
        
        try:
            # Health check first
            logger.info("🔍 Checking agent health...")
            self.agents_health = await self.check_agent_health()
            healthy_agents = [k for k, v in self.agents_health.items() if v]
            unhealthy_agents = [k for k, v in self.agents_health.items() if not v]
            
            logger.info(f"✅ Healthy agents: {healthy_agents}")
            if unhealthy_agents:
                logger.warning(f"⚠️ Unhealthy agents: {unhealthy_agents}")
            
            # Get articles from Scout agent
            urls = await self.get_bbc_articles(target_articles)
            if not urls:
                return {"error": "No articles retrieved from Scout agent"}
            
            logger.info(f"📰 Processing {len(urls)} articles in batches of {BATCH_SIZE}")
            
            # Process articles in batches
            all_results = []
            for i in range(0, len(urls), BATCH_SIZE):
                batch_urls = urls[i:i + BATCH_SIZE]
                batch_num = (i // BATCH_SIZE) + 1
                total_batches = (len(urls) + BATCH_SIZE - 1) // BATCH_SIZE
                
                logger.info(f"🔄 Processing batch {batch_num}/{total_batches}")
                
                batch_results = await self.process_batch(batch_urls)
                all_results.extend(batch_results)
                self.results = all_results
                
                # Brief pause between batches
                if i + BATCH_SIZE < len(urls):
                    await asyncio.sleep(1)
            
            # Calculate final statistics
            total_time = time.time() - self.start_time
            successful_results = [r for r in all_results if r.success]
            
            statistics = {
                "total_requested": target_articles,
                "total_urls_found": len(urls),
                "total_processed": len(all_results),
                "successful": len(successful_results),
                "failed": len(all_results) - len(successful_results),
                "total_time": total_time,
                "articles_per_second": len(all_results) / total_time if total_time > 0 else 0,
                "success_rate": (len(successful_results) / len(all_results) * 100) if all_results else 0
            }
            
            final_results = {
                "success": True,
                "statistics": statistics,
                "agents_health": self.agents_health,
                "results": [
                    {
                        "url": r.url,
                        "success": r.success,
                        "error": r.error,
                        "processing_time": r.processing_time,
                        "timestamp": r.timestamp,
                        "has_content": r.newsreader_analysis is not None,
                        "stored_in_memory": r.memory_storage is not None
                    }
                    for r in all_results
                ],
                "sample_successful_result": successful_results[0].__dict__ if successful_results else None,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add convenience metrics
            final_results.update({
                "success_rate": statistics["success_rate"],
                "articles_per_second": statistics["articles_per_second"]
            })
            
            logger.info(f"🎉 Comprehensive test complete!")
            logger.info(f"📊 Statistics: {statistics}")
            
            return final_results
            
        except Exception as e:
            logger.error(f"❌ Comprehensive test failed: {e}")
            return {"error": str(e), "results": self.results}

async def main():
    """Main execution function"""
    
    logger.info("🚀 JustNews V2 Comprehensive 100-Article Pipeline Test")
    logger.info(f"⏰ Starting at: {datetime.now()}")
    logger.info("🔧 Using GPU crash-resolved configuration")
    logger.info("🌐 Target: BBC England news articles")
    logger.info("📡 Architecture: MCP Bus multi-agent pipeline")
    
    # Get target articles from command line argument
    target_articles = 100
    if len(sys.argv) > 1:
        try:
            target_articles = int(sys.argv[1])
            logger.info(f"🎯 Target articles set to: {target_articles}")
        except ValueError:
            logger.warning(f"⚠️ Invalid argument '{sys.argv[1]}', using default: {target_articles}")
    
    # Run comprehensive test
    tester = ComprehensivePipelineTest()
    results = await tester.run_comprehensive_test(target_articles)
    
    # Save results to file
    output_file = f"comprehensive_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Display summary
    print("\n" + "="*80)
    print("🏆 COMPREHENSIVE PIPELINE TEST RESULTS")
    print("="*80)
    print(f"📄 Detailed results saved to: {output_file}")
    
    if "error" not in results:
        print(f"\n🎉 Test completed successfully!")
        print(f"📊 Processed {results['statistics']['total_processed']} articles")
        print(f"✅ Success rate: {results['success_rate']:.1f}%")
        print(f"⚡ Performance: {results['articles_per_second']:.2f} articles/second")
    else:
        print(f"\n❌ Test failed: {results['error']}")
        
    return results

if __name__ == "__main__":
    asyncio.run(main())
