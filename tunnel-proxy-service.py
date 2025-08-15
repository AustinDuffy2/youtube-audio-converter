#!/usr/bin/env python3
"""
🌐 TUNNEL PROXY SERVICE FOR YOUTUBE BYPASS
=============================================
This service creates multiple tunnels with different IP addresses
to bypass YouTube's bot detection by rotating through different endpoints.
"""

import asyncio
import subprocess
import time
import requests
import random
import json
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TunnelProxyService:
    """Manages multiple tunnel connections for IP rotation"""
    
    def __init__(self):
        self.tunnels: List[Dict] = []
        self.active_tunnels: List[str] = []
        
    async def create_ngrok_tunnels(self, count: int = 3) -> List[str]:
        """Create multiple ngrok tunnels for IP rotation"""
        tunnel_urls = []
        
        for i in range(count):
            try:
                # Start ngrok tunnel on different ports
                port = 8000 + i
                process = subprocess.Popen([
                    'ngrok', 'http', str(port), 
                    '--log=stdout',
                    '--region=us'  # Different regions for different IPs
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # Wait for tunnel to establish
                await asyncio.sleep(3)
                
                # Get tunnel URL from ngrok API
                try:
                    response = requests.get('http://127.0.0.1:4040/api/tunnels')
                    if response.status_code == 200:
                        tunnels_data = response.json()
                        for tunnel in tunnels_data.get('tunnels', []):
                            if tunnel.get('config', {}).get('addr') == f'http://localhost:{port}':
                                public_url = tunnel.get('public_url')
                                if public_url:
                                    tunnel_urls.append(public_url)
                                    self.tunnels.append({
                                        'url': public_url,
                                        'port': port,
                                        'process': process,
                                        'region': 'us'
                                    })
                                    logger.info(f"✅ Created tunnel {i+1}: {public_url}")
                except Exception as e:
                    logger.error(f"❌ Failed to get tunnel URL for port {port}: {e}")
                    
            except Exception as e:
                logger.error(f"❌ Failed to create tunnel {i+1}: {e}")
        
        self.active_tunnels = tunnel_urls
        return tunnel_urls
    
    async def create_localtunnel_connections(self, count: int = 3) -> List[str]:
        """Create multiple localtunnel connections"""
        tunnel_urls = []
        
        for i in range(count):
            try:
                port = 8000 + i
                subdomain = f"ytbypass{i}{random.randint(1000, 9999)}"
                
                # Start localtunnel
                process = subprocess.Popen([
                    'npx', 'localtunnel', '--port', str(port), 
                    '--subdomain', subdomain
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                await asyncio.sleep(2)
                
                tunnel_url = f"https://{subdomain}.loca.lt"
                tunnel_urls.append(tunnel_url)
                
                self.tunnels.append({
                    'url': tunnel_url,
                    'port': port,
                    'process': process,
                    'type': 'localtunnel'
                })
                
                logger.info(f"✅ Created localtunnel {i+1}: {tunnel_url}")
                
            except Exception as e:
                logger.error(f"❌ Failed to create localtunnel {i+1}: {e}")
        
        return tunnel_urls
    
    def get_random_tunnel(self) -> Optional[str]:
        """Get a random active tunnel URL"""
        if self.active_tunnels:
            return random.choice(self.active_tunnels)
        return None
    
    def get_tunnel_with_different_ip(self, exclude_url: str = None) -> Optional[str]:
        """Get a tunnel with a different IP address"""
        available = [url for url in self.active_tunnels if url != exclude_url]
        if available:
            return random.choice(available)
        return None
    
    async def test_tunnel_connectivity(self) -> Dict[str, bool]:
        """Test connectivity of all tunnels"""
        results = {}
        
        for tunnel in self.tunnels:
            url = tunnel['url']
            try:
                # Test with a simple health check
                response = requests.get(f"{url}/health", timeout=5)
                results[url] = response.status_code == 200
                logger.info(f"🔍 Tunnel {url}: {'✅ OK' if results[url] else '❌ FAIL'}")
            except Exception as e:
                results[url] = False
                logger.error(f"❌ Tunnel {url} failed: {e}")
        
        return results
    
    def cleanup_tunnels(self):
        """Clean up all tunnel processes"""
        for tunnel in self.tunnels:
            try:
                process = tunnel.get('process')
                if process:
                    process.terminate()
                    logger.info(f"🧹 Cleaned up tunnel: {tunnel['url']}")
            except Exception as e:
                logger.error(f"❌ Failed to cleanup tunnel: {e}")
    
    async def rotate_ip_addresses(self) -> List[str]:
        """Create tunnels with different IP addresses for rotation"""
        logger.info("🌐 Setting up IP rotation with multiple tunnels...")
        
        # Try different tunnel services for maximum IP diversity
        ngrok_tunnels = []
        localtunnel_tunnels = []
        
        try:
            # Create ngrok tunnels (premium service, better IPs)
            ngrok_tunnels = await self.create_ngrok_tunnels(2)
        except Exception as e:
            logger.warning(f"⚠️ ngrok failed: {e}")
        
        try:
            # Create localtunnel connections (free, different IP pool)
            localtunnel_tunnels = await self.create_localtunnel_connections(2)
        except Exception as e:
            logger.warning(f"⚠️ localtunnel failed: {e}")
        
        all_tunnels = ngrok_tunnels + localtunnel_tunnels
        
        if all_tunnels:
            logger.info(f"✅ Created {len(all_tunnels)} tunnels for IP rotation")
            return all_tunnels
        else:
            logger.error("❌ Failed to create any tunnels for IP rotation")
            return []

# PROXY ENDPOINT MANAGER
class ProxyEndpointManager:
    """Manages different proxy endpoints and services"""
    
    def __init__(self):
        self.free_proxies = []
        self.residential_proxies = []
        self.tunnel_service = TunnelProxyService()
    
    async def get_free_proxy_list(self) -> List[str]:
        """Fetch free proxy list from various sources"""
        proxies = []
        
        # Free proxy APIs (these change frequently)
        proxy_sources = [
            'https://api.proxyscrape.com/v2/?request=get&protocol=http&timeout=10000&country=all',
            'https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/http.txt',
            'https://raw.githubusercontent.com/clarketm/proxy-list/master/proxy-list-raw.txt'
        ]
        
        for source in proxy_sources:
            try:
                response = requests.get(source, timeout=10)
                if response.status_code == 200:
                    proxy_list = response.text.strip().split('\n')
                    proxies.extend([f"http://{proxy.strip()}" for proxy in proxy_list if proxy.strip()])
                    logger.info(f"✅ Loaded {len(proxy_list)} proxies from {source[:30]}...")
            except Exception as e:
                logger.error(f"❌ Failed to load proxies from {source[:30]}: {e}")
        
        # Remove duplicates and invalid entries
        unique_proxies = list(set([p for p in proxies if '.' in p and ':' in p]))
        self.free_proxies = unique_proxies[:50]  # Limit to 50 best proxies
        
        logger.info(f"🌐 Total unique proxies available: {len(self.free_proxies)}")
        return self.free_proxies
    
    async def test_proxy_connectivity(self, proxy: str) -> bool:
        """Test if a proxy is working"""
        try:
            response = requests.get(
                'https://httpbin.org/ip', 
                proxies={'http': proxy, 'https': proxy}, 
                timeout=5
            )
            return response.status_code == 200
        except:
            return False
    
    async def get_working_proxies(self, max_count: int = 10) -> List[str]:
        """Get a list of working proxies"""
        if not self.free_proxies:
            await self.get_free_proxy_list()
        
        working_proxies = []
        test_tasks = []
        
        # Test proxies in parallel
        for proxy in self.free_proxies[:30]:  # Test first 30
            test_tasks.append(self.test_proxy_connectivity(proxy))
        
        if test_tasks:
            results = await asyncio.gather(*test_tasks, return_exceptions=True)
            for proxy, is_working in zip(self.free_proxies[:30], results):
                if is_working and len(working_proxies) < max_count:
                    working_proxies.append(proxy)
                    logger.info(f"✅ Working proxy: {proxy}")
        
        logger.info(f"🌐 Found {len(working_proxies)} working proxies")
        return working_proxies

# MAIN TUNNEL PROXY ORCHESTRATOR
async def setup_youtube_bypass_infrastructure():
    """Set up complete bypass infrastructure with tunnels and proxies"""
    logger.info("🚀 Setting up YouTube bypass infrastructure...")
    
    # Initialize services
    tunnel_service = TunnelProxyService()
    proxy_manager = ProxyEndpointManager()
    
    # Set up IP rotation tunnels
    tunnel_urls = await tunnel_service.rotate_ip_addresses()
    
    # Get working proxy list
    working_proxies = await proxy_manager.get_working_proxies(10)
    
    # Combine all bypass methods
    bypass_endpoints = {
        'tunnels': tunnel_urls,
        'proxies': working_proxies,
        'total_endpoints': len(tunnel_urls) + len(working_proxies)
    }
    
    logger.info(f"✅ Bypass infrastructure ready:")
    logger.info(f"   🌐 Tunnels: {len(tunnel_urls)}")
    logger.info(f"   🔄 Proxies: {len(working_proxies)}")
    logger.info(f"   🎯 Total endpoints: {bypass_endpoints['total_endpoints']}")
    
    return bypass_endpoints

if __name__ == "__main__":
    # Run the bypass infrastructure setup
    asyncio.run(setup_youtube_bypass_infrastructure())
