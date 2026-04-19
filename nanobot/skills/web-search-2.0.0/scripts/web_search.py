#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
网络搜索技能
支持多种搜索引擎：百度、必应、DuckDuckGo（全部使用 Playwright）
无需 API Key
"""

import sys
import json
import re
import asyncio
from typing import Dict, Any, List, Optional

try:
    from crawl4ai import AsyncWebCrawler
    HAS_CRAWL4AI = True
except ImportError:
    HAS_CRAWL4AI = False

try:
    from playwright.sync_api import sync_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False


def search_baidu(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """使用 Playwright 进行百度搜索"""
    results = []
    
    if not HAS_PLAYWRIGHT:
        return results
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                viewport={'width': 1920, 'height': 1080}
            )
            page = context.new_page()
            
            url = f"https://www.baidu.com/s?wd={query}&rn={num_results * 2}"
            
            page.goto(url, wait_until='networkidle', timeout=30000)
            page.wait_for_selector('#content_left', timeout=10000)
            
            search_results = page.query_selector_all('#content_left .result, #content_left .c-container')
            
            for result in search_results:
                if len(results) >= num_results:
                    break
                
                try:
                    title_elem = result.query_selector('h3 a, .t a')
                    if title_elem:
                        title = title_elem.inner_text()
                        href = title_elem.get_attribute('href')
                        
                        abstract_elem = result.query_selector('.content-right_8Zs40, .c-abstract, .content-right')
                        body = ''
                        if abstract_elem:
                            body = abstract_elem.inner_text()
                        
                        if title and href and len(title) > 5:
                            if not any(x in href for x in ['baidu.com/home', 'baidu.com/s?', 'passport', 'javascript:']):
                                if href not in [r['href'] for r in results]:
                                    results.append({
                                        'title': title.strip(),
                                        'href': href,
                                        'body': body.strip()
                                    })
                except Exception:
                    continue
            
            browser.close()
    except Exception as e:
        print(f"百度搜索错误: {e}")
    
    return results


def search_bing(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """使用 Playwright 进行必应搜索"""
    results = []
    
    if not HAS_PLAYWRIGHT:
        return results
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                viewport={'width': 1920, 'height': 1080}
            )
            page = context.new_page()
            
            url = f"https://cn.bing.com/search?q={query}&count={num_results}"
            
            page.goto(url, wait_until='networkidle', timeout=30000)
            page.wait_for_selector('#b_results', timeout=10000)
            
            search_results = page.query_selector_all('#b_results .b_algo')
            
            for result in search_results:
                if len(results) >= num_results:
                    break
                
                try:
                    title_elem = result.query_selector('h2 a')
                    if title_elem:
                        title = title_elem.inner_text()
                        href = title_elem.get_attribute('href')
                        
                        snippet_elem = result.query_selector('.b_caption p, p')
                        body = ''
                        if snippet_elem:
                            body = snippet_elem.inner_text()
                        
                        if title and href and len(title) > 5:
                            if href not in [r['href'] for r in results]:
                                results.append({
                                    'title': title.strip(),
                                    'href': href,
                                    'body': body.strip()
                                })
                except Exception:
                    continue
            
            browser.close()
    except Exception as e:
        print(f"必应搜索错误: {e}")
    
    return results


def search_duckduckgo(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """使用 Playwright 进行 DuckDuckGo 搜索"""
    results = []
    
    if not HAS_PLAYWRIGHT:
        return results
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                viewport={'width': 1920, 'height': 1080}
            )
            page = context.new_page()
            
            url = f"https://html.duckduckgo.com/html/?q={query}"
            
            page.goto(url, wait_until='networkidle', timeout=30000)
            page.wait_for_selector('.result', timeout=10000)
            
            search_results = page.query_selector_all('.result')
            
            for result in search_results:
                if len(results) >= num_results:
                    break
                
                try:
                    title_elem = result.query_selector('.result__a')
                    if title_elem:
                        title = title_elem.inner_text()
                        href = title_elem.get_attribute('href')
                        
                        snippet_elem = result.query_selector('.result__snippet')
                        body = ''
                        if snippet_elem:
                            body = snippet_elem.inner_text()
                        
                        if title and href and len(title) > 5:
                            if href not in [r['href'] for r in results]:
                                results.append({
                                    'title': title.strip(),
                                    'href': href,
                                    'body': body.strip()
                                })
                except Exception:
                    continue
            
            browser.close()
    except Exception as e:
        print(f"DuckDuckGo 搜索错误: {e}")
    
    return results


async def crawl_page_async(url: str) -> Dict[str, Any]:
    """异步抓取网页内容"""
    if not HAS_CRAWL4AI:
        return {
            'success': False,
            'message': 'crawl4ai 未安装，无法抓取网页'
        }
    
    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            
            return {
                'success': True,
                'url': url,
                'title': result.metadata.get('title', ''),
                'markdown': result.markdown,
                'text': result.markdown[:5000] if result.markdown else ''
            }
    except Exception as e:
        return {
            'success': False,
            'url': url,
            'message': f'抓取失败: {str(e)}'
        }


def crawl_page(url: str) -> Dict[str, Any]:
    """抓取网页内容（同步接口）"""
    if not HAS_CRAWL4AI:
        return {
            'success': False,
            'message': 'crawl4ai 未安装，无法抓取网页'
        }
    
    try:
        return asyncio.run(crawl_page_async(url))
    except Exception as e:
        return {
            'success': False,
            'url': url,
            'message': f'抓取失败: {str(e)}'
        }


def deep_search(query: str, num_results: int = 5) -> Dict[str, Any]:
    """深度搜索 - 搜索并抓取详情"""
    search_results = search_baidu(query, num_results)
    
    if not search_results:
        return {
            'success': False,
            'query': query,
            'error': '未找到搜索结果'
        }
    
    detailed_content = []
    
    # 抓取前3个结果的详情
    for r in search_results[:3]:
        if HAS_CRAWL4AI:
            crawl_result = crawl_page(r['href'])
            if crawl_result.get('success'):
                detailed_content.append({
                    'url': r['href'],
                    'title': r['title'],
                    'content': crawl_result.get('text', '')[:2000]
                })
    
    return {
        'success': True,
        'query': query,
        'search_results': search_results,
        'detailed_info': {
            'extracted_content': detailed_content
        },
        'message': '深度搜索完成'
    }


def validate_search_params(query: str, num_results: int) -> tuple:
    """验证搜索参数"""
    if not query:
        return False, '搜索关键词不能为空', 0
    
    if not isinstance(query, str):
        return False, '搜索关键词必须是字符串', 0
    
    query = query.strip()
    if len(query) == 0:
        return False, '搜索关键词不能为空', 0
    
    if len(query) > 500:
        return False, '搜索关键词长度不能超过500字符', 0
    
    try:
        num_results = int(num_results)
    except (ValueError, TypeError):
        num_results = 5
    
    num_results = max(1, min(num_results, 20))
    
    return True, '', num_results


def validate_url(url: str) -> tuple:
    """验证 URL"""
    if not url:
        return False, 'URL 不能为空'
    
    if not isinstance(url, str):
        return False, 'URL 必须是字符串'
    
    url = url.strip()
    if len(url) == 0:
        return False, 'URL 不能为空'
    
    if not url.startswith(('http://', 'https://')):
        return False, 'URL 必须以 http:// 或 https:// 开头'
    
    if len(url) > 2000:
        return False, 'URL 长度不能超过2000字符'
    
    return True, ''


def web_search(query: str, num_results: int = 5, region: str = 'cn-zh', deep: bool = False, engine: str = 'baidu') -> Dict[str, Any]:
    """执行网络搜索（全部使用 Playwright）
    
    Args:
        query: 搜索关键词
        num_results: 返回结果数量
        region: 地区代码
        deep: 是否深度搜索
        engine: 搜索引擎，可选 'baidu', 'bing', 'duckduckgo'，默认 'baidu'
    """
    is_valid, error_msg, num_results = validate_search_params(query, num_results)
    if not is_valid:
        return {
            'success': False,
            'query': query if isinstance(query, str) else '',
            'num_results': 0,
            'results': [],
            'errors': [error_msg],
            'message': error_msg
        }
    
    query = query.strip()
    
    if not HAS_PLAYWRIGHT:
        return {
            'success': False,
            'query': query,
            'num_results': 0,
            'results': [],
            'errors': ['请安装 playwright: pip install playwright && playwright install chromium'],
            'message': '缺少依赖'
        }
    
    if deep and HAS_CRAWL4AI:
        return deep_search(query, num_results)
    
    # 根据传入的 engine 参数选择搜索引擎
    engine = engine.lower().strip() if isinstance(engine, str) else 'baidu'
    valid_engines = ['baidu', 'bing', 'duckduckgo']
    
    if engine not in valid_engines:
        return {
            'success': False,
            'query': query,
            'num_results': 0,
            'results': [],
            'errors': [f'不支持的搜索引擎: {engine}。有效选项: {", ".join(valid_engines)}'],
            'message': f'无效的搜索引擎: {engine}'
        }
    
    results = []
    engine_used = engine
    
    try:
        if engine == 'baidu':
            results = search_baidu(query, num_results)
        elif engine == 'bing':
            results = search_bing(query, num_results)
        elif engine == 'duckduckgo':
            results = search_duckduckgo(query, num_results)
    except Exception as e:
        return {
            'success': False,
            'query': query,
            'num_results': 0,
            'results': [],
            'errors': [f'{engine} 搜索失败: {str(e)}'],
            'message': f'{engine} 搜索失败'
        }
    
    return {
        'success': len(results) > 0,
        'query': query,
        'engine': engine_used,
        'num_results': len(results),
        'results': results,
        'message': '搜索完成' if results else '未找到结果'
    }


def execute(action: str, **kwargs) -> Dict[str, Any]:
    """执行技能操作"""
    if not action:
        return {'success': False, 'message': '操作类型不能为空'}
    
    if not isinstance(action, str):
        return {'success': False, 'message': '操作类型必须是字符串'}
    
    action = action.strip().lower()
    
    if action == 'search':
        query = kwargs.get('query', '')
        num_results = kwargs.get('num_results', 5)
        region = kwargs.get('region', 'cn-zh')
        deep = kwargs.get('deep', False)
        engine = kwargs.get('engine', 'baidu')
        return web_search(query, num_results, region, deep, engine)
    
    elif action == 'crawl':
        url = kwargs.get('url', '')
        is_valid, error_msg = validate_url(url)
        if not is_valid:
            return {'success': False, 'message': error_msg}
        return crawl_page(url.strip())
    
    elif action == 'deep_search':
        query = kwargs.get('query', '')
        is_valid, error_msg, num_results = validate_search_params(
            query, kwargs.get('num_results', 5)
        )
        if not is_valid:
            return {'success': False, 'message': error_msg}
        return deep_search(query.strip(), num_results)
    
    else:
        valid_actions = ['search', 'crawl', 'deep_search']
        return {
            'success': False, 
            'message': f'未知操作: {action}。有效操作: {", ".join(valid_actions)}'
        }


def main(input_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """技能入口点"""
    if input_data is None:
        input_data = {}
    
    action = input_data.get('action', 'search')
    kwargs = {k: v for k, v in input_data.items() if k != 'action'}
    return execute(action, **kwargs)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = ' '.join(sys.argv[1:])
    else:
        query = "Python tutorial"
    
    result = main({'action': 'search', 'query': query, 'num_results': 5})
    print(json.dumps(result, ensure_ascii=False, indent=2))
