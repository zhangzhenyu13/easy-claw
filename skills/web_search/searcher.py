"""
网页搜索Skill实现
基于Playwright的网页搜索工具
"""
import logging
import json
from tir_agent.skills.base import BaseSkill, SkillMetadata

logger = logging.getLogger("tir_agent.skills.web_search")


class WebSearchSkill(BaseSkill):
    """基于Playwright的网页搜索Skill"""

    def __init__(self, metadata: SkillMetadata, config: dict = None):
        super().__init__(metadata, config)
        self._browser = None
        self._playwright = None

    def execute(self, **params) -> str:
        validated = self.validate_params(params)
        action = validated.get("action", "search")

        if action == "search":
            return self._search(
                validated["query"],
                validated.get("max_results", 5),
                validated.get("engine", "bing")
            )
        elif action == "fetch":
            url = validated.get("url")
            if not url:
                return "错误：action为fetch时必须提供url参数"
            return self._fetch_page(url)
        else:
            return f"错误：不支持的操作类型 {action}"

    def _get_browser(self):
        """延迟初始化Playwright浏览器"""
        if self._browser is None:
            try:
                from playwright.sync_api import sync_playwright
                self._playwright = sync_playwright().start()
                self._browser = self._playwright.chromium.launch(
                    headless=True,
                    args=['--no-sandbox', '--disable-dev-shm-usage']
                )
                logger.info("Playwright浏览器已启动")
            except ImportError:
                raise RuntimeError("需要安装playwright：pip install playwright && playwright install chromium")
            except Exception as e:
                raise RuntimeError(f"启动浏览器失败: {e}")
        return self._browser

    def _search(self, query: str, max_results: int = 5, engine: str = "bing") -> str:
        """使用指定搜索引擎搜索（支持bing和baidu）"""
        try:
            browser = self._get_browser()
            page = browser.new_page(
                user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )

            # 根据引擎选择搜索URL和解析配置
            if engine == "baidu":
                search_url = f"https://www.baidu.com/s?wd={query}"
                result_selector = "#content_left .result, #content_left .c-container"
                title_selector = "h3 a"
                snippet_selectors = [".c-abstract", ".content-right_8Zs40", ".c-span9"]
                engine_name = "百度"
            else:  # 默认使用bing
                search_url = f"https://cn.bing.com/search?q={query}&count={max_results}"
                result_selector = "#b_results .b_algo"
                title_selector = "h2 a"
                snippet_selectors = [".b_caption p", "p"]
                engine_name = "必应"

            logger.info("使用%s搜索: %s", engine_name, query)
            page.goto(search_url, timeout=30000, wait_until="domcontentloaded")

            # 等待搜索结果加载
            page.wait_for_selector(result_selector, timeout=10000)

            # 提取搜索结果
            results = []
            items = page.query_selector_all(result_selector)

            for i, item in enumerate(items[:max_results]):
                try:
                    title_el = item.query_selector(title_selector)

                    # 尝试多个摘要选择器
                    snippet_el = None
                    for snippet_sel in snippet_selectors:
                        snippet_el = item.query_selector(snippet_sel)
                        if snippet_el:
                            break

                    title = title_el.inner_text().strip() if title_el else "无标题"
                    link = title_el.get_attribute("href") if title_el else ""
                    snippet = snippet_el.inner_text().strip() if snippet_el else ""

                    results.append({
                        "rank": i + 1,
                        "title": title,
                        "url": link,
                        "snippet": snippet
                    })
                except Exception as e:
                    logger.debug("提取第%d条结果失败: %s", i+1, e)
                    continue

            page.close()

            if not results:
                return f"使用{engine_name}搜索 '{query}' 未找到结果"

            # 格式化输出
            output = f"使用{engine_name}搜索 '{query}' 的结果（共{len(results)}条）：\n\n"
            for r in results:
                output += f"### {r['rank']}. {r['title']}\n"
                output += f"链接: {r['url']}\n"
                output += f"摘要: {r['snippet']}\n\n"

            logger.info("搜索完成，获取 %d 条结果", len(results))
            return output

        except Exception as e:
            logger.error("搜索失败: %s", str(e))
            return f"搜索失败: {str(e)}"

    def _fetch_page(self, url: str) -> str:
        """获取指定URL的页面内容"""
        try:
            browser = self._get_browser()
            page = browser.new_page(
                user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )

            logger.info("获取页面: %s", url)
            page.goto(url, timeout=30000, wait_until="domcontentloaded")

            # 移除无关元素
            for selector in ["script", "style", "nav", "header", "footer", "iframe", "noscript"]:
                for el in page.query_selector_all(selector):
                    try:
                        el.evaluate("el => el.remove()")
                    except:
                        pass

            # 尝试获取主要内容区域
            content = ""
            main_selectors = ["article", "main", "[role='main']", ".content", "#content", ".post-content", ".article-content"]
            for sel in main_selectors:
                el = page.query_selector(sel)
                if el:
                    content = el.inner_text().strip()
                    if len(content) > 100:
                        break

            # 如果没找到主要内容区域，取body
            if len(content) < 100:
                body = page.query_selector("body")
                content = body.inner_text().strip() if body else ""

            page_title = page.title()
            page.close()

            # 截断过长内容
            max_chars = 8000
            if len(content) > max_chars:
                content = content[:max_chars] + f"\n\n... (内容已截断，原文共{len(content)}字符)"

            output = f"页面标题: {page_title}\nURL: {url}\n\n{content}"
            logger.info("页面获取完成，内容长度: %d", len(content))
            return output

        except Exception as e:
            logger.error("获取页面失败: %s", str(e))
            return f"获取页面失败: {str(e)}"

    def cleanup(self):
        """清理浏览器资源"""
        try:
            if self._browser:
                self._browser.close()
                self._browser = None
            if self._playwright:
                self._playwright.stop()
                self._playwright = None
            logger.info("浏览器资源已清理")
        except Exception as e:
            logger.error("清理浏览器资源失败: %s", e)

    def __del__(self):
        self.cleanup()
