#!/usr/bin/env python3
"""Playwright 脚本：从巨潮资讯网(cninfo)下载上市公司年报 PDF"""
import asyncio
import sys
import re
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

TARGETS = [
    ('600519', '贵州茅台', 2024),
    ('600036', '招商银行', 2024),
    ('300750', '宁德时代', 2024),
    ('601318', '中国平安', 2024),
]

SAVE_DIR = Path(__file__).parent / 'docs'


async def crawl_cninfo(code, company, year):
    """用 Playwright 从 cninfo 搜索并下载年报"""
    from playwright.async_api import async_playwright

    pdf_data = None

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(locale='zh-CN', accept_downloads=True)
        page = await ctx.new_page()

        try:
            # 第一步：搜索股票公告
            search_url = f'http://www.cninfo.com.cn/new/disclosure/stock?stockCode={code}'
            print(f'  打开: {search_url}')
            await page.goto(search_url, wait_until='domcontentloaded', timeout=30000)
            await page.wait_for_timeout(3000)

            # 获取页面标题来确定是否加载成功
            title = await page.title()
            print(f'  页面标题: {title}')

            # 尝试点击"年度报告"标签
            tabs = await page.query_selector_all('a:has-text("年度报告")')
            if tabs:
                print(f'  找到年度报告标签，点击…')
                await tabs[0].click()
                await page.wait_for_timeout(3000)

            # 等待公告列表加载
            await page.wait_for_timeout(2000)

            # 查找公告列表中的链接——寻找包含年报的条目
            links = await page.query_selector_all('a[href*="announcement"]')
            for link in links:
                text = await link.inner_text()
                href = await link.get_attribute('href')
                if '年度报告' in text and '摘要' not in text:
                    print(f'  找到年报链接: {text[:50]}')
                    print(f'  打开公告详情…')
                    await link.click()
                    break

            await page.wait_for_timeout(3000)

            # 在新页面/当前页面查找 PDF 下载链接
            pdf_links = await page.query_selector_all('a[href$=".pdf"], a[href*="finalpage"], a:has-text("PDF下载"), a:has-text("PDF"), a:has-text("下载")')
            for pl in pdf_links:
                href = await pl.get_attribute('href')
                text = await pl.inner_text()
                print(f'  发现链接: {text[:30]} | {str(href)[:80]}')
                if href and ('pdf' in href.lower() or 'finalpage' in href):
                    full_url = href if href.startswith('http') else f'http://www.cninfo.com.cn{href}'
                    print(f'  打开 PDF: {full_url[:80]}')
                    await page.goto(full_url, wait_until='load', timeout=30000)
                    await page.wait_for_timeout(3000)
                    break

            # 等待 PDF 加载（通过 response 拦截）
            async def on_response(resp):
                nonlocal pdf_data
                ct = resp.headers.get('content-type', '')
                if 'application/pdf' in ct:
                    pdf_data = await resp.body()
                    print(f'  捕获 PDF: {len(pdf_data)//1024}KB')

            page.on('response', on_response)
            await page.wait_for_timeout(5000)

            # 尝试获取当前页面
            if not pdf_data:
                ct = page.response.headers.get('content-type', '')
                if 'application/pdf' in ct:
                    pdf_data = await page.response.body()

        except Exception as e:
            print(f'  ✗ 异常: {e}')
        finally:
            await browser.close()

    return pdf_data


async def download_one(code, company, year):
    print(f'下载: {company}({code}) {year}年报')
    pdf_data = await crawl_cninfo(code, company, year)

    if pdf_data and len(pdf_data) > 10000:
        filename = f'{company}_{year}年报.pdf'
        filepath = SAVE_DIR / filename
        with open(filepath, 'wb') as f:
            f.write(pdf_data)
        print(f'  ✓ 已保存: {filename} ({len(pdf_data)//1024}KB)')
        return True
    else:
        print(f'  ✗ 未获取到有效 PDF')
        return False


async def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    targets = TARGETS[:1]  # 先验证茅台
    print(f'准备下载 {len(targets)} 份年报\n')

    success = 0
    for code, name, year in targets:
        ok = await download_one(code, name, year)
        if ok:
            success += 1
        print()

    print(f'完成: {success}/{len(targets)} 成功')


if __name__ == '__main__':
    asyncio.run(main())
