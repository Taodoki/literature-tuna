import json
import sys
from pathlib import Path

import httpx
import numpy as np

import rag_config as cfg

sys.stdout.reconfigure(encoding='utf-8')

SYSTEM_PROMPT = """你是一位金融文档分析助手。你的回答必须：
1. 严格基于上方"参考文档"中的内容，不要编造数据或信息
2. 如果参考文档不包含回答问题所需的信息，请明确说明"参考文档中未找到相关信息"
3. 引用信息来源格式：【来源：文件名】
4. 用中文回答，语言专业、清晰
5. 涉及具体数据时，引用原文中的数字"""


def load_index():
    if not cfg.INDEX_JSON.exists() or not cfg.INDEX_NPY.exists():
        print('错误: 索引文件不存在，请先运行 rag_ingest.py')
        sys.exit(1)

    with open(cfg.INDEX_JSON, encoding='utf-8') as f:
        chunks = json.load(f)
    embeddings = np.load(str(cfg.INDEX_NPY))
    print(f'索引已加载: {len(chunks)} 个块, 向量维度 {embeddings.shape[1]}')
    return chunks, embeddings


def get_embedding(text, client):
    resp = client.post(
        '/embeddings',
        json={'model': cfg.EMBEDDING_MODEL, 'input': text},
    )
    resp.raise_for_status()
    return resp.json()['data'][0]['embedding']


def search(query_vec, embeddings, chunks, top_k=5):
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    emb_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_normed = embeddings / (emb_norms + 1e-10)

    scores = np.dot(emb_normed, query_norm)
    top_idx = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_idx:
        results.append({
            'text': chunks[idx]['text'],
            'source': chunks[idx]['source'],
            'section': chunks[idx]['section'],
            'score': float(scores[idx]),
        })
    return results


def build_prompt(query, results):
    parts = []
    for i, r in enumerate(results):
        header = f'--- 参考 {i+1} ---'
        src = f'【来源：{r["source"]}】' if not r['section'] else f'【来源：{r["source"]} - {r["section"]}】'
        parts.append(f'{header}\n{src}\n{r["text"]}')

    context = '\n\n'.join(parts)
    return [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': f'参考文档内容：\n\n{context}\n\n---\n\n问题：{query}'},
    ]


def generate(messages, client):
    resp = client.post(
        '/chat/completions',
        json={
            'model': cfg.LLM_MODEL,
            'messages': messages,
            'temperature': cfg.TEMPERATURE,
            'max_tokens': cfg.MAX_TOKENS,
        },
    )
    resp.raise_for_status()
    return resp.json()['choices'][0]['message']['content']


def interactive_loop(chunks, embeddings, client):
    print('\n输入问题直接查询，输入 quit 退出\n')
    while True:
        try:
            q = input('>>> ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not q:
            continue
        if q.lower() in ('quit', 'exit', 'q'):
            break

        print('检索中…')
        q_vec = get_embedding(q, client)
        results = search(q_vec, embeddings, chunks, cfg.TOP_K)

        print(f'找到 {len(results)} 个相关段落:')
        for r in results:
            preview = r['text'][:60].replace('\n', ' ')
            print(f'  [{r["score"]:.3f}] {r["source"]} → {preview}…')

        print('\n生成回答中…')
        answer = generate(build_prompt(q, results), client)
        print(f'\n回答:\n{answer}\n')


def main():
    chunks, embeddings = load_index()

    with httpx.Client(
        base_url=cfg.SILICONFLOW_BASE_URL,
        headers={'Authorization': f'Bearer {cfg.SILICONFLOW_API_KEY}'},
        timeout=120,
    ) as client:
        if len(sys.argv) > 1:
            q = ' '.join(sys.argv[1:])
            print(f'问题: {q}\n')
            q_vec = get_embedding(q, client)
            results = search(q_vec, embeddings, chunks, cfg.TOP_K)
            print(f'检索到 {len(results)} 个相关段落')
            answer = generate(build_prompt(q, results), client)
            print(f'\n回答:\n{answer}')
        else:
            interactive_loop(chunks, embeddings, client)


if __name__ == '__main__':
    main()
