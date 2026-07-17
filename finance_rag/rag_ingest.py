import json
import sys
import time
from pathlib import Path

import httpx
import numpy as np

import rag_config as cfg
from rag_utils import extract_text, smart_chunk_financial

sys.stdout.reconfigure(encoding='utf-8')


def scan_documents(doc_dir):
    supported = {'.docx', '.pdf', '.txt'}
    files = []
    for p in Path(doc_dir).iterdir():
        if p.suffix.lower() in supported and not p.name.startswith('~'):
            files.append(str(p))
    return sorted(files)


def get_embeddings_batch(texts, client):
    all_emb = []
    for i in range(0, len(texts), cfg.EMBEDDING_BATCH_SIZE):
        batch = texts[i:i + cfg.EMBEDDING_BATCH_SIZE]
        resp = client.post(
            '/embeddings',
            json={'model': cfg.EMBEDDING_MODEL, 'input': batch},
        )
        resp.raise_for_status()
        data = resp.json()['data']
        data.sort(key=lambda x: x['index'])
        all_emb.extend([d['embedding'] for d in data])

        done = min(i + cfg.EMBEDDING_BATCH_SIZE, len(texts))
        print(f'  嵌入进度: {done}/{len(texts)}')
        if done < len(texts):
            time.sleep(0.5)

    return np.array(all_emb, dtype=np.float32)


def main():
    print('=== 金融 RAG 索引构建 ===\n')

    doc_dir = cfg.DOC_DIR
    if not doc_dir.exists():
        print(f'错误: 文档目录不存在: {doc_dir}')
        return

    files = scan_documents(doc_dir)
    if not files:
        print(f'错误: {doc_dir} 中没有受支持的文件')
        return

    print(f'找到 {len(files)} 个文档:')
    for f in files:
        print(f'  {Path(f).name}')
    print()

    # 提取 + 分块
    all_chunks = []
    for fp in files:
        name = Path(fp).name
        print(f'处理: {name}')
        try:
            text = extract_text(fp)
            chunks = smart_chunk_financial(text, cfg.CHUNK_SIZE, cfg.CHUNK_OVERLAP, name)
            all_chunks.extend(chunks)
            print(f'  → {len(chunks)} 个块')
        except Exception as e:
            print(f'  !! 失败: {e}')

    if not all_chunks:
        print('错误: 未能提取出任何文本块')
        return

    print(f'\n总计 {len(all_chunks)} 个文本块\n')

    # 嵌入
    print('获取嵌入向量…')
    texts = [c['text'] for c in all_chunks]
    with httpx.Client(
        base_url=cfg.SILICONFLOW_BASE_URL,
        headers={'Authorization': f'Bearer {cfg.SILICONFLOW_API_KEY}'},
        timeout=120,
    ) as client:
        embeddings = get_embeddings_batch(texts, client)

    print(f'嵌入矩阵形状: {embeddings.shape}\n')

    # 保存
    meta = [{'text': c['text'], 'source': c['source'], 'section': c['section']} for c in all_chunks]
    with open(cfg.INDEX_JSON, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    np.save(str(cfg.INDEX_NPY), embeddings)

    print(f'索引已保存:')
    print(f'  元数据: {cfg.INDEX_JSON}')
    print(f'  向量:   {cfg.INDEX_NPY}')

    sources = set(c['source'] for c in all_chunks)
    print(f'\n完成! 共 {len(all_chunks)} 个块，来自 {len(sources)} 个文档')


if __name__ == '__main__':
    main()
