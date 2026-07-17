import re
from pathlib import Path


def extract_text(filepath):
    ext = Path(filepath).suffix.lower()
    if ext == '.docx':
        return _extract_docx(filepath)
    elif ext == '.pdf':
        return _extract_pdf(filepath)
    elif ext == '.txt':
        return _extract_txt(filepath)
    else:
        raise ValueError(f'不支持的文件格式: {ext}')


def _extract_docx(filepath):
    from docx import Document
    doc = Document(filepath)
    paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    # 段落之间双换行，方便下游按段落分块
    return '\n\n'.join(paras)


def _extract_pdf(filepath):
    from pypdf import PdfReader
    reader = PdfReader(filepath)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text.strip():
            pages.append(text)
    return '\n'.join(pages)


def _extract_txt(filepath):
    with open(filepath, encoding='utf-8') as f:
        return f.read()


def smart_chunk_financial(text, chunk_size=500, overlap=50, source=''):
    """中文金融文档智能分块：按「一、二、三、」章节标题分割后，长节按段落再切"""
    section_pat = re.compile(r'([一二三四五六七八九十]+[、．\.])', re.MULTILINE)
    parts = section_pat.split(text)

    # 没有章节标记则直接按段落切
    if len(parts) <= 1:
        return _chunk_by_paragraph(text, chunk_size, overlap, source)

    chunks = []

    # preamble——第一个章节标题之前的内容
    preamble = parts[0].strip()
    if preamble:
        if len(preamble) <= chunk_size:
            chunks.append({'text': preamble, 'source': source, 'section': ''})
        else:
            chunks.extend(_chunk_by_paragraph(preamble, chunk_size, overlap, source))

    # 章节标题与内容交替：title, content, title, content, ...
    for i in range(1, len(parts), 2):
        title = parts[i].strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ''
        full = f'{title}\n{content}'

        if len(full) <= chunk_size:
            chunks.append({'text': full, 'source': source, 'section': title})
        else:
            subs = _chunk_by_paragraph(content, chunk_size, overlap, source)
            for s in subs:
                s['section'] = title
            chunks.extend(subs)

    return chunks


def _chunk_by_paragraph(text, chunk_size=500, overlap=50, source=''):
    """按段落合并/拆分文本块，长文本按句子边界强制切分"""
    # 清洗控制字符
    text = text.replace('\xa0', ' ')

    paragraphs = re.split(r'\n\s*\n', text)
    merged = []
    current = ''

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) + 1 < chunk_size:
            current = (current + '\n' + para).strip()
        else:
            if current:
                merged.append(current)
            current = para

    if current:
        merged.append(current)

    # 强制切分过长的块（嵌入 API 有长度限制）
    MAX_CHAR = 800
    final = []
    for m in merged:
        if len(m) <= MAX_CHAR:
            final.append(m)
        else:
            final.extend(_force_split(m, MAX_CHAR, overlap))

    return [{'text': t, 'source': source, 'section': ''} for t in final]


def _force_split(text, max_len=800, overlap=50):
    """在句子边界硬切长文本"""
    # 中文/英文句子结束符
    parts = re.split(r'(?<=[。！？；\n!?;])\s*', text)
    chunks = []
    current = ''
    for part in parts:
        if not part.strip():
            continue
        if len(current) + len(part) < max_len:
            current += part
        else:
            if current:
                chunks.append(current.strip())
            current = part
    if current:
        chunks.append(current.strip())

    # 如果还是太长（极端情况：一句超长），按字符切
    if len(chunks) == 1 and len(chunks[0]) > max_len:
        chunks = [text[i:i + max_len] for i in range(0, len(text), max_len - overlap)]

    return chunks
