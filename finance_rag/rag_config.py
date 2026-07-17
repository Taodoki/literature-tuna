import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

# 加载 .env
env_path = PROJECT_ROOT / '.env'
if env_path.exists():
    with open(env_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, val = line.split('=', 1)
                os.environ.setdefault(key.strip(), val.strip())

SILICONFLOW_API_KEY = os.environ.get('SILICONFLOW_API_KEY')
if not SILICONFLOW_API_KEY:
    print('错误: 未找到 SILICONFLOW_API_KEY，请在 .env 文件中设置')
    print(f'预期路径: {env_path}')
    sys.exit(1)

SILICONFLOW_BASE_URL = 'https://api.siliconflow.cn/v1'

# 模型
EMBEDDING_MODEL = 'BAAI/bge-large-zh-v1.5'
LLM_MODEL = 'deepseek-ai/DeepSeek-V3'

# 路径
DOC_DIR = PROJECT_ROOT / 'docs'
INDEX_JSON = PROJECT_ROOT / 'rag_index.json'
INDEX_NPY = PROJECT_ROOT / 'rag_index.npy'

# 分块参数
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# 检索参数
TOP_K = 5

# 生成参数
TEMPERATURE = 0.3
MAX_TOKENS = 1024

# 嵌入参数
EMBEDDING_BATCH_SIZE = 16
EMBEDDING_DIM = 1024
