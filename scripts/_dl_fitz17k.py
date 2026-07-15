import sys
sys.path.insert(0, 'src')
from loguru import logger
from mnkai.data.download.fitzpatrick17k import download_dataset_hf
from pathlib import Path

logger.info('Starting Fitzpatrick17k HF download...')
stats = download_dataset_hf(
    image_root=r'D:\AIFINAL\images\fitz17k',
    meta_out='data/manifests/fitz17k_hf.csv'
)
logger.info(f'Done: {stats}')
print(f'RESULT: {stats}')
