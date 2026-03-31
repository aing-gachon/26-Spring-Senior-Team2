import os
import random
import numpy as np
import torch
import yaml
import logging

def set_seed(seed: int = 42):
    """
    고정된 시드(Seed)를 설정하여 실험의 재현성을 보장합니다.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_config(config_path: str) -> dict:
    """
    YAML 설정 파일을 로드합니다.
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def setup_logger(name: str, log_file: str, level=logging.INFO):
    """
    터미널 출력 및 파일 저장을 동시 지원하는 로거를 생성합니다.
    """
    # 디렉터리가 없다면 생성
    if os.path.dirname(log_file):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    formatter = logging.Formatter('%(asctime)s %(levelname)s [%(name)s] %(message)s')
    
    handler = logging.FileHandler(log_file, encoding='utf-8')
    handler.setFormatter(formatter)
    
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 중복 로그 핸들러 방지
    if not logger.handlers:
        logger.addHandler(handler)
        logger.addHandler(console)
    
    return logger
