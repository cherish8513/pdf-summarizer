import os
from typing import Optional, Tuple
from dotenv import load_dotenv


class ClientConfig:
    _instance: Optional['ClientConfig'] = None
    _initialized: bool = False

    def __new__(cls) -> 'ClientConfig':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._load_config()
            ClientConfig._initialized = True

    def _load_config(self):
        """환경 변수 로드 및 설정 초기화"""
        load_dotenv()

        # LLM 설정
        self.groq_api_key: str = os.getenv('GROQ_API_KEY', '')
        self.groq_api_base: str = os.getenv('GORQ_API_BASE', 'https://api.groq.com/openai/v1')
        self.llm_model: str = os.getenv('LLM_MODEL', '')
        self.llm_temperature: float = self._get_float_env('LLM_TEMPERATURE', 0.2)
        self.llm_max_tokens: int = self._get_int_env('LLM_MAX_TOKENS', 100)

        # 임베딩 모델 설정
        self.embedding_model: str = os.getenv('EMBEDDING_MODEL', '')

        # 텍스트 처리 설정
        self.chunk_size: int = self._get_int_env('CHUNK_SIZE', 1000)
        self.chunk_overlap: int = self._get_int_env('CHUNK_OVERLAP', 50)
        self.max_tfidf_features: int = self._get_int_env('MAX_TFIDF_FEATURES', 5000)
        self.ngram_range: Tuple[int, int] = self._parse_ngram_range(os.getenv('NGRAM_RANGE', '1,2'))
        self.min_df: int = self._get_int_env('MIN_DF', 1)
        self.max_df: float = self._get_float_env('MAX_DF', 0.9)

        # 청킹 조정 계수
        self.long_sentence_multiplier: float = self._get_float_env('LONG_SENTENCE_MULTIPLIER', 1.2)
        self.short_sentence_multiplier: float = self._get_float_env('SHORT_SENTENCE_MULTIPLIER', 0.8)
        self.long_sentence_threshold: int = self._get_int_env('LONG_SENTENCE_THRESHOLD', 100)
        self.short_sentence_threshold: int = self._get_int_env('SHORT_SENTENCE_THRESHOLD', 50)

        # 검색 설정
        self.default_k: int = self._get_int_env('DEFAULT_K', 4)
        self.search_type: str = os.getenv('SEARCH_TYPE', 'similarity')
        self.search_multiplier: int = self._get_int_env('SEARCH_MULTIPLIER', 2)
        self.semantic_weight: float = self._get_float_env('SEMANTIC_WEIGHT', 0.7)
        self.min_chunk_length: int = self._get_int_env('MIN_CHUNK_LENGTH', 30)
        self.min_similarity_threshold: float = self._get_float_env('MIN_SIMILARITY_THRESHOLD', 0.0)

        # 메모리 관리
        self.max_documents: int = self._get_int_env('MAX_DOCUMENTS', 10)

    def _get_bool_env(self, key: str, default: bool) -> bool:
        """환경 변수를 boolean으로 변환"""
        value = os.getenv(key, str(default)).lower()
        return value in ('true', '1', 'yes', 'on')

    def _get_int_env(self, key: str, default: int) -> int:
        """환경 변수를 정수로 변환"""
        try:
            return int(os.getenv(key, str(default)))
        except ValueError:
            return default

    def _get_float_env(self, key: str, default: float) -> float:
        """환경 변수를 실수로 변환"""
        try:
            return float(os.getenv(key, str(default)))
        except ValueError:
            return default

    def _parse_ngram_range(self, value: str) -> Tuple[int, int]:
        """NGRAM_RANGE 값을 파싱하여 튜플로 변환"""
        try:
            parts = value.split(',')
            if len(parts) == 2:
                return int(parts[0].strip()), int(parts[1].strip())
            else:
                return 1, 2
        except (ValueError, IndexError):
            return 1, 2

    def validate_config(self) -> bool:
        """필수 설정값들이 제대로 설정되었는지 검증"""
        required_keys = [
            'groq_api_key',
            'langsmith_api_key'
        ]

        missing_keys = []
        for key in required_keys:
            if not getattr(self, key, None):
                missing_keys.append(key.upper())

        if missing_keys:
            print(f"경고: 다음 필수 환경 변수가 설정되지 않았습니다: {', '.join(missing_keys)}")
            return False

        return True

    def __repr__(self) -> str:
        """설정 정보를 문자열로 반환 (민감한 정보는 마스킹)"""
        return f"""ClientConfig(
    langsmith_tracing={self.langsmith_tracing},
    langsmith_endpoint={self.langsmith_endpoint},
    langsmith_project={self.langsmith_project},
    embedding_model={self.embedding_model},
    max_tfidf_features={self.max_tfidf_features},
    ngram_range={self.ngram_range},
    default_k={self.default_k},
    max_documents={self.max_documents}
)"""


config = ClientConfig()