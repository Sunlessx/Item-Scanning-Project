from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    typesense_host: str = ""
    typesense_port: int = 443
    typesense_protocol: str = "https"
    typesense_api_key: str = ""
    typesense_collection: str = "products"

    index_path: str = "data/embeddings.index"
    metadata_path: str = "data/metadata.pkl"

    clip_model_name: str = "openai/clip-vit-base-patch32"
    clip_device: str = "cpu"

    search_top_k: int = 20
    max_upload_bytes: int = 10_485_760  # 10 MB

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
