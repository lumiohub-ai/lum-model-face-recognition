"""Database configuration and connection management for pgvector."""

import os
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
from loguru import logger

from .validators import validate_client_slug, validate_schema_name


class DatabaseConfig:
    """PostgreSQL database configuration with pgvector support."""

    def __init__(self):
        """Initialize database configuration from environment variables.

        Raises:
            ValueError: If required environment variables are not set
        """
        self.host = os.getenv('POSTGRES_HOST', 'localhost')
        self.port = int(os.getenv('POSTGRES_PORT', 5433))
        self.user = os.getenv('POSTGRES_USER', 'face_recognition')

        # SECURITY: Require password to be explicitly set (no default)
        self.password = os.getenv('POSTGRES_PASSWORD')
        if not self.password:
            raise ValueError(
                "POSTGRES_PASSWORD environment variable is required. "
                "Please set a secure password in your environment."
            )

        self.database = os.getenv('POSTGRES_DB', 'face_embeddings')

        # Build connection string
        self.connection_string = (
            f"postgresql://{self.user}:{self.password}@"
            f"{self.host}:{self.port}/{self.database}"
        )

        # Create engine with connection pooling
        self.engine = create_engine(
            self.connection_string,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,  # Verify connections before use
            pool_recycle=3600,   # Recycle connections after 1 hour
        )

        logger.info(f"Database configured: {self.host}:{self.port}/{self.database}")

    def test_connection(self) -> bool:
        """Test database connection.

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Database connection test successful")
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False

    def init_schema(self, client_slug: str) -> None:
        """Create schema and tables for a client if not exists.

        Args:
            client_slug: Organization slug (e.g., 'humblebee', 'dev')

        Raises:
            ValueError: If client_slug contains invalid characters
        """
        # SECURITY: Validate client_slug to prevent SQL injection
        validated_slug = validate_client_slug(client_slug)
        schema_name = f"org_{validated_slug}"

        # Double-check the schema name itself
        validate_schema_name(schema_name)

        try:
            with self.engine.begin() as conn:
                # Create schema
                conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema_name}"))
                logger.info(f"Schema ensured: {schema_name}")

                # Enable pgvector extension
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                logger.info("pgvector extension enabled")

                # Create face_embeddings table
                conn.execute(text(f"""
                    CREATE TABLE IF NOT EXISTS {schema_name}.face_embeddings (
                        id SERIAL PRIMARY KEY,
                        user_id VARCHAR(50) NOT NULL,
                        user_name VARCHAR(255) NOT NULL,
                        external_id VARCHAR(100),
                        image_url TEXT NOT NULL,
                        embedding VECTOR(512) NOT NULL,
                        embedding_metadata JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                logger.info(f"Table created: {schema_name}.face_embeddings")

                # Create index on user_id for fast lookups
                conn.execute(text(f"""
                    CREATE INDEX IF NOT EXISTS idx_{schema_name}_user_id
                    ON {schema_name}.face_embeddings(user_id)
                """))

                # Create vector similarity index (ivfflat)
                # Note: This requires some data to be inserted first for optimal performance
                conn.execute(text(f"""
                    CREATE INDEX IF NOT EXISTS idx_{schema_name}_embedding_cosine
                    ON {schema_name}.face_embeddings
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                """))
                logger.info(f"Indexes created for {schema_name}")

                logger.info(f"✅ Schema initialization complete: {schema_name}")

        except Exception as e:
            logger.error(f"Failed to initialize schema {schema_name}: {e}")
            raise

    def drop_schema(self, client_slug: str, cascade: bool = True) -> None:
        """Drop schema for a client.

        Args:
            client_slug: Organization slug
            cascade: If True, drop all objects in schema

        Raises:
            ValueError: If client_slug contains invalid characters

        Warning:
            This will delete all data for the client!
        """
        # SECURITY: Validate client_slug to prevent SQL injection
        validated_slug = validate_client_slug(client_slug)
        schema_name = f"org_{validated_slug}"

        # Double-check the schema name itself
        validate_schema_name(schema_name)

        try:
            with self.engine.begin() as conn:
                cascade_str = "CASCADE" if cascade else ""
                conn.execute(text(f"DROP SCHEMA IF EXISTS {schema_name} {cascade_str}"))
                logger.warning(f"Schema dropped: {schema_name}")

        except Exception as e:
            logger.error(f"Failed to drop schema {schema_name}: {e}")
            raise

    @contextmanager
    def get_connection(self):
        """Context manager for database connections.

        Usage:
            with db_config.get_connection() as conn:
                result = conn.execute(text("SELECT ..."))
        """
        conn = self.engine.connect()
        try:
            yield conn
        finally:
            conn.close()

    def close(self) -> None:
        """Close all database connections and dispose of engine."""
        if self.engine:
            self.engine.dispose()
            logger.info("Database engine disposed")
