#!/usr/bin/env python3
"""Migrate existing embeddings to populate image_url_norm field.

This script normalizes all existing image URLs in the database and populates
the image_url_norm column to enable deduplication on restart.

Run this ONCE after upgrading to the URL normalization feature.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from face_recognition.storage.pgvector_store import PgVectorStore
from face_recognition.storage.url_utils import normalize_image_url
from sqlalchemy import text
from loguru import logger

def migrate_normalize_urls(client_slug: str = None):
    """Populate image_url_norm for all existing embeddings."""

    if not client_slug:
        client_slug = os.getenv('CLIENT_SLUG', 'dev')

    logger.info("=" * 80)
    logger.info(f"MIGRATING IMAGE URLS FOR: {client_slug}")
    logger.info("=" * 80)

    store = PgVectorStore(client_slug)

    # Get all embeddings without normalized URLs
    with store.db_config.get_connection() as conn:
        result = conn.execute(text(f"""
            SELECT id, image_url
            FROM {store.schema_name}.face_embeddings
            WHERE image_url_norm IS NULL
        """))

        rows = result.fetchall()
        total = len(rows)

        if total == 0:
            logger.info("✅ No embeddings need migration - all URLs already normalized")
            return

        logger.info(f"Found {total} embeddings to migrate")

        # Update each embedding with normalized URL
        updated = 0
        for idx, (embedding_id, image_url) in enumerate(rows, 1):
            try:
                image_url_norm = normalize_image_url(image_url)

                conn.execute(text(f"""
                    UPDATE {store.schema_name}.face_embeddings
                    SET image_url_norm = :image_url_norm
                    WHERE id = :id
                """), {
                    'id': embedding_id,
                    'image_url_norm': image_url_norm
                })

                updated += 1

                if idx % 100 == 0:
                    logger.info(f"Progress: {idx}/{total} ({idx/total*100:.1f}%)")

            except Exception as e:
                logger.error(f"Failed to normalize URL for embedding {embedding_id}: {e}")

        conn.commit()

        logger.info("=" * 80)
        logger.info(f"✅ Migration complete: {updated}/{total} URLs normalized")
        logger.info("=" * 80)

        # Check for potential duplicates after normalization
        logger.info("Checking for duplicates...")
        result = conn.execute(text(f"""
            SELECT user_id, image_url_norm, COUNT(*) as count
            FROM {store.schema_name}.face_embeddings
            WHERE image_url_norm IS NOT NULL
            GROUP BY user_id, image_url_norm
            HAVING COUNT(*) > 1
        """))

        duplicates = result.fetchall()

        if duplicates:
            logger.warning(f"⚠️  Found {len(duplicates)} duplicate (user_id, url) pairs:")
            for user_id, url_norm, count in duplicates[:10]:  # Show first 10
                logger.warning(f"  - User {user_id}: {url_norm} ({count} copies)")

            logger.warning("These duplicates will be automatically ignored on next restart")
            logger.warning("Consider cleaning up manually: Keep best quality, delete others")
        else:
            logger.info("✅ No duplicates detected - database is clean")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Migrate image URLs to normalized format")
    parser.add_argument("--client-slug", help="Client organization slug (default: from CLIENT_SLUG env)")
    args = parser.parse_args()

    migrate_normalize_urls(client_slug=args.client_slug)
