#!/usr/bin/env python3
"""Migrate existing .pkl files to pgvector database.

This script helps migrate face embeddings from pickle files to pgvector database.
It can migrate a single client or all clients at once.

Usage:
    python scripts/migrate_pkl_to_pgvector.py --client dev
    python scripts/migrate_pkl_to_pgvector.py --all
    python scripts/migrate_pkl_to_pgvector.py --client dev --dry-run
"""

import os
import sys
import pickle
import argparse
from pathlib import Path
from loguru import logger

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.face_recognition.storage.pgvector_store import PgVectorStore


def migrate_client(client_slug: str, pkl_path: str, dry_run: bool = False) -> dict:
    """Migrate one client's pickle file to pgvector.

    Args:
        client_slug: Organization slug (e.g., 'humblebee', 'dev')
        pkl_path: Path to pickle file
        dry_run: If True, only show what would be migrated without actually doing it

    Returns:
        dict: Migration results with counts
    """
    logger.info(f"{'[DRY RUN] ' if dry_run else ''}Migrating {client_slug} from {pkl_path}")

    results = {
        'client_slug': client_slug,
        'pkl_path': pkl_path,
        'total_embeddings': 0,
        'migrated': 0,
        'failed': 0,
        'errors': []
    }

    try:
        # Load pickle file
        logger.info(f"Loading pickle file: {pkl_path}")
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        embeddings = data['embeddings']
        names = data['names']
        results['total_embeddings'] = len(embeddings)

        logger.info(f"Found {len(embeddings)} embeddings for {len(set(names))} unique users")

        if dry_run:
            logger.info("[DRY RUN] Would migrate these embeddings:")
            for i, name in enumerate(names[:5]):  # Show first 5
                logger.info(f"  - {name} (embedding shape: {embeddings[i].shape})")
            if len(names) > 5:
                logger.info(f"  ... and {len(names) - 5} more")
            return results

        # Initialize pgvector store
        logger.info(f"Initializing pgvector store for {client_slug}")
        store = PgVectorStore(client_slug)

        # Clear existing embeddings (optional - uncomment if needed)
        # existing_count = store.clear_all_embeddings()
        # logger.info(f"Cleared {existing_count} existing embeddings")

        # Migrate each embedding
        for i, (embedding, name) in enumerate(zip(embeddings, names)):
            try:
                # Extract user info from name
                # Assuming format: "UserName_id" or just "UserName"
                parts = name.split('_')
                user_name = parts[0]
                user_id = parts[1] if len(parts) > 1 else parts[0]

                store.add_embedding(
                    user_id=user_id,
                    user_name=user_name,
                    image_url=f"migrated_from_pkl_{i}",  # Placeholder
                    embedding=embedding,
                    metadata={'migrated_from_pkl': True, 'original_index': i}
                )

                results['migrated'] += 1

                if (i + 1) % 10 == 0:
                    logger.info(f"Migrated {i + 1}/{len(embeddings)} embeddings...")

            except Exception as e:
                logger.error(f"Failed to migrate embedding {i} ({name}): {e}")
                results['failed'] += 1
                results['errors'].append(f"{name}: {str(e)}")

        logger.success(
            f"✅ Migration complete: {results['migrated']}/{results['total_embeddings']} embeddings migrated"
        )

        if results['failed'] > 0:
            logger.warning(f"⚠️  {results['failed']} embeddings failed to migrate")

    except Exception as e:
        logger.error(f"❌ Failed to migrate {client_slug}: {e}")
        results['errors'].append(str(e))

    return results


def find_all_pkl_files(base_path: str) -> list:
    """Find all .pkl files in the storage directory.

    Args:
        base_path: Base path to search for pkl files

    Returns:
        List of tuples: (client_slug, pkl_path)
    """
    pkl_files = []

    base_dir = Path(base_path)
    if not base_dir.exists():
        logger.warning(f"Base path does not exist: {base_path}")
        return pkl_files

    # Search for main.pkl files in subdirectories
    for client_dir in base_dir.iterdir():
        if client_dir.is_dir():
            pkl_file = client_dir / 'main.pkl'

            if pkl_file.exists():
                client_slug = client_dir.name
                pkl_files.append((client_slug, str(pkl_file)))
                logger.info(f"Found: {client_slug} -> {pkl_file}")

    return pkl_files


def main():
    """Main migration function."""
    parser = argparse.ArgumentParser(description='Migrate .pkl files to pgvector database')

    parser.add_argument(
        '--client',
        type=str,
        help='Client slug to migrate (e.g., dev, humblebee)'
    )

    parser.add_argument(
        '--pkl-path',
        type=str,
        help='Path to specific .pkl file (optional, auto-detected if not provided)'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Migrate all clients found in storage directory'
    )

    parser.add_argument(
        '--base-path',
        type=str,
        default='/app/volumes/src/embeddings/main.pkl',
        help='Base path for storage (default: /app/volumes/src/embeddings/main.pkl)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be migrated without actually doing it'
    )

    args = parser.parse_args()

    # Configure logger
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )

    # Validate arguments
    if not args.client and not args.all:
        logger.error("❌ Must specify either --client or --all")
        parser.print_help()
        return 1

    if args.client and args.all:
        logger.error("❌ Cannot specify both --client and --all")
        return 1

    all_results = []

    try:
        if args.client:
            # Migrate single client
            if args.pkl_path:
                pkl_path = args.pkl_path
            else:
                # Auto-detect pkl path
                fr_slug = os.getenv('FR_SLUG', 'face-recognition')
                pkl_path = f'{args.base_path}/{args.client}/main.pkl'

            if not os.path.exists(pkl_path):
                logger.error(f"❌ Pickle file not found: {pkl_path}")
                return 1

            result = migrate_client(args.client, pkl_path, dry_run=args.dry_run)
            all_results.append(result)

        elif args.all:
            # Migrate all clients
            pkl_files = find_all_pkl_files(args.base_path)

            if not pkl_files:
                logger.warning(f"No .pkl files found in {args.base_path}")
                return 0

            logger.info(f"Found {len(pkl_files)} client(s) to migrate")

            for client_slug, pkl_path in pkl_files:
                result = migrate_client(client_slug, pkl_path, dry_run=args.dry_run)
                all_results.append(result)
                print()  # Blank line between clients

        # Print summary
        print("\n" + "=" * 70)
        print("MIGRATION SUMMARY")
        print("=" * 70)

        total_migrated = sum(r['migrated'] for r in all_results)
        total_failed = sum(r['failed'] for r in all_results)
        total_embeddings = sum(r['total_embeddings'] for r in all_results)

        for result in all_results:
            status = "✅" if result['failed'] == 0 else "⚠️"
            print(f"{status} {result['client_slug']}: {result['migrated']}/{result['total_embeddings']} migrated")

        print(f"\nTotal: {total_migrated}/{total_embeddings} embeddings migrated")

        if total_failed > 0:
            print(f"⚠️  {total_failed} embeddings failed")

        if args.dry_run:
            print("\n[DRY RUN] No changes were made")

        return 0

    except Exception as e:
        logger.exception(f"❌ Migration failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
