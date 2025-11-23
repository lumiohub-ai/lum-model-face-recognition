"""Input validation utilities for database operations.

This module provides security validation functions to prevent SQL injection
and other security vulnerabilities.
"""

import re
from typing import Optional


def validate_client_slug(slug: str) -> str:
    """Validate client slug to prevent SQL injection.

    Args:
        slug: Client slug to validate

    Returns:
        str: Validated slug

    Raises:
        ValueError: If slug contains invalid characters

    Note:
        Only allows lowercase letters, numbers, and underscores.
        This prevents SQL injection when using slugs in table/schema names.
    """
    if not slug:
        raise ValueError("client_slug cannot be empty")

    # Only allow lowercase alphanumeric and underscores
    if not re.match(r'^[a-z0-9_]+$', slug):
        raise ValueError(
            f"Invalid client_slug: '{slug}'. "
            "Only lowercase letters, numbers, and underscores are allowed."
        )

    # Additional safety: limit length
    if len(slug) > 63:  # PostgreSQL identifier limit
        raise ValueError(f"client_slug too long: {len(slug)} chars (max 63)")

    return slug


def validate_schema_name(schema_name: str) -> str:
    """Validate schema name to prevent SQL injection.

    Args:
        schema_name: Schema name to validate

    Returns:
        str: Validated schema name

    Raises:
        ValueError: If schema name contains invalid characters
    """
    if not schema_name:
        raise ValueError("schema_name cannot be empty")

    # Allow alphanumeric and underscores
    if not re.match(r'^[a-z0-9_]+$', schema_name):
        raise ValueError(
            f"Invalid schema_name: '{schema_name}'. "
            "Only lowercase letters, numbers, and underscores are allowed."
        )

    if len(schema_name) > 63:
        raise ValueError(f"schema_name too long: {len(schema_name)} chars (max 63)")

    return schema_name


def sanitize_identifier(identifier: str, max_length: int = 63) -> str:
    """Sanitize a SQL identifier (table, column, schema name).

    Args:
        identifier: SQL identifier to sanitize
        max_length: Maximum allowed length (default: 63 for PostgreSQL)

    Returns:
        str: Sanitized identifier

    Raises:
        ValueError: If identifier is invalid after sanitization
    """
    if not identifier:
        raise ValueError("Identifier cannot be empty")

    # Remove any non-alphanumeric characters except underscore
    sanitized = re.sub(r'[^a-z0-9_]', '', identifier.lower())

    if not sanitized:
        raise ValueError(f"Identifier '{identifier}' contains no valid characters")

    if len(sanitized) > max_length:
        raise ValueError(f"Identifier too long: {len(sanitized)} chars (max {max_length})")

    return sanitized
