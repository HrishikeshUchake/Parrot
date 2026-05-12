"""Privacy and data sanitization service for protecting sensitive information.

This module handles anonymization and obfuscation of content before sending to
external LLM services. It provides a flexible interface to replace or redact
sensitive information in social media posts.

Uses Presidio (https://microsoft.github.io/presidio/) for PII detection and anonymization.
"""
from __future__ import annotations
import logging
import re
from dataclasses import dataclass
from typing import Optional

try:
    from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class SanitizationConfig:
    """Configuration for content sanitization."""

    # Enable/disable specific sanitization types
    anonymize_usernames: bool = True
    anonymize_emails: bool = True
    redact_phone_numbers: bool = True
    redact_urls: bool = True
    redact_personal_identifiers: bool = True

    # Replacement strategies
    username_replacement: str = "[USER]"
    email_replacement: str = "[EMAIL]"
    phone_replacement: str = "[PHONE]"
    url_replacement: str = "[URL]"
    identifier_replacement: str = "[ID]"


class ContentPrivatizer:
    """Handles content privatization and anonymization.

    This class sanitizes post content and metadata before sending to remote LLMs.
    It can be extended with custom sanitization rules.
    """

    def __init__(self, config: Optional[SanitizationConfig] = None):
        """Initialize the privatizer with optional configuration.

        Args:
            config: SanitizationConfig object. Uses defaults if not provided.
        """
        self.config = config or SanitizationConfig()

    def privatize_content(self, content: str) -> str:
        """Sanitize post content.

        Applies configured sanitization rules to remove or replace sensitive
        information in the post text.

        Args:
            content: The post content string to sanitize.

        Returns:
            Sanitized content string.
        """
        # TODO: Implement content sanitization logic
        # This is where you'll add your custom privatization rules
        raise NotImplementedError("Subclass must implement privatize_content()")

    def privatize_author(self, username: str, acct: str) -> tuple[str, str]:
        """Sanitize author information.

        Args:
            username: The account username
            acct: The account acct (email-like identifier)

        Returns:
            Tuple of (sanitized_username, sanitized_acct)
        """
        # TODO: Implement author sanitization
        raise NotImplementedError("Subclass must implement privatize_author()")

    def privatize_tags(self, tags: list[str]) -> list[str]:
        """Sanitize tags/hashtags.

        Args:
            tags: List of tag strings

        Returns:
            List of sanitized tagsOUTPUT
        """
        # TODO: Implement tag sanitization
        raise NotImplementedError("Subclass must implement privatize_tags()")

    def privatize_context(self, context: str) -> str:
        """Sanitize formatted context string for LLM.

        This is the main entry point - takes the full formatted context
        (as produced by synthesis._format_context) and applies all
        privatization rules.

        Args:
            context: The formatted context string with multiple posts

        Returns:
            Fully sanitized context ready for LLM
        """
        # TODO: Implement context-level sanitization
        raise NotImplementedError("Subclass must implement privatize_context()")

    def add_known_entities(self, entities: list[str]) -> None:
        """Dynamically add specific known entities (like un-@'d usernames) to ensure they are caught."""
        pass

class NoOpPrivatizer(ContentPrivatizer):
    """Pass-through privatizer that returns content unchanged.

    Useful for development/testing when no sanitization is needed.
    """

    def privatize_content(self, content: str) -> str:
        """Return content unchanged."""
        return content

    def privatize_author(self, username: str, acct: str) -> tuple[str, str]:
        """Return author info unchanged."""
        return username, acct

    def privatize_tags(self, tags: list[str]) -> list[str]:
        """Return tags unchanged."""
        return tags

    def privatize_context(self, context: str) -> str:
        """Return context unchanged."""
        return context


class PresidioPrivatizer(ContentPrivatizer):
    """Presidio-based PII anonymizer with support for restoration.

    Uses Microsoft Presidio to detect and replace PII (emails, phone numbers,
    names, etc.) with consistent tokens. Maintains a mapping of tokens to
    original values for later restoration.

    Tokens are consistent across calls - the same PII value maps to the same
    token (e.g., "email@example.com" always becomes [EMAIL_ADDRESS_1]).

    Example:
        >>> privatizer = PresidioPrivatizer()
        >>> result = privatizer.anonymize("My name is John and email is john@example.com")
        >>> print(result['anonymized'])
        'My name is [PERSON_1] and email is [EMAIL_ADDRESS_1]'
        >>> restored = privatizer.restore("[PERSON_1] sent an email to [EMAIL_ADDRESS_1]")
        >>> print(restored)
        'John sent an email to john@example.com'
    """

    def __init__(self, config: Optional[SanitizationConfig] = None):
        """Initialize Presidio-based privatizer.

        Args:
            config: SanitizationConfig object (some fields are ignored for Presidio)

        Raises:
            ImportError: If presidio-analyzer is not installed
        """
        if not PRESIDIO_AVAILABLE:
            raise ImportError(
                "presidio-analyzer is required for PresidioPrivatizer. "
                "Install with: pip install presidio-analyzer"
            )

        super().__init__(config)
        self.analyzer = AnalyzerEngine()
        
        # Add custom recognizer for social media usernames (@username)
        # We ensure the username doesn't end with a dot or hyphen by requiring a word char at the end
        username_pattern = Pattern(
            name="username_pattern",
            regex=r"(?i)\B@[\w\.-]*[\w]",
            score=0.9
        )
        username_recognizer = PatternRecognizer(
            supported_entity="USERNAME",
            patterns=[username_pattern]
        )
        self.analyzer.registry.add_recognizer(username_recognizer)
        
        self.mappings = {}  # token -> original value
        self.reverse_mappings = {}  # original value -> token (for consistency)
        self.counters = {}  # entity_type -> count

    def add_known_entities(self, entities: list[str]) -> None:
        """Dynamically add specific known entities (like un-@'d usernames) to ensure they are caught."""
        from presidio_analyzer import Pattern, PatternRecognizer
        import re
        
        for e in entities:
            e = e.strip().lstrip("@")
            if not e or len(e) < 3:
                continue
                
            # Create an exact word boundary regex for this entity
            pattern = Pattern(
                name=f"known_entity_{e}",
                regex=rf"(?i)\b{re.escape(e)}\b",
                score=1.0
            )
            # Add it to the analyzer registry as a USERNAME
            recognizer = PatternRecognizer(
                supported_entity="USERNAME",
                patterns=[pattern]
            )
            self.analyzer.registry.add_recognizer(recognizer)

    def anonymize(self, text: str) -> dict:
        """Detect and anonymize PII in text with consistent tokens.

        Each detected entity is replaced with a token like [EMAIL_ADDRESS_1].
        The same entity value always maps to the same token within this session.

        Args:
            text: Original text potentially containing PII

        Returns:
            dict with:
                - original: input text
                - anonymized: text with PII replaced by tokens
                - pii_found: list of detected entities with details
                - mappings: dict mapping tokens to original values
        """
        # Detect PII using Presidio
        # We explicitly specify entities to avoid noisy false positives from the default PERSON recognizer
        # on technical terms like "MLOps" or "React" while still catching explicit USERNAMEs and real PII.
        target_entities = [
            "EMAIL_ADDRESS", 
            "PHONE_NUMBER", 
            "CREDIT_CARD", 
            "IP_ADDRESS", 
            "CRYPTO",
            "IBAN_CODE",
            "US_SSN",
            "US_PASSPORT",
            "USERNAME",
            "DATE_TIME",
            "LOCATION"
        ]
        results = self.analyzer.analyze(text=text, language="en", entities=target_entities)

        if not results:
            return {
                "original": text,
                "anonymized": text,
                "pii_found": [],
                "mappings": {}
            }

        # Sort detections by position to handle correctly
        results_sorted = sorted(results, key=lambda x: x.start)

        out_parts = []
        last = 0
        pii_found = []

        for r in results_sorted:
            # Skip overlapping detections (keep first one)
            if r.start < last:
                continue

            entity_type = r.entity_type
            original_value = text[r.start:r.end]

            # Use a normalized value for checking to ensure case-insensitivity
            normalized_value = original_value.lower()
            
            # Treat "@username" and "username" as the exact same entity 
            if entity_type == "USERNAME" and normalized_value.startswith("@"):
                normalized_value = normalized_value[1:]

            # Check if we've seen this exact value before
            if normalized_value in self.reverse_mappings:
                # Reuse the existing token for consistency
                token = self.reverse_mappings[normalized_value]
            else:
                # Create a new token for this value
                self.counters[entity_type] = self.counters.get(entity_type, 0) + 1
                token = f"[{entity_type}_{self.counters[entity_type]}]"

                # Store the mapping both ways
                self.mappings[token] = original_value
                self.reverse_mappings[normalized_value] = token

            # Append text before entity
            out_parts.append(text[last:r.start])

            # Append token
            out_parts.append(token)

            # Update position pointer
            last = r.end

            pii_found.append({
                "entity_type": entity_type,
                "original_value": original_value,
                "token": token,
                "confidence": r.score
            })

        # Append remaining text
        out_parts.append(text[last:])

        anonymized_text = "".join(out_parts)

        return {
            "original": text,
            "anonymized": anonymized_text,
            "pii_found": pii_found,
            "mappings": self.mappings.copy()
        }

    def restore(self, text: str) -> str:
        """Restore anonymized tokens back to original PII values.

        Replaces all tokens (e.g., [EMAIL_ADDRESS_1]) with their original values.

        Args:
            text: Text with anonymization tokens

        Returns:
            Text with tokens replaced by original values
        """
        if not self.mappings:
            return text

        restored = text
        for token, original in self.mappings.items():
            restored = restored.replace(token, original)

        return restored

    def validate_tokens(self, text: str) -> dict:
        """Validate that all tokens in text exist in our mappings.

        This checks the integrity of tokens returned by the remote LLM to detect:
        - Renumbered tokens (e.g., [EMAIL_ADDRESS_1] → [EMAIL_ADDRESS_2])
        - Invented tokens (tokens not in our mappings)
        - Missing tokens (expected tokens that were removed)
        - Malformed tokens (spacing issues, etc.)

        Args:
            text: Text that may contain anonymization tokens

        Returns:
            dict with:
                - is_valid: bool - all tokens are valid
                - found_tokens: list - tokens found in text
                - unknown_tokens: list - tokens not in our mappings
                - missing_tokens: list - tokens from mappings not found in text
                - warnings: list - validation warnings
        """
        # Extract all tokens from text (pattern: [WORD_NUMBER])
        token_pattern = r'\[([A-Z_]+_\d+)\]'
        found_tokens = re.findall(token_pattern, text)
        found_tokens_with_brackets = [f"[{t}]" for t in found_tokens]

        warnings = []
        unknown_tokens = []
        missing_tokens = []

        # Check for unknown tokens (not in our mappings)
        for token in found_tokens_with_brackets:
            if token not in self.mappings:
                unknown_tokens.append(token)
                warnings.append(f"Unknown token found: {token} (not in mappings)")

        # Check for missing tokens (should exist but don't)
        for token in self.mappings.keys():
            if token not in text:
                missing_tokens.append(token)
                warnings.append(f"Expected token missing from LLM response: {token}")

        is_valid = len(unknown_tokens) == 0 and len(missing_tokens) == 0

        return {
            "is_valid": is_valid,
            "found_tokens": found_tokens_with_brackets,
            "unknown_tokens": unknown_tokens,
            "missing_tokens": missing_tokens,
            "warnings": warnings,
        }

    def restore_with_validation(self, text: str, strict: bool = False) -> tuple[str, dict]:
        """Restore tokens with integrity validation.

        Validates that all tokens are correct before restoring. Can optionally
        refuse to restore if tokens are invalid.

        Args:
            text: Text with anonymization tokens
            strict: If True, raise error on invalid tokens. If False, warn and continue.

        Returns:
            Tuple of (restored_text, validation_result)

        Raises:
            ValueError: If strict=True and tokens are invalid
        """
        # Validate first
        validation = self.validate_tokens(text)

        if not validation["is_valid"] and strict:
            error_msg = "Token integrity check failed:\n"
            error_msg += f"  Unknown tokens: {validation['unknown_tokens']}\n"
            error_msg += f"  Missing tokens: {validation['missing_tokens']}"
            raise ValueError(error_msg)

        if validation["warnings"]:
            for warning in validation["warnings"]:
                logger.warning(f"Token validation: {warning}")

        # Restore normally
        restored = self.restore(text)

        return restored, validation

    def reset(self):
        """Reset all mappings and counters for a new session.

        This clears the token->value mappings and value->token reverse mappings,
        so tokens will be regenerated from scratch on the next anonymize() call.
        """
        self.mappings = {}
        self.reverse_mappings = {}
        self.counters = {}

    def get_current_mappings(self) -> dict:
        """Get current token->value mappings without resetting.

        Useful for debugging or logging what was anonymized.

        Returns:
            Copy of current mappings dict
        """
        return self.mappings.copy()

    def privatize_content(self, content: str) -> str:
        """Sanitize post content using Presidio.

        Args:
            content: The post content string to sanitize.

        Returns:
            Anonymized content string with PII replaced by tokens.
        """
        result = self.anonymize(content)
        return result["anonymized"]

    def privatize_author(self, username: str, acct: str) -> tuple[str, str]:
        """Sanitize author information using Presidio.

        Args:
            username: The account username
            acct: The account acct (email-like identifier)

        Returns:
            Tuple of (anonymized_username, anonymized_acct)
        """
        anon_username = self.privatize_content(username)
        anon_acct = self.privatize_content(acct)
        return anon_username, anon_acct

    def privatize_tags(self, tags: list[str]) -> list[str]:
        """Sanitize tags/hashtags using Presidio.

        Args:
            tags: List of tag strings

        Returns:
            List of anonymized tags
        """
        return [self.privatize_content(tag) for tag in tags]

    def privatize_context(self, context: str) -> str:
        """Sanitize formatted context string for LLM using Presidio.

        This is the main entry point for privacy handling in the agent pipeline.
        Takes the full formatted context (as produced by synthesis._format_context)
        and applies Presidio-based anonymization.

        Args:
            context: The formatted context string with multiple posts

        Returns:
            Fully anonymized context ready for remote LLM
        """
        result = self.anonymize(context)
        return result["anonymized"]
