"""
RAG Retrieval Module for Corporate Tax Law
Responsible AI Version - Source citations, relevance filtering, audit logging

Fixed Version:
- Relevance threshold filtering (refuses to return irrelevant results)
- JSON schema validation
- Embedding persistence (save/load)
- Empty input protection
- Query audit logging
- Clear naming (RetrievalModule, not full RAG)
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Configure audit logger
logger = logging.getLogger("tax_law_retrieval")
logger.setLevel(logging.INFO)


REQUIRED_FIELDS = {"text"}  # Minimum required fields in each rule
OPTIONAL_FIELDS = {"rule_id", "title", "source", "section", "effective_date"}


class TaxLawRetriever:
    """
    Semantic retrieval module for corporate tax law.
    Retrieves relevant tax rules with source citations and relevance filtering.

    This is a Retrieval module (the "R" in RAG). It does NOT generate answers.
    Pair with an LLM to complete the RAG pipeline.

    Responsible AI features:
    - Refuses to return results below relevance threshold
    - All results include source citations
    - Query audit logging for traceability
    - Schema validation on load
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        min_relevance: float = 0.3,
        output_dir: str = "output/rag",
    ):
        """
        Parameters
        ----------
        model_name : sentence-transformer model for encoding
        min_relevance : minimum cosine similarity to return a result (0.0 - 1.0)
        output_dir : directory for embeddings cache and audit logs
        """
        self.encoder = SentenceTransformer(model_name)
        self.model_name = model_name
        self.min_relevance = min_relevance
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.tax_rules = None
        self.rule_texts = None
        self.embeddings = None
        self._query_log = []

    # ===========================
    # Data Loading
    # ===========================

    def load_rules(self, json_path: str = "data/tax_rules_sample.json"):
        """
        Load and validate corporate tax rules from JSON file.

        Expected JSON format: list of dicts, each must have "text" field.
        Optional fields: rule_id, title, source, section, effective_date.
        """
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Tax rules file not found: {json_path}")

        with open(path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        # --- Schema validation ---
        if not isinstance(raw_data, list):
            raise ValueError("JSON file must contain a list of rule objects.")

        validated_rules = []
        errors = []

        for i, rule in enumerate(raw_data):
            if not isinstance(rule, dict):
                errors.append(f"Item {i}: not a dict, skipping.")
                continue

            missing = REQUIRED_FIELDS - set(rule.keys())
            if missing:
                errors.append(f"Item {i}: missing required fields {missing}, skipping.")
                continue

            if not rule["text"] or not rule["text"].strip():
                errors.append(f"Item {i}: empty 'text' field, skipping.")
                continue

            validated_rules.append(rule)

        if errors:
            logger.warning(
                f"Schema validation: {len(errors)} issues found:\n" +
                "\n".join(errors[:10])  # Show first 10
            )

        if not validated_rules:
            raise ValueError("No valid rules found after schema validation.")

        self.tax_rules = validated_rules
        self.rule_texts = [rule["text"] for rule in self.tax_rules]

        # Try to load cached embeddings, otherwise compute
        cache_path = self.output_dir / f"embeddings_{path.stem}_{self.model_name.replace('/', '_')}.npy"
        if cache_path.exists():
            self.embeddings = np.load(cache_path)
            if self.embeddings.shape[0] != len(self.rule_texts):
                logger.info("Embedding cache size mismatch, recomputing...")
                self._compute_and_cache_embeddings(cache_path)
            else:
                logger.info(f"Loaded cached embeddings from {cache_path}")
        else:
            self._compute_and_cache_embeddings(cache_path)

        print(f"✅ Loaded {len(self.tax_rules)} corporate tax rules successfully.")

    def _compute_and_cache_embeddings(self, cache_path: Path):
        """Compute embeddings and save to disk."""
        self.embeddings = self.encoder.encode(self.rule_texts, show_progress_bar=True)
        np.save(cache_path, self.embeddings)
        logger.info(f"Computed and cached embeddings to {cache_path}")

    # ===========================
    # Query / Retrieval
    # ===========================

    def query(
        self,
        question: str,
        top_k: int = 3,
        min_relevance: Optional[float] = None,
    ) -> List[dict]:
        """
        Retrieve most relevant tax rules with source citations.

        Parameters
        ----------
        question : natural language query
        top_k : maximum number of results to return
        min_relevance : override instance-level threshold (0.0 - 1.0)

        Returns
        -------
        list of dict: Each result contains rule_id, title, text, source, relevance.
        If no results meet the threshold, returns a single dict with a warning.
        """
        if self.embeddings is None:
            raise ValueError("No rules loaded. Call load_rules() first.")

        # Input validation
        if not question or not question.strip():
            raise ValueError("Query must be a non-empty string.")

        threshold = min_relevance if min_relevance is not None else self.min_relevance

        # Encode query
        q_emb = self.encoder.encode([question.strip()])
        similarities = cosine_similarity(q_emb, self.embeddings)[0]

        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        # Filter by relevance threshold
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score < threshold:
                continue

            rule = self.tax_rules[idx]
            results.append({
                "rule_id": rule.get("rule_id", f"rule_{idx}"),
                "title": rule.get("title", "Untitled Rule"),
                "text": rule["text"],
                "source": rule.get("source", "Corporate Tax Law"),
                "section": rule.get("section", None),
                "relevance": score,
            })

        # Audit log
        self._log_query(question, results, float(similarities.max()))

        # If no results pass threshold, return explicit warning
        if not results:
            return [{
                "warning": "No sufficiently relevant rules found for this query.",
                "max_similarity": float(similarities.max()),
                "threshold": threshold,
                "suggestion": "Try rephrasing your question or lowering min_relevance.",
            }]

        return results

    # ===========================
    # Context Formatting (for LLM integration)
    # ===========================

    def format_context(self, results: List[dict], max_chars: int = 3000) -> str:
        """
        Format retrieval results as context string for LLM prompt injection.
        This bridges the gap between retrieval and generation in a RAG pipeline.

        Parameters
        ----------
        results : output from query()
        max_chars : maximum total characters in context

        Returns
        -------
        str : formatted context with citations
        """
        if not results or "warning" in results[0]:
            return "No relevant tax rules found for the given query."

        context_parts = []
        total_chars = 0

        for r in results:
            entry = (
                f"[{r['rule_id']}] {r['title']}\n"
                f"Source: {r['source']}\n"
                f"Relevance: {r['relevance']:.3f}\n"
                f"Content: {r['text']}\n"
            )

            if total_chars + len(entry) > max_chars:
                break

            context_parts.append(entry)
            total_chars += len(entry)

        return "\n---\n".join(context_parts)

    # ===========================
    # Audit & Logging
    # ===========================

    def _log_query(self, question: str, results: List[dict], max_similarity: float):
        """Log query for audit trail."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "n_results": len(results),
            "max_similarity": max_similarity,
            "result_ids": [r.get("rule_id") for r in results],
        }
        self._query_log.append(entry)
        logger.info(f"Query: '{question[:50]}...' -> {len(results)} results (max_sim={max_similarity:.3f})")

    def get_audit_log(self) -> List[dict]:
        """Return full query audit log."""
        return list(self._query_log)

    def export_audit_log(self, path: Optional[str] = None):
        """Export audit log to JSON file."""
        if path is None:
            path = self.output_dir / "query_audit_log.json"

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self._query_log, f, indent=2, ensure_ascii=False)

        print(f"📝 Audit log exported to {path} ({len(self._query_log)} entries)")

    # ===========================
    # Utility
    # ===========================

    def stats(self) -> dict:
        """Return current state summary."""
        return {
            "n_rules": len(self.tax_rules) if self.tax_rules else 0,
            "model_name": self.model_name,
            "min_relevance": self.min_relevance,
            "embedding_dim": self.embeddings.shape[1] if self.embeddings is not None else None,
            "n_queries_logged": len(self._query_log),
        }
