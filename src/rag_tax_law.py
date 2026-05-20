"""
RAG Pipeline for Corporate Tax Law Retrieval
Responsible AI Version - Always returns source citations
"""

import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class TaxLawRAG:
    """
    Lightweight RAG module for corporate tax law retrieval.
    Emphasizes traceability and source citation (core Responsible AI principle).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model_name)
        self.tax_rules = None
        self.rule_texts = None
        self.embeddings = None

    def load_rules(self, json_path: str = "data/tax_rules_sample.json"):
        """Load corporate tax rules from JSON file."""
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Tax rules file not found: {json_path}")

        with open(path, 'r', encoding='utf-8') as f:
            self.tax_rules = json.load(f)

        self.rule_texts = [rule["text"] for rule in self.tax_rules]
        self.embeddings = self.encoder.encode(self.rule_texts)

        print(f"✅ Loaded {len(self.tax_rules)} corporate tax rules successfully.")

    def query(self, question: str, top_k: int = 3):
        """
        Retrieve most relevant tax rules with source citations.
        
        Returns
        -------
        list of dict: Each result contains rule_id, title, text, source, relevance
        """
        if self.embeddings is None:
            raise ValueError("Please call load_rules() first.")

        q_emb = self.encoder.encode([question])
        similarities = cosine_similarity(q_emb, self.embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            rule = self.tax_rules[idx]
            results.append({
                "rule_id": rule.get("rule_id", f"rule_{idx}"),
                "title": rule.get("title", "Untitled Rule"),
                "text": rule["text"],
                "source": rule.get("source", "Corporate Tax Law"),
                "relevance": float(similarities[idx])
            })
        return results
