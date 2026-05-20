"""
RAG Retrieval Module for German Corporate Tax Law
==================================================
Covers: KStG, GewStG, EStG, AO, UmwStG, AStG, DBA

Responsible AI features:
- Source citations with §-references
- Relevance threshold (refuses low-confidence results)
- Schema validation
- Embedding cache
- Full audit logging

Fixed Version for German jurisdiction.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


logger = logging.getLogger("german_tax_retrieval")
logger.setLevel(logging.INFO)

REQUIRED_FIELDS = {"text", "gesetz"}
OPTIONAL_FIELDS = {"rule_id", "paragraph", "absatz", "title", "effective_date", "keywords"}


# ===========================
# Sample German Tax Rules
# ===========================

SAMPLE_GERMAN_TAX_RULES = [
    {
        "rule_id": "KStG_8b_1",
        "gesetz": "KStG",
        "paragraph": "§8b Abs. 1",
        "title": "Beteiligungserträge - Freistellung",
        "text": "Bezüge im Sinne des §20 Abs. 1 Nr. 1, 2, 9 und 10 Buchstabe a des Einkommensteuergesetzes bleiben bei der Ermittlung des Einkommens außer Ansatz. §8b Abs. 5 KStG: Von den Bezügen gelten 5% als Ausgaben, die nicht als Betriebsausgaben abgezogen werden dürfen (faktische Steuerbelastung von 5% × 15% ≈ 0,75%).",
        "keywords": ["Beteiligungserträge", "Dividenden", "Schachtelprivileg", "5% Pauschale"],
    },
    {
        "rule_id": "KStG_8c_1",
        "gesetz": "KStG",
        "paragraph": "§8c Abs. 1",
        "title": "Verlustabzug bei Körperschaften - Anteilsübertragung",
        "text": "Werden innerhalb von fünf Jahren mittelbar oder unmittelbar mehr als 50% des gezeichneten Kapitals, der Mitgliedschaftsrechte, der Beteiligungsrechte oder der Stimmrechte an einer Körperschaft an einen Erwerber übertragen, sind die bis zum schädlichen Beteiligungserwerb nicht genutzten Verluste vollständig nicht mehr abziehbar. Die Konzernklausel (§8c Abs. 1 Satz 4) und die Stille-Reserven-Klausel (§8c Abs. 1 Satz 5-8) können den Verlustuntergang verhindern.",
        "keywords": ["Verlustvorträge", "Mantelkauf", "Anteilsübertragung", "Konzernklausel"],
    },
    {
        "rule_id": "KStG_14",
        "gesetz": "KStG",
        "paragraph": "§14 Abs. 1",
        "title": "Organschaft - Voraussetzungen",
        "text": "Verpflichtet sich eine Europäische Gesellschaft, AG oder KGaA mit Geschäftsleitung und Sitz im Inland (Organgesellschaft) durch einen Gewinnabführungsvertrag, ihren ganzen Gewinn an ein einziges anderes gewerbliches Unternehmen abzuführen, so ist das Einkommen der Organgesellschaft dem Organträger zuzurechnen. Voraussetzungen: finanzielle Eingliederung (Mehrheit der Stimmrechte), Gewinnabführungsvertrag für mindestens 5 Jahre, tatsächliche Durchführung.",
        "keywords": ["Organschaft", "Gewinnabführungsvertrag", "Organträger", "Organgesellschaft"],
    },
    {
        "rule_id": "GewStG_8",
        "gesetz": "GewStG",
        "paragraph": "§8 Nr. 1",
        "title": "Hinzurechnungen bei der Gewerbesteuer",
        "text": "Dem Gewinn aus Gewerbebetrieb werden folgende Beträge wieder hinzugerechnet, soweit sie bei der Ermittlung des Gewinns abgesetzt worden sind: 25% der Summe aus Entgelten für Schulden (Zinsen), Renten und dauernde Lasten, Gewinnanteile des stillen Gesellschafters, 20% der Miet- und Pachtzinsen für bewegliche Wirtschaftsgüter, 50% der Miet- und Pachtzinsen für unbewegliche Wirtschaftsgüter, 25% der Aufwendungen für Rechteüberlassungen. Freibetrag: 200.000 EUR.",
        "keywords": ["Hinzurechnung", "Gewerbesteuer", "Zinsen", "Mieten", "Pachten"],
    },
    {
        "rule_id": "GewStG_9",
        "gesetz": "GewStG",
        "paragraph": "§9 Nr. 1",
        "title": "Kürzungen bei der Gewerbesteuer",
        "text": "Die Summe des Gewinns und der Hinzurechnungen wird gekürzt um 1,2% des Einheitswerts des zum Betriebsvermögen gehörenden Grundbesitzes (einfache Kürzung). Bei Unternehmen, die ausschließlich eigenen Grundbesitz verwalten (erweiterte Kürzung nach §9 Nr. 1 Satz 2 GewStG), wird der auf die Verwaltung entfallende Gewerbeertrag vollständig gekürzt.",
        "keywords": ["Kürzung", "Grundbesitz", "erweiterte Kürzung", "Immobilien"],
    },
    {
        "rule_id": "GewStG_11",
        "gesetz": "GewStG",
        "paragraph": "§11 Abs. 1-3",
        "title": "Steuermesszahl und Hebesatz",
        "text": "Die Gewerbesteuer wird auf der Grundlage des Steuermessbetrags festgesetzt. Die Steuermesszahl beträgt 3,5%. Die Gemeinde bestimmt den Hebesatz, mit dem die Gewerbesteuer erhoben wird. Der Hebesatz muss mindestens 200% betragen. Effektive GewSt-Belastung = 3,5% × Hebesatz/100. Beispiel München (490%): 3,5% × 4,9 = 17,15%.",
        "keywords": ["Hebesatz", "Steuermesszahl", "Gemeinde", "Gewerbesteuer"],
    },
    {
        "rule_id": "EStG_7g",
        "gesetz": "EStG",
        "paragraph": "§7g",
        "title": "Investitionsabzugsbetrag",
        "text": "Steuerpflichtige können für die künftige Anschaffung oder Herstellung von abnutzbaren beweglichen Wirtschaftsgütern des Anlagevermögens bis zu 50% der voraussichtlichen Anschaffungs- oder Herstellungskosten gewinnmindernd abziehen (Investitionsabzugsbetrag). Voraussetzung: Betriebsvermögen nicht mehr als 235.000 EUR (Bilanzierung) oder Gewinn nicht mehr als 200.000 EUR. Investition muss innerhalb von 3 Jahren erfolgen.",
        "keywords": ["Investitionsabzugsbetrag", "IAB", "Sonderabschreibung", "KMU"],
    },
    {
        "rule_id": "EStG_6b",
        "gesetz": "EStG",
        "paragraph": "§6b",
        "title": "Übertragung stiller Reserven",
        "text": "Steuerpflichtige, die Grund und Boden, Aufwuchs, Gebäude oder Binnenschiffe veräußern, können im Wirtschaftsjahr der Veräußerung die aufgedeckten stillen Reserven auf angeschaffte oder hergestellte Ersatzwirtschaftsgüter übertragen und damit die Versteuerung des Veräußerungsgewinns aufschieben. Reinvestitionsfrist: 4 Jahre (Gebäude: 6 Jahre). §6b-Rücklage möglich.",
        "keywords": ["stille Reserven", "Reinvestition", "§6b-Rücklage", "Veräußerungsgewinn"],
    },
    {
        "rule_id": "AStG_1",
        "gesetz": "AStG",
        "paragraph": "§1 AStG",
        "title": "Internationale Verrechnungspreise",
        "text": "Werden Einkünfte eines Steuerpflichtigen dadurch gemindert, dass er im Rahmen seiner Geschäftsbeziehungen zum Ausland Bedingungen vereinbart, die von denen abweichen, die voneinander unabhängige Dritte unter gleichen Verhältnissen vereinbart hätten (Fremdvergleichsgrundsatz), so sind seine Einkünfte so anzusetzen, wie sie bei Vereinbarung fremdüblicher Bedingungen angefallen wären. Dokumentationspflicht nach §90 Abs. 3 AO.",
        "keywords": ["Verrechnungspreise", "Fremdvergleich", "Transfer Pricing", "AStG"],
    },
    {
        "rule_id": "KStG_27",
        "gesetz": "KStG",
        "paragraph": "§27",
        "title": "Steuerliches Einlagekonto",
        "text": "Die unbeschränkt steuerpflichtige Kapitalgesellschaft hat die nicht in das Nennkapital geleisteten Einlagen am Schluss jedes Wirtschaftsjahrs auf einem besonderen Konto (steuerliches Einlagekonto) auszuweisen. Ausschüttungen aus dem steuerlichen Einlagekonto sind beim Anteilseigner keine steuerpflichtigen Einkünfte, sondern mindern die Anschaffungskosten der Beteiligung.",
        "keywords": ["Einlagekonto", "Kapitalrücklage", "Ausschüttung", "§27 KStG"],
    },
    {
        "rule_id": "UmwStG_11",
        "gesetz": "UmwStG",
        "paragraph": "§11",
        "title": "Verschmelzung - Wertansatz",
        "text": "Bei Verschmelzung einer Körperschaft auf eine andere Körperschaft sind die übergehenden Wirtschaftsgüter in der steuerlichen Schlussbilanz der übertragenden Körperschaft mit dem gemeinen Wert anzusetzen. Auf Antrag können die Wirtschaftsgüter mit dem Buchwert oder einem Zwischenwert angesetzt werden (steuerneutrale Verschmelzung), sofern die Voraussetzungen des §11 Abs. 2 UmwStG erfüllt sind.",
        "keywords": ["Verschmelzung", "Umwandlung", "Buchwert", "steuerneutral"],
    },
    {
        "rule_id": "Zinsschranke",
        "gesetz": "EStG/KStG",
        "paragraph": "§4h EStG / §8a KStG",
        "title": "Zinsschranke",
        "text": "Zinsaufwendungen eines Betriebs sind abziehbar in Höhe des Zinsertrags desselben Wirtschaftsjahrs zuzüglich 30% des verrechenbaren EBITDA. Darüber hinausgehende Zinsaufwendungen (Nettozinsaufwand) sind nicht abziehbar und werden vorgetragen. Ausnahmen: Freigrenze 3 Mio. EUR, Stand-alone-Klausel, Escape-Klausel (Eigenkapitalvergleich). Seit 2024: ATAD-Umsetzung verschärft.",
        "keywords": ["Zinsschranke", "EBITDA", "Nettozinsaufwand", "Fremdfinanzierung"],
    },
]


class GermanTaxLawRetriever:
    """
    Semantic retrieval for German corporate tax law.
    Covers KStG, GewStG, EStG, AO, UmwStG, AStG, DBA.

    Responsible AI features:
    - §-genaue Quellenangaben (paragraph-level citations)
    - Relevance filtering (refuses garbage results)
    - Embedding persistence
    - Full audit trail
    """

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",  # German-capable
        min_relevance: float = 0.3,
        output_dir: str = "output/german_tax_rag",
    ):
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

    def load_rules(self, json_path: Optional[str] = None):
        """
        Load German tax rules from JSON or use built-in samples.

        JSON format: list of dicts with at minimum "text" and "gesetz" fields.
        """
        if json_path is not None:
            path = Path(json_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {json_path}")
            with open(path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
        else:
            # Use built-in German tax rules
            raw_data = SAMPLE_GERMAN_TAX_RULES
            print("ℹ️  Using built-in German tax rules (12 rules). "
                  "Provide json_path for full Bundesrecht database.")

        # Validate
        validated = []
        for i, rule in enumerate(raw_data):
            if not isinstance(rule, dict):
                continue
            if "text" not in rule or not rule["text"].strip():
                logger.warning(f"Rule {i}: missing/empty 'text', skipping.")
                continue
            if "gesetz" not in rule:
                rule["gesetz"] = "Unbekannt"
            validated.append(rule)

        if not validated:
            raise ValueError("No valid rules found.")

        self.tax_rules = validated
        self.rule_texts = [r["text"] for r in self.tax_rules]

        # Embeddings (with cache)
        cache_name = f"de_tax_emb_{len(self.tax_rules)}_{self.model_name.replace('/', '_')}.npy"
        cache_path = self.output_dir / cache_name

        if cache_path.exists():
            self.embeddings = np.load(cache_path)
            if self.embeddings.shape[0] != len(self.rule_texts):
                self._compute_embeddings(cache_path)
            else:
                logger.info(f"Loaded cached embeddings: {cache_path}")
        else:
            self._compute_embeddings(cache_path)

        print(f"✅ {len(self.tax_rules)} deutsche Steuervorschriften geladen.")

    def _compute_embeddings(self, cache_path: Path):
        self.embeddings = self.encoder.encode(self.rule_texts, show_progress_bar=True)
        np.save(cache_path, self.embeddings)

    # ===========================
    # Query / Retrieval
    # ===========================

    def query(
        self,
        question: str,
        top_k: int = 3,
        min_relevance: Optional[float] = None,
        filter_gesetz: Optional[str] = None,
    ) -> List[dict]:
        """
        Retrieve relevant German tax rules.

        Parameters
        ----------
        question : query in German or English
        top_k : max results
        min_relevance : override threshold
        filter_gesetz : filter by law (e.g. "KStG", "GewStG")

        Returns
        -------
        list of result dicts with paragraph citations
        """
        if self.embeddings is None:
            raise ValueError("No rules loaded. Call load_rules() first.")

        if not question or not question.strip():
            raise ValueError("Query must be non-empty.")

        threshold = min_relevance if min_relevance is not None else self.min_relevance

        # Encode & compute similarity
        q_emb = self.encoder.encode([question.strip()])
        sims = cosine_similarity(q_emb, self.embeddings)[0]

        # Optional: filter by Gesetz
        if filter_gesetz:
            for i, rule in enumerate(self.tax_rules):
                if rule.get("gesetz", "").upper() != filter_gesetz.upper():
                    sims[i] = -1  # Exclude

        # Rank
        top_indices = np.argsort(sims)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            score = float(sims[idx])
            if score < threshold:
                continue

            rule = self.tax_rules[idx]
            results.append({
                "rule_id": rule.get("rule_id", f"rule_{idx}"),
                "gesetz": rule.get("gesetz", ""),
                "paragraph": rule.get("paragraph", ""),
                "title": rule.get("title", ""),
                "text": rule["text"],
                "keywords": rule.get("keywords", []),
                "relevance": score,
                "citation": f"{rule.get('paragraph', '')} {rule.get('gesetz', '')}".strip(),
            })

        # Log
        self._log_query(question, results, float(sims.max()))

        if not results:
            return [{
                "warning": "Keine ausreichend relevanten Vorschriften gefunden.",
                "max_similarity": float(sims.max()),
                "threshold": threshold,
                "hinweis": "Versuchen Sie eine andere Formulierung oder senken Sie min_relevance.",
            }]

        return results

    def format_context(self, results: List[dict], max_chars: int = 4000) -> str:
        """Format results as LLM context with German legal citations."""
        if not results or "warning" in results[0]:
            return "Keine relevanten Steuervorschriften für diese Anfrage gefunden."

        parts = []
        total = 0

        for r in results:
            entry = (
                f"📖 [{r['citation']}] {r['title']}\n"
                f"   Gesetz: {r['gesetz']} | Relevanz: {r['relevance']:.3f}\n"
                f"   Inhalt: {r['text']}\n"
            )
            if total + len(entry) > max_chars:
                break
            parts.append(entry)
            total += len(entry)

        return "\n{'---'}\n".join(parts)

    # ===========================
    # Audit
    # ===========================

    def _log_query(self, question, results, max_sim):
        self._query_log.append({
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "n_results": len(results),
            "max_similarity": max_sim,
            "citations": [r.get("citation") for r in results],
        })

    def get_audit_log(self) -> List[dict]:
        return list(self._query_log)

    def export_audit_log(self, path=None):
        path = path or (self.output_dir / "audit_log.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self._query_log, f, indent=2, ensure_ascii=False)
        print(f"📝 Audit log: {path} ({len(self._query_log)} Einträge)")

    def stats(self) -> dict:
        return {
            "n_rules": len(self.tax_rules) if self.tax_rules else 0,
            "gesetze": list(set(r.get("gesetz") for r in (self.tax_rules or []))),
            "model": self.model_name,
            "min_relevance": self.min_relevance,
        }
