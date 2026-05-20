"""
German Tax Law Versioning & Temporal Validity
==============================================
Handles the constant evolution of German tax legislation.

Core problems solved:
1. Same § has different content depending on the tax year
2. New laws supersede old ones with transitional rules
3. BMF-Schreiben can reinterpret existing law
4. BFH rulings can invalidate provisions (e.g. §8c KStG BVerfG 2017)
5. Annual Hebesatz changes per municipality

Architecture:
- Each rule has temporal metadata (gueltig_ab, gueltig_bis, veranlagungszeitraum)
- Query always requires a reference date or tax year
- System warns when using potentially outdated rules
- Change history (Änderungshistorie) is preserved
- Supports "as-of" queries (law as it stood on date X)

Key German tax law change sources:
- Jahressteuergesetz (annual)
- Wachstumschancengesetz 2024
- ATAD-Umsetzungsgesetz 2021/2024
- KöMoG 2021
- JStG 2022, 2023, 2024
- BMF-Schreiben (administrative guidance)
- BFH-Urteile (court decisions)

Fixed & Production-Ready Version.
"""

import json
import logging
import warnings
from pathlib import Path
from datetime import date, datetime
from typing import Optional, List, Dict, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger("german_tax_versioning")


# ===========================
# Enums & Data Structures
# ===========================

class Rechtsquelle(Enum):
    """Source type of a legal provision."""
    GESETZ = "Gesetz"                    # Parliamentary law
    VERORDNUNG = "Verordnung"            # Regulation
    BMF_SCHREIBEN = "BMF-Schreiben"      # Ministry guidance
    BFH_URTEIL = "BFH-Urteil"           # Federal Tax Court ruling
    BVERFG_URTEIL = "BVerfG-Urteil"     # Constitutional Court
    EU_RICHTLINIE = "EU-Richtlinie"     # EU Directive
    DBA = "DBA"                          # Double tax treaty


class Aenderungstyp(Enum):
    """Type of change."""
    NEUFASSUNG = "Neufassung"            # Complete rewrite
    AENDERUNG = "Änderung"              # Amendment
    AUFHEBUNG = "Aufhebung"             # Repeal
    EINFUEGUNG = "Einfügung"            # Insertion of new provision
    NEUINTERPRETATION = "Neuinterpretation"  # Reinterpretation (BMF/BFH)
    VERFASSUNGSWIDRIG = "Verfassungswidrig"  # Declared unconstitutional


@dataclass
class VersionMetadata:
    """Temporal and source metadata for a single version of a rule."""
    version_id: str
    gueltig_ab: date                       # Effective from
    gueltig_bis: Optional[date] = None     # Effective until (None = still valid)
    veranlagungszeitraum_ab: Optional[int] = None   # Applies from tax year
    veranlagungszeitraum_bis: Optional[int] = None  # Applies until tax year (None = ongoing)
    aenderungsgesetz: str = ""             # Law that introduced this version
    aenderungstyp: Aenderungstyp = Aenderungstyp.AENDERUNG
    rechtsquelle: Rechtsquelle = Rechtsquelle.GESETZ
    bundesgesetzblatt: str = ""            # BGBl reference
    hinweis: str = ""                      # Additional notes

    def is_valid_on(self, reference_date: date) -> bool:
        """Check if this version was/is valid on a specific date."""
        if self.gueltig_ab > reference_date:
            return False
        if self.gueltig_bis and self.gueltig_bis < reference_date:
            return False
        return True

    def applies_to_year(self, tax_year: int) -> bool:
        """Check if this version applies to a specific Veranlagungszeitraum."""
        if self.veranlagungszeitraum_ab and tax_year < self.veranlagungszeitraum_ab:
            return False
        if self.veranlagungszeitraum_bis and tax_year > self.veranlagungszeitraum_bis:
            return False
        return True


@dataclass
class VersionedRule:
    """
    A single tax rule with full version history.

    Example: §8c KStG has had multiple versions:
    - Original 2008 version
    - 2010 amendment (Konzernklausel added)
    - 2017 BVerfG partial unconstitutionality
    - 2018 legislative fix
    - 2024 further amendments
    """
    rule_id: str
    gesetz: str                             # e.g. "KStG"
    paragraph: str                          # e.g. "§8c Abs. 1"
    title: str
    versions: List[Dict] = field(default_factory=list)
    # Each version: {"text": ..., "metadata": VersionMetadata, ...}

    def get_version_for_date(self, reference_date: date) -> Optional[Dict]:
        """Get the applicable version for a specific date."""
        applicable = [
            v for v in self.versions
            if v["metadata"].is_valid_on(reference_date)
        ]
        if not applicable:
            return None
        # Return the most recent applicable version
        return sorted(applicable, key=lambda v: v["metadata"].gueltig_ab, reverse=True)[0]

    def get_version_for_year(self, tax_year: int) -> Optional[Dict]:
        """Get the applicable version for a specific tax year."""
        applicable = [
            v for v in self.versions
            if v["metadata"].applies_to_year(tax_year)
        ]
        if not applicable:
            return None
        return sorted(applicable, key=lambda v: v["metadata"].gueltig_ab, reverse=True)[0]

    def get_change_history(self) -> List[Dict]:
        """Get the full amendment history."""
        history = []
        for v in sorted(self.versions, key=lambda v: v["metadata"].gueltig_ab):
            meta = v["metadata"]
            history.append({
                "version_id": meta.version_id,
                "gueltig_ab": meta.gueltig_ab.isoformat(),
                "aenderungsgesetz": meta.aenderungsgesetz,
                "aenderungstyp": meta.aenderungstyp.value,
                "hinweis": meta.hinweis,
            })
        return history


# ===========================
# Sample Versioned Rules
# ===========================

def _build_sample_versioned_rules() -> List[VersionedRule]:
    """
    Build sample rules with version history.
    Demonstrates how the same paragraph changes over time.
    """

    rules = []

    # --- §8c KStG: Verlustabzug bei Körperschaften ---
    rule_8c = VersionedRule(
        rule_id="KStG_8c",
        gesetz="KStG",
        paragraph="§8c",
        title="Verlustabzug bei Körperschaften",
        versions=[
            {
                "text": (
                    "Werden innerhalb von fünf Jahren mittelbar oder unmittelbar mehr als "
                    "25 Prozent des gezeichneten Kapitals an einen Erwerber übertragen, "
                    "sind die Verluste anteilig nicht abziehbar (quotaler Untergang). "
                    "Bei mehr als 50 Prozent gehen die Verluste vollständig unter."
                ),
                "metadata": VersionMetadata(
                    version_id="8c_v1_2008",
                    gueltig_ab=date(2008, 1, 1),
                    gueltig_bis=date(2016, 12, 31),
                    veranlagungszeitraum_ab=2008,
                    veranlagungszeitraum_bis=2016,
                    aenderungsgesetz="Unternehmensteuerreformgesetz 2008",
                    aenderungstyp=Aenderungstyp.NEUFASSUNG,
                    bundesgesetzblatt="BGBl. I 2007, S. 1912",
                ),
            },
            {
                "text": (
                    "BVerfG-Beschluss vom 29.03.2017 (2 BvL 6/11): Der quotale "
                    "Verlustuntergang bei Übertragungen von 25-50% (§8c Abs. 1 Satz 1 KStG) "
                    "ist verfassungswidrig und nichtig für die Jahre 2008-2015. "
                    "Der vollständige Untergang bei >50% bleibt bestehen."
                ),
                "metadata": VersionMetadata(
                    version_id="8c_v2_bverfg_2017",
                    gueltig_ab=date(2017, 3, 29),
                    gueltig_bis=date(2017, 12, 31),
                    veranlagungszeitraum_ab=2008,
                    veranlagungszeitraum_bis=2015,
                    aenderungsgesetz="BVerfG 2 BvL 6/11",
                    aenderungstyp=Aenderungstyp.VERFASSUNGSWIDRIG,
                    rechtsquelle=Rechtsquelle.BVERFG_URTEIL,
                    hinweis="Quotaler Untergang (25-50%) nichtig für VZ 2008-2015",
                ),
            },
            {
                "text": (
                    "Werden innerhalb von fünf Jahren mittelbar oder unmittelbar mehr als "
                    "50 Prozent des gezeichneten Kapitals an einen Erwerber übertragen, "
                    "sind die bis zum schädlichen Beteiligungserwerb nicht genutzten Verluste "
                    "vollständig nicht mehr abziehbar. Der quotale Untergang (25-50%) wurde "
                    "gestrichen. Ausnahmen: Konzernklausel (Satz 4), Stille-Reserven-Klausel "
                    "(Satz 5-8), fortführungsgebundener Verlustvortrag (§8d KStG)."
                ),
                "metadata": VersionMetadata(
                    version_id="8c_v3_2018",
                    gueltig_ab=date(2018, 1, 1),
                    gueltig_bis=None,  # Still valid
                    veranlagungszeitraum_ab=2016,
                    veranlagungszeitraum_bis=None,
                    aenderungsgesetz="Gesetz zur Vermeidung von Umsatzsteuerausfällen (JStG 2018)",
                    aenderungstyp=Aenderungstyp.AENDERUNG,
                    bundesgesetzblatt="BGBl. I 2018, S. 2338",
                    hinweis="Quotaler Untergang endgültig gestrichen; nur noch >50% relevant",
                ),
            },
        ],
    )
    rules.append(rule_8c)

    # --- §7g EStG: Investitionsabzugsbetrag ---
    rule_7g = VersionedRule(
        rule_id="EStG_7g",
        gesetz="EStG",
        paragraph="§7g",
        title="Investitionsabzugsbetrag (IAB)",
        versions=[
            {
                "text": (
                    "Investitionsabzugsbetrag bis 40% der voraussichtlichen Anschaffungskosten. "
                    "Voraussetzung: Betriebsvermögen nicht mehr als 235.000 EUR bzw. Gewinn "
                    "nicht über 100.000 EUR. Investition innerhalb von 3 Jahren."
                ),
                "metadata": VersionMetadata(
                    version_id="7g_v1_2016",
                    gueltig_ab=date(2016, 1, 1),
                    gueltig_bis=date(2019, 12, 31),
                    veranlagungszeitraum_ab=2016,
                    veranlagungszeitraum_bis=2019,
                    aenderungsgesetz="Steueränderungsgesetz 2015",
                    aenderungstyp=Aenderungstyp.AENDERUNG,
                ),
            },
            {
                "text": (
                    "Investitionsabzugsbetrag bis 50% der voraussichtlichen Anschaffungs- "
                    "oder Herstellungskosten (erhöht von 40%). Einheitliche Gewinngrenze: "
                    "200.000 EUR für alle Einkunftsarten. Investition innerhalb von 3 Jahren "
                    "(Corona-bedingt verlängert auf 4 Jahre für 2017/2018er IAB). "
                    "Elektronische Übermittlung der Summen und Salden nicht mehr erforderlich."
                ),
                "metadata": VersionMetadata(
                    version_id="7g_v2_2020",
                    gueltig_ab=date(2020, 1, 1),
                    gueltig_bis=None,
                    veranlagungszeitraum_ab=2020,
                    veranlagungszeitraum_bis=None,
                    aenderungsgesetz="JStG 2020",
                    aenderungstyp=Aenderungstyp.AENDERUNG,
                    bundesgesetzblatt="BGBl. I 2020, S. 3096",
                    hinweis="50% statt 40%; einheitliche 200k-Gewinngrenze",
                ),
            },
        ],
    )
    rules.append(rule_7g)

    # --- Zinsschranke §4h EStG / §8a KStG ---
    rule_zinsschranke = VersionedRule(
        rule_id="Zinsschranke",
        gesetz="EStG/KStG",
        paragraph="§4h EStG / §8a KStG",
        title="Zinsschranke",
        versions=[
            {
                "text": (
                    "Zinsaufwendungen abziehbar bis Zinsertrag + 30% des EBITDA. "
                    "Freigrenze: 3 Mio. EUR Nettozinsaufwand. "
                    "Ausnahmen: Stand-alone-Klausel, Escape-Klausel (EK-Vergleich). "
                    "EBITDA-Vortrag: 5 Jahre vortragsfähig."
                ),
                "metadata": VersionMetadata(
                    version_id="zins_v1_2008",
                    gueltig_ab=date(2008, 1, 1),
                    gueltig_bis=date(2023, 12, 31),
                    veranlagungszeitraum_ab=2008,
                    veranlagungszeitraum_bis=2023,
                    aenderungsgesetz="Unternehmensteuerreformgesetz 2008",
                    aenderungstyp=Aenderungstyp.NEUFASSUNG,
                ),
            },
            {
                "text": (
                    "ATAD-Umsetzung (Anti Tax Avoidance Directive): "
                    "Zinsaufwendungen weiterhin abziehbar bis Zinsertrag + 30% des EBITDA. "
                    "Freigrenze bleibt bei 3 Mio. EUR. ABER: EBITDA-Vortrag wird auf "
                    "5 Jahre begrenzt (vorher unbegrenzt in der Praxis). "
                    "Verschärfung der Escape-Klausel: Eigenkapitalquote des Konzerns "
                    "wird strenger geprüft. Zinsvortrag weiterhin zeitlich unbegrenzt. "
                    "Neue Anti-Fragmentierungsregel für nahestehende Personen."
                ),
                "metadata": VersionMetadata(
                    version_id="zins_v2_2024_atad",
                    gueltig_ab=date(2024, 1, 1),
                    gueltig_bis=None,
                    veranlagungszeitraum_ab=2024,
                    veranlagungszeitraum_bis=None,
                    aenderungsgesetz="Wachstumschancengesetz / ATAD-UmsG",
                    aenderungstyp=Aenderungstyp.AENDERUNG,
                    bundesgesetzblatt="BGBl. I 2024, S. 108",
                    hinweis="ATAD-konforme Verschärfung; praxisrelevant ab VZ 2024",
                ),
            },
        ],
    )
    rules.append(rule_zinsschranke)

    # --- GewSt Hinzurechnung §8 Nr. 1 GewStG ---
    rule_hinzu = VersionedRule(
        rule_id="GewStG_8_1",
        gesetz="GewStG",
        paragraph="§8 Nr. 1 GewStG",
        title="Gewerbesteuerliche Hinzurechnungen",
        versions=[
            {
                "text": (
                    "Hinzurechnung von 25% der Summe aus: Entgelten für Schulden, "
                    "Renten und dauernde Lasten, Gewinnanteilen stiller Gesellschafter. "
                    "Dazu: 20% der Miet-/Pachtzinsen für bewegliche WG, "
                    "50% für unbewegliche WG, 25% für Rechteüberlassungen. "
                    "Freibetrag: 100.000 EUR."
                ),
                "metadata": VersionMetadata(
                    version_id="hinzu_v1_2008",
                    gueltig_ab=date(2008, 1, 1),
                    gueltig_bis=date(2019, 12, 31),
                    veranlagungszeitraum_ab=2008,
                    veranlagungszeitraum_bis=2019,
                    aenderungsgesetz="Unternehmensteuerreformgesetz 2008",
                    aenderungstyp=Aenderungstyp.NEUFASSUNG,
                ),
            },
            {
                "text": (
                    "Hinzurechnung von 25% der Summe aus: Entgelten für Schulden, "
                    "Renten und dauernde Lasten, Gewinnanteilen stiller Gesellschafter. "
                    "Dazu: 20% der Miet-/Pachtzinsen für bewegliche WG, "
                    "50% für unbewegliche WG, 25% für Rechteüberlassungen. "
                    "Freibetrag erhöht auf 200.000 EUR (vorher 100.000 EUR). "
                    "Praxishinweis: Entlastung insbesondere für Leasingunternehmen "
                    "und mietintensive Betriebe."
                ),
                "metadata": VersionMetadata(
                    version_id="hinzu_v2_2020",
                    gueltig_ab=date(2020, 1, 1),
                    gueltig_bis=None,
                    veranlagungszeitraum_ab=2020,
                    veranlagungszeitraum_bis=None,
                    aenderungsgesetz="JStG 2019 / Wachstumschancengesetz",
                    aenderungstyp=Aenderungstyp.AENDERUNG,
                    hinweis="Freibetrag verdoppelt auf 200k",
                ),
            },
        ],
    )
    rules.append(rule_hinzu)

    # --- Degressive AfA (Wachstumschancengesetz 2024) ---
    rule_afa = VersionedRule(
        rule_id="EStG_7_2_degrAfA",
        gesetz="EStG",
        paragraph="§7 Abs. 2",
        title="Degressive Abschreibung (Wiedereinführung)",
        versions=[
            {
                "text": (
                    "Die degressive AfA für bewegliche Wirtschaftsgüter war seit 2011 "
                    "nicht mehr möglich (§7 Abs. 2 EStG a.F. aufgehoben). "
                    "Nur lineare AfA nach §7 Abs. 1 EStG zulässig."
                ),
                "metadata": VersionMetadata(
                    version_id="degr_afa_v0_abgeschafft",
                    gueltig_ab=date(2011, 1, 1),
                    gueltig_bis=date(2019, 12, 31),
                    veranlagungszeitraum_ab=2011,
                    veranlagungszeitraum_bis=2019,
                    aenderungsgesetz="JStG 2010",
                    aenderungstyp=Aenderungstyp.AUFHEBUNG,
                ),
            },
            {
                "text": (
                    "Corona-Konjunkturpaket: Befristete Wiedereinführung der degressiven AfA "
                    "für bewegliche Wirtschaftsgüter des Anlagevermögens. Bis zu 25%, "
                    "maximal das 2,5-fache des linearen AfA-Satzes. Gilt für Anschaffungen "
                    "nach dem 31.12.2019 und vor dem 01.01.2022."
                ),
                "metadata": VersionMetadata(
                    version_id="degr_afa_v1_corona",
                    gueltig_ab=date(2020, 6, 30),
                    gueltig_bis=date(2021, 12, 31),
                    veranlagungszeitraum_ab=2020,
                    veranlagungszeitraum_bis=2021,
                    aenderungsgesetz="Zweites Corona-Steuerhilfegesetz",
                    aenderungstyp=Aenderungstyp.EINFUEGUNG,
                    bundesgesetzblatt="BGBl. I 2020, S. 1512",
                ),
            },
            {
                "text": (
                    "Wachstumschancengesetz: Erneute Einführung der degressiven AfA "
                    "für bewegliche Wirtschaftsgüter. Bis zu 20% (vorher 25%), "
                    "maximal das 2-fache des linearen Satzes. Gilt für Anschaffungen "
                    "nach dem 31.03.2024 und vor dem 01.01.2025. "
                    "Zusätzlich: degressive AfA für Wohngebäude (§7 Abs. 5a EStG) "
                    "mit 5% für Neubauten (Baubeginn nach 30.09.2023, Fertigstellung bis 2029)."
                ),
                "metadata": VersionMetadata(
                    version_id="degr_afa_v2_2024",
                    gueltig_ab=date(2024, 3, 28),
                    gueltig_bis=None,
                    veranlagungszeitraum_ab=2024,
                    veranlagungszeitraum_bis=None,
                    aenderungsgesetz="Wachstumschancengesetz",
                    aenderungstyp=Aenderungstyp.EINFUEGUNG,
                    bundesgesetzblatt="BGBl. I 2024, S. 108",
                    hinweis="20% statt 25%; auch Wohngebäude mit 5%",
                ),
            },
        ],
    )
    rules.append(rule_afa)

    return rules


# ===========================
# Versioned Retriever
# ===========================

class VersionAwareRetriever:
    """
    Tax law retriever that respects temporal validity.

    Key principle: A query for VZ 2020 must NOT return the 2024 version
    of a rule, and vice versa.

    Usage:
        retriever = VersionAwareRetriever()
        retriever.load_versioned_rules()
        results = retriever.query("Verlustuntergang bei Anteilsübertragung", tax_year=2022)
    """

    def __init__(self, min_relevance: float = 0.3):
        self.min_relevance = min_relevance
        self.rules: List[VersionedRule] = []
        self._encoder = None  # Lazy load
        self._today = date.today()

    def load_versioned_rules(self, rules: Optional[List[VersionedRule]] = None):
        """Load versioned rules (or use built-in samples)."""
        if rules is not None:
            self.rules = rules
        else:
            self.rules = _build_sample_versioned_rules()
            print(f"ℹ️  {len(self.rules)} versionierte Vorschriften geladen "
                  f"(gesamt {sum(len(r.versions) for r in self.rules)} Versionen)")

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        return self._encoder

    def query(
        self,
        question: str,
        tax_year: Optional[int] = None,
        reference_date: Optional[date] = None,
        top_k: int = 3,
    ) -> List[Dict]:
        """
        Query with temporal awareness.

        Parameters
        ----------
        question : search query
        tax_year : Veranlagungszeitraum (e.g. 2023)
        reference_date : specific date for validity check
        top_k : max results

        Returns
        -------
        List of results with version metadata and freshness warnings
        """
        if not self.rules:
            raise ValueError("No rules loaded. Call load_versioned_rules() first.")

        # Default: current year if nothing specified
        if tax_year is None and reference_date is None:
            tax_year = self._today.year
            warnings.warn(
                f"Kein Veranlagungszeitraum angegeben. Verwende aktuelles Jahr: {tax_year}. "
                f"Für andere Jahre: query(..., tax_year=2021)",
                UserWarning,
            )

        # Get applicable versions
        candidates = []
        for rule in self.rules:
            if tax_year:
                version = rule.get_version_for_year(tax_year)
            elif reference_date:
                version = rule.get_version_for_date(reference_date)
            else:
                version = None

            if version:
                candidates.append({
                    "rule_id": rule.rule_id,
                    "gesetz": rule.gesetz,
                    "paragraph": rule.paragraph,
                    "title": rule.title,
                    "text": version["text"],
                    "metadata": version["metadata"],
                    "has_newer_version": self._has_newer_version(rule, tax_year or self._today.year),
                })

        if not candidates:
            return [{
                "warning": f"Keine gültige Vorschrift für VZ {tax_year or reference_date} gefunden.",
                "hinweis": "Überprüfen Sie den Veranlagungszeitraum.",
            }]

        # Semantic ranking
        encoder = self._get_encoder()
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        q_emb = encoder.encode([question])
        texts = [c["text"] for c in candidates]
        t_embs = encoder.encode(texts)
        sims = cosine_similarity(q_emb, t_embs)[0]

        # Sort by relevance
        ranked = sorted(
            zip(candidates, sims),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        results = []
        for cand, score in ranked:
            if score < self.min_relevance:
                continue

            meta = cand["metadata"]
            result = {
                "rule_id": cand["rule_id"],
                "gesetz": cand["gesetz"],
                "paragraph": cand["paragraph"],
                "title": cand["title"],
                "text": cand["text"],
                "relevance": float(score),
                "version_id": meta.version_id,
                "gueltig_ab": meta.gueltig_ab.isoformat(),
                "gueltig_bis": meta.gueltig_bis.isoformat() if meta.gueltig_bis else "aktuell gültig",
                "aenderungsgesetz": meta.aenderungsgesetz,
                "veranlagungszeitraum": f"ab VZ {meta.veranlagungszeitraum_ab}" + (
                    f" bis VZ {meta.veranlagungszeitraum_bis}" if meta.veranlagungszeitraum_bis else " (laufend)"
                ),
            }

            # Freshness warnings
            freshness_warnings = self._check_freshness(meta, tax_year)
            if freshness_warnings:
                result["warnungen"] = freshness_warnings

            if cand["has_newer_version"]:
                result["hinweis_neuere_fassung"] = (
                    "⚠️ Es existiert eine neuere Fassung dieser Vorschrift. "
                    "Prüfen Sie, ob die neuere Version für Ihren Fall relevant ist."
                )

            results.append(result)

        if not results:
            return [{
                "warning": "Keine ausreichend relevanten Vorschriften gefunden.",
                "max_similarity": float(max(sims)) if len(sims) > 0 else 0.0,
            }]

        return results

    def _has_newer_version(self, rule: VersionedRule, tax_year: int) -> bool:
        """Check if there's a newer version than the one for this tax year."""
        current = rule.get_version_for_year(tax_year)
        if not current:
            return False
        latest = max(rule.versions, key=lambda v: v["metadata"].gueltig_ab)
        return latest["metadata"].version_id != current["metadata"].version_id

    def _check_freshness(self, meta: VersionMetadata, tax_year: Optional[int]) -> List[str]:
        """Generate warnings about potentially outdated information."""
        warns = []

        # Rule is more than 5 years old without update
        age_years = (self._today - meta.gueltig_ab).days / 365.25
        if age_years > 5 and meta.gueltig_bis is None:
            warns.append(
                f"Diese Fassung ist seit {meta.gueltig_ab.isoformat()} in Kraft "
                f"({age_years:.0f} Jahre). Prüfen Sie aktuelle Rechtsprechung und BMF-Schreiben."
            )

        # BVerfG ruling → extra caution
        if meta.aenderungstyp == Aenderungstyp.VERFASSUNGSWIDRIG:
            warns.append(
                "⚠️ Diese Vorschrift wurde (teilweise) für verfassungswidrig erklärt. "
                "Prüfen Sie die Folgegesetzgebung."
            )

        # Future-dated rule being queried for current year
        if tax_year and meta.veranlagungszeitraum_ab and tax_year < meta.veranlagungszeitraum_ab:
            warns.append(
                f"Diese Fassung gilt erst ab VZ {meta.veranlagungszeitraum_ab}, "
                f"Sie fragen aber nach VZ {tax_year}."
            )

        return warns

    def get_change_history(self, rule_id: str) -> List[Dict]:
        """Get full amendment history for a specific rule."""
        for rule in self.rules:
            if rule.rule_id == rule_id:
                return rule.get_change_history()
        return []

    def compare_versions(self, rule_id: str, year_a: int, year_b: int) -> Dict:
        """
        Compare two versions of a rule across different tax years.
        Useful for: "What changed between VZ 2019 and VZ 2024?"
        """
        for rule in self.rules:
            if rule.rule_id == rule_id:
                v_a = rule.get_version_for_year(year_a)
                v_b = rule.get_version_for_year(year_b)

                result = {
                    "rule_id": rule_id,
                    "paragraph": rule.paragraph,
                    "comparison": f"VZ {year_a} vs. VZ {year_b}",
                }

                if v_a is None:
                    result[f"vz_{year_a}"] = "Keine gültige Fassung"
                else:
                    result[f"vz_{year_a}"] = {
                        "text": v_a["text"],
                        "version_id": v_a["metadata"].version_id,
                        "aenderungsgesetz": v_a["metadata"].aenderungsgesetz,
                    }

                if v_b is None:
                    result[f"vz_{year_b}"] = "Keine gültige Fassung"
                else:
                    result[f"vz_{year_b}"] = {
                        "text": v_b["text"],
                        "version_id": v_b["metadata"].version_id,
                        "aenderungsgesetz": v_b["metadata"].aenderungsgesetz,
                    }

                if v_a and v_b:
                    result["has_changed"] = v_a["metadata"].version_id != v_b["metadata"].version_id
                else:
                    result["has_changed"] = True

                return result

        return {"error": f"Rule '{rule_id}' not found."}


# ===========================
# Hebesatz Versioning
# ===========================

class HebesatzTracker:
    """
    Track municipal Hebesatz changes over time.
    Hebesätze change frequently (annual budget decisions by Gemeinderat).

    Example: Monheim dropped from 435% to 250% in 2012, causing controversy.
    """

    def __init__(self):
        self._data: Dict[str, List[Dict]] = {}

    def load_sample_data(self):
        """Load sample Hebesatz history for major cities."""
        self._data = {
            "München": [
                {"year": 2016, "hebesatz": 490},
                {"year": 2024, "hebesatz": 490},  # unchanged
            ],
            "Frankfurt am Main": [
                {"year": 2016, "hebesatz": 460},
                {"year": 2024, "hebesatz": 460},
            ],
            "Berlin": [
                {"year": 2016, "hebesatz": 410},
                {"year": 2024, "hebesatz": 410},
            ],
            "Monheim am Rhein": [
                {"year": 2009, "hebesatz": 435},
                {"year": 2012, "hebesatz": 300},
                {"year": 2013, "hebesatz": 250},  # Massive reduction
                {"year": 2024, "hebesatz": 250},
            ],
            "Offenbach am Main": [
                {"year": 2016, "hebesatz": 500},
                {"year": 2020, "hebesatz": 500},
            ],
            "Grünwald": [  # Famous low-tax Munich suburb
                {"year": 2016, "hebesatz": 240},
                {"year": 2024, "hebesatz": 240},
            ],
            "Leverkusen": [
                {"year": 2016, "hebesatz": 475},
                {"year": 2023, "hebesatz": 250},  # Drastic reduction
                {"year": 2024, "hebesatz": 250},
            ],
        }
        print(f"ℹ️  Hebesatz-Daten für {len(self._data)} Gemeinden geladen.")

    def get_hebesatz(self, gemeinde: str, year: int) -> Optional[int]:
        """Get the applicable Hebesatz for a municipality and year."""
        if gemeinde not in self._data:
            return None

        entries = sorted(self._data[gemeinde], key=lambda x: x["year"])
        applicable = None
        for entry in entries:
            if entry["year"] <= year:
                applicable = entry["hebesatz"]
        return applicable

    def get_history(self, gemeinde: str) -> List[Dict]:
        """Full Hebesatz history for a municipality."""
        return self._data.get(gemeinde, [])

    def find_changes(self, gemeinde: str, from_year: int, to_year: int) -> Dict:
        """Detect Hebesatz changes in a period."""
        hs_from = self.get_hebesatz(gemeinde, from_year)
        hs_to = self.get_hebesatz(gemeinde, to_year)

        return {
            "gemeinde": gemeinde,
            "from_year": from_year,
            "to_year": to_year,
            "hebesatz_from": hs_from,
            "hebesatz_to": hs_to,
            "changed": hs_from != hs_to,
            "difference": (hs_to - hs_from) if (hs_from and hs_to) else None,
        }


# ===========================
# Integration: Version-Aware Pipeline
# ===========================

class GermanTaxVersionPipeline:
    """
    Complete version-aware tax analysis pipeline.

    Ensures that:
    1. Model predictions are interpreted against the correct statutory rate (year-specific)
    2. SHAP explanations link to the correct version of the law
    3. Users are warned about law changes that affect their analysis
    """

    def __init__(self):
        self.retriever = VersionAwareRetriever()
        self.hebesatz_tracker = HebesatzTracker()
        self._initialized = False

    def initialize(self):
        self.retriever.load_versioned_rules()
        self.hebesatz_tracker.load_sample_data()
        self._initialized = True

    def analyze_for_year(
        self,
        question: str,
        tax_year: int,
        gemeinde: Optional[str] = None,
        top_k: int = 3,
    ) -> Dict:
        """
        Full analysis anchored to a specific Veranlagungszeitraum.

        Returns:
        - Applicable law version
        - Statutory rate for that year + municipality
        - Warnings about law changes
        - Change history
        """
        if not self._initialized:
            self.initialize()

        result = {
            "veranlagungszeitraum": tax_year,
            "abfrage_datum": datetime.now().isoformat(),
        }

        # Statutory rate
        hebesatz = 400  # Default
        if gemeinde:
            hs = self.hebesatz_tracker.get_hebesatz(gemeinde, tax_year)
            if hs:
                hebesatz = hs
                result["gemeinde"] = gemeinde
                result["hebesatz"] = hs

                # Check if Hebesatz changed recently
                hs_prev = self.hebesatz_tracker.get_hebesatz(gemeinde, tax_year - 1)
                if hs_prev and hs_prev != hs:
                    result["hebesatz_warnung"] = (
                        f"⚠️ Hebesatz in {gemeinde} hat sich geändert: "
                        f"{hs_prev}% (VZ {tax_year-1}) → {hs}% (VZ {tax_year})"
                    )

        from german_tax_models import compute_statutory_rate
        result["gesetzlicher_steuersatz"] = compute_statutory_rate(hebesatz)

        # Legal provisions
        law_results = self.retriever.query(question, tax_year=tax_year, top_k=top_k)
        result["rechtsgrundlagen"] = law_results

        # Check for upcoming changes
        future_results = self.retriever.query(question, tax_year=tax_year + 1, top_k=1)
        if future_results and "warning" not in future_results[0]:
            current_version = law_results[0].get("version_id") if law_results and "version_id" in law_results[0] else None
            future_version = future_results[0].get("version_id")
            if current_version and future_version and current_version != future_version:
                result["vorschau_naechstes_jahr"] = {
                    "hinweis": f"⚠️ Für VZ {tax_year + 1} gilt eine andere Fassung!",
                    "neue_fassung": future_results[0].get("aenderungsgesetz", ""),
                }

        return result

    def version_comparison_report(self, rule_id: str, year_a: int, year_b: int) -> str:
        """Generate a human-readable comparison report."""
        if not self._initialized:
            self.initialize()

        comp = self.retriever.compare_versions(rule_id, year_a, year_b)

        if "error" in comp:
            return comp["error"]

        lines = [
            f"═══ Versionsvergleich: {comp['paragraph']} ═══",
            f"Vergleich: VZ {year_a} vs. VZ {year_b}",
            f"Änderung: {'JA' if comp['has_changed'] else 'NEIN'}",
            "",
        ]

        for key in [f"vz_{year_a}", f"vz_{year_b}"]:
            lines.append(f"── {key.upper()} ──")
            val = comp[key]
            if isinstance(val, str):
                lines.append(f"  {val}")
            else:
                lines.append(f"  Version: {val['version_id']}")
                lines.append(f"  Änderungsgesetz: {val['aenderungsgesetz']}")
                lines.append(f"  Text: {val['text'][:200]}...")
            lines.append("")

        return "\n".join(lines)
