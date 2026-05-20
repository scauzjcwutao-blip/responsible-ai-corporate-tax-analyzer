"""
German Tax Law Version Download & Management
==============================================
Provides mechanisms to:
1. Download the LATEST version of all tax rules (aktuelle Fassung)
2. Download historical versions for any Veranlagungszeitraum
3. Export snapshots (JSON, CSV, Excel)
4. Incremental updates (delta sync)
5. Offline cache with integrity checks

Data Sources (Production):
- Bundesgesetzblatt (BGBl) via API
- Gesetze-im-Internet.de (BMJ official)
- dejure.org (structured legal texts)
- buzer.de (change history/Synopsen)
- ELSTER/ERiC interfaces
- Gemeinde-Hebesatz via DESTATIS

This module provides the framework and a simulation layer.
In production, replace SimulatedSource with real API connectors.

Author: Tax AI System
License: Internal Use
"""

import json
import hashlib
import shutil
import logging
import zipfile
import csv
from pathlib import Path
from datetime import date, datetime
from typing import Optional, List, Dict, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import io

logger = logging.getLogger("german_tax_download")


# ===========================
# Configuration
# ===========================

class DownloadConfig:
    """Central configuration for download paths and sources."""

    BASE_DIR = Path("tax_law_database")
    LATEST_DIR = BASE_DIR / "aktuell"
    HISTORY_DIR = BASE_DIR / "historisch"
    EXPORT_DIR = BASE_DIR / "export"
    MANIFEST_FILE = BASE_DIR / "manifest.json"
    HEBESATZ_DIR = BASE_DIR / "hebesaetze"

    # Supported export formats
    FORMATS = ["json", "csv", "xlsx", "markdown"]

    # Source URLs (production – replace with real endpoints)
    SOURCES = {
        "gesetze_im_internet": "https://www.gesetze-im-internet.de/",
        "dejure": "https://dejure.org/gesetze/",
        "buzer_synopse": "https://www.buzer.de/",
        "destatis_hebesaetze": "https://www.destatis.de/",
        "bundesgesetzblatt": "https://www.bgbl.de/",
    }

    @classmethod
    def ensure_dirs(cls):
        """Create all necessary directories."""
        for d in [cls.BASE_DIR, cls.LATEST_DIR, cls.HISTORY_DIR, 
                  cls.EXPORT_DIR, cls.HEBESATZ_DIR]:
            d.mkdir(parents=True, exist_ok=True)


# ===========================
# Version Manifest
# ===========================

@dataclass
class ManifestEntry:
    """Tracks what's been downloaded and when."""
    rule_id: str
    gesetz: str
    paragraph: str
    version_id: str
    gueltig_ab: str
    gueltig_bis: Optional[str]
    download_timestamp: str
    checksum_sha256: str
    source: str
    file_path: str


class VersionManifest:
    """
    Tracks all downloaded versions with integrity checksums.
    Enables: "What do I have locally?" and "Is my data up-to-date?"
    """

    def __init__(self, manifest_path: Path = DownloadConfig.MANIFEST_FILE):
        self.path = manifest_path
        self.entries: List[ManifestEntry] = []
        self._load()

    def _load(self):
        if self.path.exists():
            with open(self.path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            self.entries = [ManifestEntry(**e) for e in raw.get("entries", [])]
        else:
            self.entries = []

    def save(self):
        data = {
            "schema_version": "1.0",
            "last_updated": datetime.now().isoformat(),
            "total_entries": len(self.entries),
            "entries": [asdict(e) for e in self.entries],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def add(self, entry: ManifestEntry):
        # Remove old entry for same rule+version if exists
        self.entries = [
            e for e in self.entries
            if not (e.rule_id == entry.rule_id and e.version_id == entry.version_id)
        ]
        self.entries.append(entry)
        self.save()

    def get_latest_for_rule(self, rule_id: str) -> Optional[ManifestEntry]:
        """Get the most recently downloaded version of a rule."""
        matches = [e for e in self.entries if e.rule_id == rule_id]
        if not matches:
            return None
        return sorted(matches, key=lambda e: e.download_timestamp, reverse=True)[0]

    def get_all_for_rule(self, rule_id: str) -> List[ManifestEntry]:
        """Get all downloaded versions of a rule."""
        return sorted(
            [e for e in self.entries if e.rule_id == rule_id],
            key=lambda e: e.gueltig_ab,
        )

    def get_rules_by_gesetz(self, gesetz: str) -> List[ManifestEntry]:
        return [e for e in self.entries if e.gesetz.upper() == gesetz.upper()]

    def summary(self) -> Dict:
        gesetze = set(e.gesetz for e in self.entries)
        return {
            "total_versions": len(self.entries),
            "unique_rules": len(set(e.rule_id for e in self.entries)),
            "gesetze": sorted(gesetze),
            "last_download": max((e.download_timestamp for e in self.entries), default=None),
        }


# ===========================
# Download Source Interface
# ===========================

class TaxLawSource:
    """
    Abstract interface for a tax law data source.
    In production, implement concrete classes for each API.
    """

    def fetch_latest(self, gesetz: str, paragraph: str) -> Optional[Dict]:
        raise NotImplementedError

    def fetch_version(self, gesetz: str, paragraph: str, tax_year: int) -> Optional[Dict]:
        raise NotImplementedError

    def fetch_all_versions(self, gesetz: str, paragraph: str) -> List[Dict]:
        raise NotImplementedError

    def fetch_hebesaetze(self, year: int) -> Dict[str, int]:
        raise NotImplementedError

    def check_updates_since(self, last_check: datetime) -> List[Dict]:
        raise NotImplementedError


class SimulatedSource(TaxLawSource):
    """
    Simulated data source using built-in German tax rules.
    Replace with real API connectors in production.
    """

    def __init__(self):
        from german_tax_versioning import _build_sample_versioned_rules, HebesatzTracker
        self._rules = _build_sample_versioned_rules()
        self._hebesatz = HebesatzTracker()
        self._hebesatz.load_sample_data()

    def fetch_latest(self, gesetz: str, paragraph: str) -> Optional[Dict]:
        for rule in self._rules:
            if rule.gesetz.upper() == gesetz.upper() and paragraph.lower() in rule.paragraph.lower():
                # Return the most recent version
                latest = max(rule.versions, key=lambda v: v["metadata"].gueltig_ab)
                return self._format_version(rule, latest)
        return None

    def fetch_version(self, gesetz: str, paragraph: str, tax_year: int) -> Optional[Dict]:
        for rule in self._rules:
            if rule.gesetz.upper() == gesetz.upper() and paragraph.lower() in rule.paragraph.lower():
                version = rule.get_version_for_year(tax_year)
                if version:
                    return self._format_version(rule, version)
        return None

    def fetch_all_versions(self, gesetz: str, paragraph: str) -> List[Dict]:
        results = []
        for rule in self._rules:
            if rule.gesetz.upper() == gesetz.upper() and paragraph.lower() in rule.paragraph.lower():
                for v in rule.versions:
                    results.append(self._format_version(rule, v))
        return results

    def fetch_hebesaetze(self, year: int) -> Dict[str, int]:
        result = {}
        for gemeinde in self._hebesatz._data:
            hs = self._hebesatz.get_hebesatz(gemeinde, year)
            if hs:
                result[gemeinde] = hs
        return result

    def check_updates_since(self, last_check: datetime) -> List[Dict]:
        """Simulate checking for updates (in production: compare BGBl dates)."""
        # Simulate: rules updated after last_check
        updates = []
        for rule in self._rules:
            for v in rule.versions:
                # Simulate: if gueltig_ab is after last_check date
                if v["metadata"].gueltig_ab > last_check.date():
                    updates.append({
                        "rule_id": rule.rule_id,
                        "paragraph": rule.paragraph,
                        "version_id": v["metadata"].version_id,
                        "gueltig_ab": v["metadata"].gueltig_ab.isoformat(),
                        "aenderungsgesetz": v["metadata"].aenderungsgesetz,
                    })
        return updates

    def _format_version(self, rule, version) -> Dict:
        meta = version["metadata"]
        return {
            "rule_id": rule.rule_id,
            "gesetz": rule.gesetz,
            "paragraph": rule.paragraph,
            "title": rule.title,
            "text": version["text"],
            "version_id": meta.version_id,
            "gueltig_ab": meta.gueltig_ab.isoformat(),
            "gueltig_bis": meta.gueltig_bis.isoformat() if meta.gueltig_bis else None,
            "veranlagungszeitraum_ab": meta.veranlagungszeitraum_ab,
            "veranlagungszeitraum_bis": meta.veranlagungszeitraum_bis,
            "aenderungsgesetz": meta.aenderungsgesetz,
            "aenderungstyp": meta.aenderungstyp.value,
            "rechtsquelle": meta.rechtsquelle.value,
            "bundesgesetzblatt": meta.bundesgesetzblatt,
            "hinweis": meta.hinweis,
        }

    def list_available(self) -> List[Dict]:
        """List all available rules and their versions."""
        result = []
        for rule in self._rules:
            result.append({
                "rule_id": rule.rule_id,
                "gesetz": rule.gesetz,
                "paragraph": rule.paragraph,
                "title": rule.title,
                "n_versions": len(rule.versions),
                "versions": [v["metadata"].version_id for v in rule.versions],
            })
        return result


# ===========================
# Download Manager
# ===========================

class GermanTaxDownloadManager:
    """
    Main interface for downloading and managing tax law versions.

    Usage:
        manager = GermanTaxDownloadManager()
        
        # Download latest version of everything
        manager.download_latest_all()
        
        # Download specific historical version
        manager.download_for_year("KStG", "§8c", 2015)
        
        # Download complete history of a paragraph
        manager.download_all_versions("KStG", "§8c")
        
        # Export for offline use
        manager.export_snapshot(tax_year=2023, format="json")
        
        # Check for updates
        manager.check_and_update()
    """

    def __init__(self, source: Optional[TaxLawSource] = None):
        DownloadConfig.ensure_dirs()
        self.source = source or SimulatedSource()
        self.manifest = VersionManifest()
        self._last_check: Optional[datetime] = None

    # ===========================
    # Download: Latest Version
    # ===========================

    def download_latest(self, gesetz: str, paragraph: str) -> Optional[Path]:
        """
        Download the current/latest version of a specific paragraph.
        
        Returns the path to the saved file.
        """
        print(f"⬇️  Downloading latest: {paragraph} {gesetz}...")

        data = self.source.fetch_latest(gesetz, paragraph)
        if data is None:
            print(f"   ❌ Not found: {paragraph} {gesetz}")
            return None

        # Save to latest directory
        file_path = self._save_rule(data, DownloadConfig.LATEST_DIR)

        # Update manifest
        checksum = self._compute_checksum(data)
        self.manifest.add(ManifestEntry(
            rule_id=data["rule_id"],
            gesetz=data["gesetz"],
            paragraph=data["paragraph"],
            version_id=data["version_id"],
            gueltig_ab=data["gueltig_ab"],
            gueltig_bis=data.get("gueltig_bis"),
            download_timestamp=datetime.now().isoformat(),
            checksum_sha256=checksum,
            source="latest",
            file_path=str(file_path),
        ))

        print(f"   ✅ Saved: {file_path.name} (Version: {data['version_id']})")
        return file_path

    def download_latest_all(self) -> List[Path]:
        """Download the latest version of ALL available rules."""
        print("=" * 60)
        print("⬇️  DOWNLOADING ALL LATEST VERSIONS")
        print("=" * 60)

        available = self.source.list_available()
        paths = []

        for rule_info in available:
            path = self.download_latest(rule_info["gesetz"], rule_info["paragraph"])
            if path:
                paths.append(path)

        print(f"\n✅ Downloaded {len(paths)} aktuelle Fassungen.")
        return paths

    # ===========================
    # Download: Historical Versions
    # ===========================

    def download_for_year(self, gesetz: str, paragraph: str, tax_year: int) -> Optional[Path]:
        """
        Download the version applicable to a specific Veranlagungszeitraum.
        """
        print(f"⬇️  Downloading {paragraph} {gesetz} for VZ {tax_year}...")

        data = self.source.fetch_version(gesetz, paragraph, tax_year)
        if data is None:
            print(f"   ❌ No version found for VZ {tax_year}")
            return None

        # Save to history directory under year subfolder
        year_dir = DownloadConfig.HISTORY_DIR / f"VZ_{tax_year}"
        year_dir.mkdir(parents=True, exist_ok=True)
        file_path = self._save_rule(data, year_dir)

        # Manifest
        checksum = self._compute_checksum(data)
        self.manifest.add(ManifestEntry(
            rule_id=data["rule_id"],
            gesetz=data["gesetz"],
            paragraph=data["paragraph"],
            version_id=data["version_id"],
            gueltig_ab=data["gueltig_ab"],
            gueltig_bis=data.get("gueltig_bis"),
            download_timestamp=datetime.now().isoformat(),
            checksum_sha256=checksum,
            source=f"historical_VZ_{tax_year}",
            file_path=str(file_path),
        ))

        print(f"   ✅ Saved: VZ_{tax_year}/{file_path.name}")
        return file_path

    def download_all_versions(self, gesetz: str, paragraph: str) -> List[Path]:
        """
        Download the COMPLETE version history of a paragraph.
        Essential for audit trails and Synopsen.
        """
        print(f"⬇️  Downloading all versions: {paragraph} {gesetz}...")

        versions = self.source.fetch_all_versions(gesetz, paragraph)
        if not versions:
            print(f"   ❌ No versions found")
            return []

        paths = []
        history_subdir = DownloadConfig.HISTORY_DIR / f"{gesetz}_{paragraph.replace('§', 'P').replace(' ', '_')}"
        history_subdir.mkdir(parents=True, exist_ok=True)

        for data in versions:
            file_path = self._save_rule(data, history_subdir, include_version_in_name=True)
            paths.append(file_path)

            checksum = self._compute_checksum(data)
            self.manifest.add(ManifestEntry(
                rule_id=data["rule_id"],
                gesetz=data["gesetz"],
                paragraph=data["paragraph"],
                version_id=data["version_id"],
                gueltig_ab=data["gueltig_ab"],
                gueltig_bis=data.get("gueltig_bis"),
                download_timestamp=datetime.now().isoformat(),
                checksum_sha256=checksum,
                source="full_history",
                file_path=str(file_path),
            ))

        print(f"   ✅ {len(paths)} Versionen heruntergeladen.")
        return paths

    def download_year_snapshot(self, tax_year: int) -> List[Path]:
        """
        Download ALL rules as they applied in a specific year.
        Creates a complete "state of the law" snapshot for that year.
        """
        print(f"\n{'=' * 60}")
        print(f"📸 SNAPSHOT: Gesamte Rechtslage VZ {tax_year}")
        print(f"{'=' * 60}")

        available = self.source.list_available()
        paths = []

        for rule_info in available:
            path = self.download_for_year(
                rule_info["gesetz"],
                rule_info["paragraph"],
                tax_year,
            )
            if path:
                paths.append(path)

        print(f"\n✅ Snapshot VZ {tax_year}: {len(paths)} Vorschriften gesichert.")
        return paths

    # ===========================
    # Download: Hebesätze
    # ===========================

    def download_hebesaetze(self, year: int) -> Optional[Path]:
        """Download municipal Hebesätze for a specific year."""
        print(f"⬇️  Downloading Hebesätze for {year}...")

        data = self.source.fetch_hebesaetze(year)
        if not data:
            print("   ❌ No Hebesatz data available")
            return None

        file_path = DownloadConfig.HEBESATZ_DIR / f"hebesaetze_{year}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump({
                "year": year,
                "download_timestamp": datetime.now().isoformat(),
                "n_gemeinden": len(data),
                "data": data,
            }, f, indent=2, ensure_ascii=False)

        print(f"   ✅ {len(data)} Gemeinden → {file_path.name}")
        return file_path

    # ===========================
    # Update Check
    # ===========================

    def check_and_update(self) -> Dict:
        """
        Check for updates since last download and fetch new versions.
        
        Returns summary of changes found.
        """
        print("\n🔄 Checking for updates...")

        if self._last_check is None:
            # Default: check for anything from the last 30 days
            from datetime import timedelta
            self._last_check = datetime.now() - timedelta(days=30)

        updates = self.source.check_updates_since(self._last_check)

        if not updates:
            print("   ✅ Alles auf dem neuesten Stand.")
            self._last_check = datetime.now()
            return {"status": "up_to_date", "updates": []}

        print(f"   ⚠️  {len(updates)} Aktualisierungen gefunden:")
        for u in updates:
            print(f"      • {u['paragraph']} ({u['aenderungsgesetz']})")

        # Auto-download updates
        downloaded = []
        for u in updates:
            gesetz = u["rule_id"].split("_")[0] if "_" in u["rule_id"] else "KStG"
            path = self.download_latest(gesetz, u["paragraph"])
            if path:
                downloaded.append(u)

        self._last_check = datetime.now()

        return {
            "status": "updated",
            "updates_found": len(updates),
            "updates_downloaded": len(downloaded),
            "details": updates,
        }

    # ===========================
    # Export
    # ===========================

    def export_snapshot(
        self,
        tax_year: int,
        format: str = "json",
        include_history: bool = False,
    ) -> Path:
        """
        Export a complete snapshot for a tax year in the specified format.
        
        Formats: json, csv, markdown
        """
        if format not in DownloadConfig.FORMATS:
            raise ValueError(f"Unsupported format '{format}'. Use: {DownloadConfig.FORMATS}")

        print(f"\n📤 Exporting VZ {tax_year} snapshot as {format.upper()}...")

        # Collect all rules for this year
        available = self.source.list_available()
        rules_data = []

        for rule_info in available:
            data = self.source.fetch_version(
                rule_info["gesetz"], rule_info["paragraph"], tax_year
            )
            if data:
                rules_data.append(data)

        if not rules_data:
            print("   ❌ No data to export")
            return Path("")

        # Also collect Hebesätze
        hebesaetze = self.source.fetch_hebesaetze(tax_year)

        # Export
        export_dir = DownloadConfig.EXPORT_DIR / f"VZ_{tax_year}"
        export_dir.mkdir(parents=True, exist_ok=True)

        if format == "json":
            return self._export_json(rules_data, hebesaetze, tax_year, export_dir)
        elif format == "csv":
            return self._export_csv(rules_data, hebesaetze, tax_year, export_dir)
        elif format == "markdown":
            return self._export_markdown(rules_data, hebesaetze, tax_year, export_dir)
        else:
            return self._export_json(rules_data, hebesaetze, tax_year, export_dir)

    def _export_json(self, rules, hebesaetze, year, export_dir) -> Path:
        output = {
            "meta": {
                "veranlagungszeitraum": year,
                "export_timestamp": datetime.now().isoformat(),
                "n_rules": len(rules),
                "system": "German Tax AI v2.0",
            },
            "rechtsvorschriften": rules,
            "hebesaetze": hebesaetze,
        }
        path = export_dir / f"steuerrecht_VZ{year}.json"
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"   ✅ Exported: {path}")
        return path

    def _export_csv(self, rules, hebesaetze, year, export_dir) -> Path:
        path = export_dir / f"steuerrecht_VZ{year}.csv"
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=[
                "rule_id", "gesetz", "paragraph", "title", "version_id",
                "gueltig_ab", "gueltig_bis", "aenderungsgesetz", "text",
            ])
            writer.writeheader()
            for r in rules:
                writer.writerow({
                    "rule_id": r.get("rule_id", ""),
                    "gesetz": r.get("gesetz", ""),
                    "paragraph": r.get("paragraph", ""),
                    "title": r.get("title", ""),
                    "version_id": r.get("version_id", ""),
                    "gueltig_ab": r.get("gueltig_ab", ""),
                    "gueltig_bis": r.get("gueltig_bis", ""),
                    "aenderungsgesetz": r.get("aenderungsgesetz", ""),
                    "text": r.get("text", ""),
                })
        print(f"   ✅ Exported: {path}")
        return path

    def _export_markdown(self, rules, hebesaetze, year, export_dir) -> Path:
        path = export_dir / f"steuerrecht_VZ{year}.md"
        with open(path, 'w', encoding='utf-8') as f:
            f.write(f"# Deutsches Steuerrecht – Veranlagungszeitraum {year}\n\n")
            f.write(f"*Exportiert am {datetime.now().strftime('%d.%m.%Y %H:%M')}*\n\n")
            f.write("---\n\n")

            # Group by Gesetz
            by_gesetz = {}
            for r in rules:
                g = r.get("gesetz", "Sonstiges")
                by_gesetz.setdefault(g, []).append(r)

            for gesetz, gesetz_rules in sorted(by_gesetz.items()):
                f.write(f"## {gesetz}\n\n")
                for r in gesetz_rules:
                    f.write(f"### {r['paragraph']} – {r.get('title', '')}\n\n")
                    f.write(f"**Version:** {r.get('version_id', '?')}  \n")
                    f.write(f"**Gültig ab:** {r.get('gueltig_ab', '?')}  \n")
                    f.write(f"**Änderungsgesetz:** {r.get('aenderungsgesetz', '?')}  \n\n")
                    f.write(f"> {r.get('text', '')}\n\n")
                    if r.get("hinweis"):
                        f.write(f"*Hinweis: {r['hinweis']}*\n\n")
                    f.write("---\n\n")

            # Hebesätze
            if hebesaetze:
                f.write("## Gewerbesteuer-Hebesätze\n\n")
                f.write("| Gemeinde | Hebesatz |\n|---|---|\n")
                for gemeinde, hs in sorted(hebesaetze.items()):
                    f.write(f"| {gemeinde} | {hs}% |\n")

        print(f"   ✅ Exported: {path}")
        return path

    def export_zip_archive(self, tax_year: int) -> Path:
        """
        Create a ZIP archive with all formats for easy distribution.
        Ideal for: Wirtschaftsprüfer, Steuerberater, Archive.
        """
        print(f"\n📦 Creating ZIP archive for VZ {tax_year}...")

        # Export all formats first
        self.export_snapshot(tax_year, format="json")
        self.export_snapshot(tax_year, format="csv")
        self.export_snapshot(tax_year, format="markdown")

        # Create ZIP
        export_dir = DownloadConfig.EXPORT_DIR / f"VZ_{tax_year}"
        zip_path = DownloadConfig.EXPORT_DIR / f"steuerrecht_VZ{tax_year}_komplett.zip"

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file in export_dir.glob("*"):
                if file.suffix != ".zip":
                    zf.write(file, file.name)

            # Add manifest
            manifest_data = json.dumps(
                self.manifest.summary(), indent=2, ensure_ascii=False
            )
            zf.writestr("manifest_info.json", manifest_data)

            # Add README
            readme = self._generate_readme(tax_year)
            zf.writestr("README.md", readme)

        print(f"   ✅ Archive: {zip_path} ({zip_path.stat().st_size / 1024:.1f} KB)")
        return zip_path

    def _generate_readme(self, tax_year: int) -> str:
        return f"""# Steuerrecht-Datenpaket VZ {tax_year}

## Inhalt

- `steuerrecht_VZ{tax_year}.json` – Maschinenlesbare Version (für AI/ML-Systeme)
- `steuerrecht_VZ{tax_year}.csv` – Tabellarische Version (Excel-kompatibel)
- `steuerrecht_VZ{tax_year}.md` – Menschenlesbare Version (Markdown)
- `manifest_info.json` – Metadaten und Prüfsummen

## Veranlagungszeitraum

Dieses Paket enthält die für VZ {tax_year} geltenden Fassungen der
relevanten Steuervorschriften (KStG, GewStG, EStG, AStG, UmwStG).

## Haftungsausschluss

Dieses Datenpaket dient ausschließlich Informationszwecken.
Maßgeblich sind ausschließlich die im Bundesgesetzblatt veröffentlichten Texte.
Keine Gewähr für Vollständigkeit oder Aktualität.

## Aktualisierungen

Prüfen Sie regelmäßig auf Gesetzesänderungen:
- Jahressteuergesetz (jährlich, meist Q4)
- BMF-Schreiben (laufend)
- BFH-Urteile (laufend)

Erstellt am: {datetime.now().strftime('%d.%m.%Y %H:%M')}
System: German Tax AI v2.0
"""

    # ===========================
    # Helper Methods
    # ===========================

    def _save_rule(self, data: Dict, directory: Path, include_version_in_name: bool = False) -> Path:
        """Save a rule to disk as JSON."""
        name_parts = [
            data.get("gesetz", "unknown"),
            data.get("paragraph", "").replace("§", "P").replace(" ", "_"),
        ]
        if include_version_in_name:
            name_parts.append(data.get("version_id", "v0"))

        filename = "_".join(name_parts) + ".json"
        # Sanitize filename
        filename = "".join(c for c in filename if c.isalnum() or c in "._-")

        file_path = directory / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return file_path

    def _compute_checksum(self, data: Dict) -> str:
        """SHA-256 checksum for integrity verification."""
        content = json.dumps(data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    # ===========================
    # Status & Info
    # ===========================

    def status(self) -> Dict:
        """Overall status of the local database."""
        summary = self.manifest.summary()

        # Check disk usage
        total_size = sum(
            f.stat().st_size for f in DownloadConfig.BASE_DIR.rglob("*") if f.is_file()
        ) if DownloadConfig.BASE_DIR.exists() else 0

        return {
            **summary,
            "disk_usage_kb": total_size / 1024,
            "last_update_check": self._last_check.isoformat() if self._last_check else None,
            "base_directory": str(DownloadConfig.BASE_DIR),
        }

    def list_downloaded(self, gesetz: Optional[str] = None) -> List[Dict]:
        """List what's currently downloaded."""
        entries = self.manifest.entries
        if gesetz:
            entries = [e for e in entries if e.gesetz.upper() == gesetz.upper()]

        return [
            {
                "rule_id": e.rule_id,
                "paragraph": e.paragraph,
                "version_id": e.version_id,
                "gueltig_ab": e.gueltig_ab,
                "source": e.source,
                "downloaded": e.download_timestamp,
            }
            for e in entries
        ]

    def verify_integrity(self) -> Dict:
        """Verify checksums of all downloaded files."""
        print("🔍 Verifying integrity of downloaded files...")
        ok = 0
        failed = 0
        missing = 0

        for entry in self.manifest.entries:
            path = Path(entry.file_path)
            if not path.exists():
                missing += 1
                continue

            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            actual_checksum = self._compute_checksum(data)

            if actual_checksum == entry.checksum_sha256:
                ok += 1
            else:
                failed += 1
                logger.warning(f"Checksum mismatch: {path}")

        result = {
            "total": len(self.manifest.entries),
            "ok": ok,
            "failed": failed,
            "missing": missing,
            "integrity": "PASS" if (failed == 0 and missing == 0) else "FAIL",
        }
        print(f"   Result: {result['integrity']} ({ok} OK, {failed} failed, {missing} missing)")
        return result


# ===========================
# CLI Interface
# ===========================

def cli_main():
    """
    Command-line interface for tax law downloads.
    
    Usage:
        python german_tax_download.py latest           → Download all latest
        python german_tax_download.py year 2023        → Snapshot for VZ 2023
        python german_tax_download.py history KStG §8c → All versions of §8c
        python german_tax_download.py export 2023 json → Export VZ 2023 as JSON
        python german_tax_download.py status           → Show status
        python german_tax_download.py update           → Check for updates
        python german_tax_download.py zip 2023         → Create ZIP archive
    """
    import sys

    manager = GermanTaxDownloadManager()

    if len(sys.argv) < 2:
        print(cli_main.__doc__)
        return

    command = sys.argv[1].lower()

    if command == "latest":
        manager.download_latest_all()

    elif command == "year":
        if len(sys.argv) < 3:
            print("Usage: ... year <YYYY>")
            return
        year = int(sys.argv[2])
        manager.download_year_snapshot(year)

    elif command == "history":
        if len(sys.argv) < 4:
            print("Usage: ... history <Gesetz> <Paragraph>")
            return
        gesetz = sys.argv[2]
        paragraph = sys.argv[3]
        manager.download_all_versions(gesetz, paragraph)

    elif command == "export":
        if len(sys.argv) < 4:
            print("Usage: ... export <YYYY> <format>")
            return
        year = int(sys.argv[2])
        fmt = sys.argv[3]
        manager.export_snapshot(year, format=fmt)

    elif command == "zip":
        if len(sys.argv) < 3:
            print("Usage: ... zip <YYYY>")
            return
        year = int(sys.argv[2])
        manager.export_zip_archive(year)

    elif command == "status":
        status = manager.status()
        print(json.dumps(status, indent=2, ensure_ascii=False))

    elif command == "update":
        result = manager.check_and_update()
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))

    elif command == "verify":
        manager.verify_integrity()

    elif command == "hebesatz":
        if len(sys.argv) < 3:
            print("Usage: ... hebesatz <YYYY>")
            return
        year = int(sys.argv[2])
        manager.download_hebesaetze(year)

    else:
        print(f"Unknown command: {command}")
        print(cli_main.__doc__)


if __name__ == "__main__":
    cli_main()
