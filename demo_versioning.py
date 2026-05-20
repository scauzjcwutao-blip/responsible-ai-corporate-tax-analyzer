"""
Demo: German Tax Law Versioning
================================
Shows how the same query returns different results depending on the tax year.
"""

from datetime import date
from german_tax_versioning import (
    VersionAwareRetriever,
    HebesatzTracker,
    GermanTaxVersionPipeline,
)


def main():
    print("=" * 70)
    print("🇩🇪 GERMAN TAX LAW VERSIONING DEMO")
    print("=" * 70)

    # --- 1. Version-Aware Retrieval ---
    print("\n📚 1. Same query, different tax years:\n")

    retriever = VersionAwareRetriever()
    retriever.load_versioned_rules()

    query = "Verlustuntergang bei Anteilsübertragung über 25 Prozent"

    for year in [2010, 2016, 2022]:
        print(f"\n  ▶ VZ {year}: '{query}'")
        results = retriever.query(query, tax_year=year, top_k=1)
        if "warning" not in results[0]:
            r = results[0]
            print(f"    Version: {r['version_id']}")
            print(f"    Gültig: {r['gueltig_ab']} bis {r['gueltig_bis']}")
            print(f"    Änderungsgesetz: {r['aenderungsgesetz']}")
            print(f"    Text: {r['text'][:120]}...")
            if "warnungen" in r:
                for w in r["warnungen"]:
                    print(f"    ⚠️  {w}")
            if "hinweis_neuere_fassung" in r:
                print(f"    {r['hinweis_neuere_fassung']}")

    # --- 2. Hebesatz History ---
    print("\n\n📊 2. Hebesatz-Entwicklung:\n")

    tracker = HebesatzTracker()
    tracker.load_sample_data()

    for city in ["München", "Monheim am Rhein", "Leverkusen"]:
        change = tracker.find_changes(city, 2010, 2024)
        print(f"  {city}: {change['hebesatz_from']}% → {change['hebesatz_to']}% "
              f"({'geändert' if change['changed'] else 'unverändert'})")

    # --- 3. Version Comparison ---
    print("\n\n🔄 3. Versionsvergleich §8c KStG (VZ 2010 vs. VZ 2022):\n")

    comp = retriever.compare_versions("KStG_8c", 2010, 2022)
    if comp.get("has_changed"):
        print(f"  ✅ Änderung festgestellt!")
        for key in [f"vz_2010", f"vz_2022"]:
            val = comp[key]
            if isinstance(val, dict):
                print(f"  [{key}] {val['version_id']}: {val['text'][:100]}...")

    # --- 4. Full Pipeline ---
    print("\n\n🔗 4. Complete Version-Aware Pipeline:\n")

    pipeline = GermanTaxVersionPipeline()
    pipeline.initialize()

    analysis = pipeline.analyze_for_year(
        question="Zinsschranke und Abzugsfähigkeit von Zinsaufwendungen",
        tax_year=2023,
        gemeinde="München",
    )

    print(f"  VZ: {analysis['veranlagungszeitraum']}")
    print(f"  Gemeinde: {analysis.get('gemeinde')} (Hebesatz: {analysis.get('hebesatz')}%)")
    print(f"  Gesetzlicher Steuersatz: {analysis['gesetzlicher_steuersatz']['combined_pct']}")

    if "rechtsgrundlagen" in analysis and "warning" not in analysis["rechtsgrundlagen"][0]:
        r = analysis["rechtsgrundlagen"][0]
        print(f"  Rechtsgrundlage: {r['paragraph']} ({r['version_id']})")
        print(f"  Änderungsgesetz: {r['aenderungsgesetz']}")

    if "vorschau_naechstes_jahr" in analysis:
        print(f"  {analysis['vorschau_naechstes_jahr']['hinweis']}")
        print(f"  Neue Fassung durch: {analysis['vorschau_naechstes_jahr']['neue_fassung']}")

    # --- 5. Change History ---
    print("\n\n📜 5. Änderungshistorie Zinsschranke:\n")

    history = retriever.get_change_history("Zinsschranke")
    for entry in history:
        print(f"  {entry['gueltig_ab']} | {entry['aenderungstyp']} | {entry['aenderungsgesetz']}")

    print("\n✅ Versioning demo complete.")


if __name__ == "__main__":
    main()
