"""
Demo: German Corporate Tax AI System
=====================================
End-to-end: synthetic data → model training → prediction → SHAP explanation → law retrieval
"""

import numpy as np
import pandas as pd
from german_tax_models import GermanTaxModels, compute_statutory_rate, GERMAN_TAX_FEATURES
from german_tax_law_rag import GermanTaxLawRetriever


def generate_synthetic_german_data(n=500, seed=42):
    """Generate realistic synthetic German corporate tax data."""
    rng = np.random.default_rng(seed)

    data = pd.DataFrame({
        'umsatz': rng.lognormal(16, 1.5, n),                    # Revenue
        'gewinn_vor_steuern': rng.lognormal(14, 1.5, n),        # Pre-tax profit
        'bilanzsumme': rng.lognormal(17, 1.2, n),               # Total assets
        'eigenkapitalquote': rng.beta(3, 4, n),                  # Equity ratio
        'verschuldungsgrad': rng.exponential(2, n),              # Debt/equity
        'abschreibungen': rng.lognormal(12, 1, n),              # Depreciation
        'forschung_entwicklung': rng.exponential(500000, n),     # R&D
        'personalaufwand': rng.lognormal(15, 1, n),             # Personnel
        'zinsaufwand': rng.exponential(200000, n),              # Interest
        'mieten_pachten': rng.exponential(100000, n),           # Rent
        'hebesatz': rng.choice([250, 350, 400, 410, 420, 440, 460, 470, 490], n),
        'verlustvortraege': rng.exponential(300000, n) * rng.binomial(1, 0.3, n),
        'organschaft': rng.binomial(1, 0.15, n),
        'ausschuettungen': rng.exponential(200000, n),
        'beteiligungsertraege': rng.exponential(100000, n) * rng.binomial(1, 0.25, n),
        'dauerschuldzinsen': rng.exponential(150000, n),
        'investitionsabzugsbetrag': rng.exponential(50000, n) * rng.binomial(1, 0.2, n),
        'rechtsform_gmbh': rng.binomial(1, 0.7, n),
        'rechtsform_ag': 0,  # will fill below
        'branche_code': rng.choice(range(10, 90), n),
        'bundesland_code': rng.choice(range(1, 17), n),
        'mitarbeiter_anzahl': rng.lognormal(4, 1.5, n).astype(int),
        'gruendungsjahr': rng.integers(1950, 2023, n),
    })

    data['rechtsform_ag'] = ((1 - data['rechtsform_gmbh']) * rng.binomial(1, 0.5, n)).astype(int)

    # --- Simulate effective tax rate ---
    # Base: statutory rate from Hebesatz
    statutory = np.array([compute_statutory_rate(h)['combined_rate'] for h in data['hebesatz']])

    # Deviations based on features
    etr = statutory.copy()

    # Beteiligungserträge → §8b KStG exemption lowers ETR
    etr -= 0.02 * (data['beteiligungsertraege'] > 100000).astype(float)

    # Verlustvorträge → lower ETR
    etr -= 0.03 * (data['verlustvortraege'] > 500000).astype(float)

    # Organschaft → often lowers group ETR
    etr -= 0.015 * data['organschaft']

    # High interest + Zinsschranke → ETR goes up
    etr += 0.01 * (data['zinsaufwand'] > 500000).astype(float)

    # Hinzurechnung (rent) → ETR goes up
    etr += 0.008 * (data['mieten_pachten'] > 200000).astype(float)

    # R&D → no direct German tax credit, but affects profit → slight effect
    etr -= 0.005 * (data['forschung_entwicklung'] > 1000000).astype(float)

    # IAB → lowers ETR
    etr -= 0.01 * (data['investitionsabzugsbetrag'] > 100000).astype(float)

    # Add noise
    etr += rng.normal(0, 0.01, n)
    etr = np.clip(etr, 0.10, 0.40)

    data['effective_tax_rate'] = etr

    return data


def main():
    print("=" * 60)
    print("🇩🇪 GERMAN CORPORATE TAX AI SYSTEM - DEMO")
    print("=" * 60)

    # --- 1. Generate Data ---
    print("\n📊 Generating synthetic German corporate tax data...")
    df = generate_synthetic_german_data(n=500)

    feature_cols = [c for c in df.columns if c != 'effective_tax_rate']
    X = df[feature_cols]
    y = df['effective_tax_rate']

    # Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"   Training: {len(X_train)} | Test: {len(X_test)}")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Target: effective_tax_rate (mean={y.mean():.4f}, std={y.std():.4f})")

    # --- 2. Statutory Rate Reference ---
    print("\n📐 German statutory rates (for reference):")
    for city, hs in [("München", 490), ("Berlin", 410), ("Monheim", 250)]:
        rates = compute_statutory_rate(hs)
        print(f"   {city} (Hebesatz {hs}%): {rates['combined_pct']}")

    # --- 3. Train Model ---
    print("\n🤖 Training XGBoost model...")
    tm = GermanTaxModels()
    model = tm.get_model('xgboost', n_estimators=200, max_depth=6, learning_rate=0.05)
    tm.fit('de_tax_v1', model, X_train, y_train, feature_names=feature_cols)

    # Evaluate
    from sklearn.metrics import mean_absolute_error, r2_score
    preds = tm.predict('de_tax_v1', X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"   MAE: {mae:.4f} ({mae*100:.2f} Prozentpunkte)")
    print(f"   R²:  {r2:.4f}")

    # --- 4. Prediction with Breakdown ---
    print("\n📋 Prediction breakdown (first 5 test samples):")
    breakdown = tm.predict_with_breakdown('de_tax_v1', X_test)
    sample = breakdown.head()
    for i, row in sample.iterrows():
        print(f"   Unternehmen {i}: statutory={row['statutory_rate']:.2%} | "
              f"predicted={row['predicted_effective_rate']:.2%} | "
              f"deviation={row['deviation_pct_points']:+.2f}pp")

    # --- 5. SHAP Explanation ---
    print("\n🔍 Computing SHAP explanations...")
    importance = tm.get_feature_importance('de_tax_v1', X_test, top_n=10)
    print("\n   Top 10 Features (mean |SHAP|):")
    for _, row in importance.iterrows():
        print(f"   {row['feature']:30s} → {row['mean_abs_shap']:.6f}")

    # --- 6. Tax Law Retrieval ---
    print("\n" + "=" * 60)
    print("📚 TAX LAW RETRIEVAL (RAG)")
    print("=" * 60)

    retriever = GermanTaxLawRetriever()
    retriever.load_rules()  # Uses built-in sample rules

    queries = [
        "Wie werden Beteiligungserträge bei der Körperschaftsteuer behandelt?",
        "Wann gehen Verlustvorträge bei Anteilsübertragung unter?",
        "Welche Hinzurechnungen gibt es bei der Gewerbesteuer?",
        "Was ist die Zinsschranke und wie wird sie berechnet?",
        "Voraussetzungen für eine ertragsteuerliche Organschaft",
    ]

    for q in queries:
        print(f"\n❓ Frage: {q}")
        results = retriever.query(q, top_k=2)

        if "warning" in results[0]:
            print(f"   ⚠️  {results[0]['warning']}")
        else:
            for r in results:
                print(f"   📖 {r['citation']} - {r['title']} (Relevanz: {r['relevance']:.3f})")

    # --- 7. Combined: SHAP → RAG ---
    print("\n" + "=" * 60)
    print("🔗 COMBINED: SHAP finding → automatic law lookup")
    print("=" * 60)

    top_feature = importance.iloc[0]['feature']
    print(f"\n   Top SHAP feature: '{top_feature}'")
    print(f"   → Automatic legal lookup...")

    # Map feature to legal query
    feature_to_query = {
        "hebesatz": "Wie wird der Gewerbesteuer-Hebesatz angewendet?",
        "beteiligungsertraege": "Freistellung von Beteiligungserträgen §8b KStG",
        "verlustvortraege": "Verlustabzug und Verlustuntergang bei Körperschaften",
        "zinsaufwand": "Zinsschranke und Abzugsbeschränkung für Zinsen",
        "mieten_pachten": "Gewerbesteuerliche Hinzurechnung von Mieten und Pachten",
        "organschaft": "Voraussetzungen der ertragsteuerlichen Organschaft",
        "forschung_entwicklung": "Steuerliche Förderung von Forschung und Entwicklung",
        "investitionsabzugsbetrag": "Investitionsabzugsbetrag nach §7g EStG",
    }

    query_text = feature_to_query.get(top_feature, f"Steuerliche Behandlung von {top_feature}")
    results = retriever.query(query_text, top_k=1)

    if "warning" not in results[0]:
        r = results[0]
        print(f"   📖 Relevante Vorschrift: {r['citation']} - {r['title']}")
        print(f"   📝 {r['text'][:200]}...")

    print("\n✅ Demo complete.")


if __name__ == "__main__":
    main()
