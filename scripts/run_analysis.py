"""
TrustyFile - Script d'analyse batch en ligne de commande.

Analyse tous les fichiers PDF d'un répertoire et affiche les résultats.

Usage:
    python scripts/run_analysis.py [dossier]
    python scripts/run_analysis.py samples/id_doc
    python scripts/run_analysis.py samples/legitimate

Par défaut (sans argument), analyse le dossier samples/id_doc.
"""

import sys
import os
import logging
from pathlib import Path

# Le script vit dans scripts/ : on ajoute la racine du dépôt au chemin
# d'import pour que `from src...` fonctionne quel que soit le dossier courant.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Optionnel : activer les logs pour voir ce que fait chaque module
# logging.basicConfig(level=logging.DEBUG)

from src.analyzer import TrustyFileAnalyzer


def main():
    # Lire le dossier depuis les arguments, ou utiliser la valeur par défaut
    folder = sys.argv[1] if len(sys.argv) > 1 else "samples/id_doc"

    if not os.path.isdir(folder):
        print(f"Erreur : '{folder}' n'est pas un dossier valide.")
        sys.exit(1)

    # Lister les PDF du dossier (triés alphabétiquement)
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".pdf"))

    if not files:
        print(f"Aucun fichier PDF trouvé dans '{folder}'")
        sys.exit(0)

    print(f"Analyse de {len(files)} fichier(s) dans '{folder}'...")

    # Initialiser l'analyseur sans vérification externe (plus rapide)
    analyzer = TrustyFileAnalyzer(enable_external=False)

    for f in files:
        path = os.path.join(folder, f)

        try:
            result = analyzer.analyze(path)

            print(f"\n{'='*60}")
            print(f"Fichier : {f}")
            print(f"Type    : {result.document_type}")
            print(f"Score   : {result.trust_score}/100  |  Risque : {result.risk_level}")

            # Afficher les flags de chaque module (uniquement si des flags existent)
            for mod in result.modules:
                if mod.flags:
                    print(f"  [{mod.module}] score={mod.score}  confidence={mod.confidence:.2f}")
                    for flag in mod.flags:
                        print(f"    [{flag.severity.upper()}] {flag.message}")

            # Si aucun flag : document sans alerte
            total_flags = sum(len(mod.flags) for mod in result.modules)
            if total_flags == 0:
                print("  Aucun problème détecté.")

        except Exception as e:
            print(f"\n{'='*60}")
            print(f"Fichier : {f}")
            print(f"ERREUR  : {e}")

    print(f"\n{'='*60}")
    print(f"Terminé. {len(files)} fichier(s) analysé(s).")


if __name__ == "__main__":
    main()
