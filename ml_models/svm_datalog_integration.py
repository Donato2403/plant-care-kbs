# svm_datalog_integration.py
"""
Plant Care KBS - Integrazione SVM-Datalog per Diagnosi Ibrida
Combinazione apprendimento automatico e ragionamento simbolico
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from plant_svm_classifier import PlantSVMClassifier, PlantDiseaseDataGenerator
from datalog.plant_diagnostics_engine import PlantDiagnosticEngine
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import json

class HybridDiagnosticSystem:
    """
    Sistema diagnostico ibrido che combina:
    1. Ragionamento simbolico (Datalog/ASP) 
    2. Apprendimento statistico (SVM)

    Seguendo l'architettura neuro-simbolica per KBS
    """

    def __init__(self):
        self.svm_classifier = PlantSVMClassifier()
        self.datalog_engine = PlantDiagnosticEngine()
        self.is_trained = False

    def train_svm_component(self, n_samples: int = 2000):
        """
        Addestra la componente SVM del sistema ibrido

        Args:
            n_samples: Numero di campioni per il training
        """
        print("🤖 Training componente SVM...")

        # Genera dataset basato su conoscenza simbolica
        generator = PlantDiseaseDataGenerator()
        df = generator.generate_synthetic_dataset(n_samples)

        # Addestra SVM
        metrics = self.svm_classifier.train(df)
        self.is_trained = True

        print(f"✅ SVM addestrato - Accuracy: {metrics['test_accuracy']:.3f}")
        return metrics

    def symbolic_diagnosis(self, plant: str, symptoms: List[str], season: str = "summer") -> Dict:
        """
        Esegue diagnosi simbolica con Datalog/ASP

        Args:
            plant: Nome della pianta
            symptoms: Lista di sintomi osservati
            season: Stagione corrente

        Returns:
            Risultati diagnosi simbolica
        """
        print(f"🧠 Diagnosi simbolica per {plant}...")

        # Setup contesto Datalog
        self.datalog_engine.clear_observations()
        self.datalog_engine.set_current_season(season)

        # Aggiungi osservazioni sintomi
        for symptom in symptoms:
            self.datalog_engine.add_observation(plant, symptom)

        # Esegui ragionamento
        result = self.datalog_engine.run_diagnosis()

        return result

    def statistical_diagnosis(self, features: Dict) -> Dict:
        """
        Esegue diagnosi statistica con SVM

        Args:
            features: Features numeriche per SVM

        Returns:
            Risultati diagnosi statistica
        """
        if not self.is_trained:
            raise ValueError("SVM non addestrato! Eseguire train_svm_component() prima.")

        print("📊 Diagnosi statistica con SVM...")

        return self.svm_classifier.predict(features)

    def convert_symptoms_to_features(self, plant: str, symptoms: List[str], 
                                   season: str = "summer", 
                                   organ: str = "leaf") -> Dict:
        """
        Converte sintomi simbolici in features numeriche per SVM

        Args:
            plant: Nome pianta
            symptoms: Lista sintomi
            season: Stagione
            organ: Organo colpito

        Returns:
            Dizionario features per SVM
        """
        # Inizializza tutte le features a 0
        features = {}

        # One-hot encoding piante
        plants = ['rose', 'tomato', 'basil', 'olive']
        for p in plants:
            features[f'plant_{p}'] = 1 if plant == p else 0

        # One-hot encoding stagioni
        seasons = ['spring', 'summer', 'autumn', 'winter']
        for s in seasons:
            features[f'season_{s}'] = 1 if season == s else 0

        # One-hot encoding organi
        organs = ['leaf', 'stem', 'root', 'flower', 'fruit']
        for o in organs:
            features[f'affected_organ_{o}'] = 1 if organ == o else 0

        # Features numeriche
        features['num_symptoms'] = len(symptoms)
        features['environmental_humidity'] = 0.7  # Default
        features['environmental_temperature'] = 25.0  # Default

        # Severity scores per sintomi
        symptom_types = [
            'white_spots', 'black_spots', 'brown_spots', 'yellow_spots',
            'leaf_yellowing', 'wilting', 'deformation', 'stunted_growth',
            'leaf_drop', 'root_rot'
        ]

        for symptom_type in symptom_types:
            if symptom_type in symptoms:
                features[f'severity_{symptom_type}'] = 0.8  # Alto per sintomi presenti
            else:
                features[f'severity_{symptom_type}'] = 0.1  # Basso per sintomi assenti

        return features

    def hybrid_diagnosis(self, plant: str, symptoms: List[str], 
                        season: str = "summer", organ: str = "leaf") -> Dict:
        """
        Diagnosi ibrida combinando approcci simbolico e statistico

        Args:
            plant: Nome pianta
            symptoms: Lista sintomi
            season: Stagione
            organ: Organo colpito

        Returns:
            Risultati diagnosi ibrida con aggregazione
        """
        print(f"🔬 DIAGNOSI IBRIDA - Plant Care KBS")
        print(f"   Pianta: {plant}")
        print(f"   Sintomi: {symptoms}")
        print(f"   Stagione: {season}")
        print("=" * 50)

        results = {
            'plant': plant,
            'symptoms': symptoms,
            'season': season,
            'organ': organ
        }

        try:
            # 1. Diagnosi simbolica (Datalog/ASP)
            symbolic_result = self.symbolic_diagnosis(plant, symptoms, season)
            results['symbolic'] = symbolic_result

            # 2. Diagnosi statistica (SVM) 
            features = self.convert_symptoms_to_features(plant, symptoms, season, organ)
            statistical_result = self.statistical_diagnosis(features)
            results['statistical'] = statistical_result

            # 3. Aggregazione risultati
            aggregated = self.aggregate_diagnoses(symbolic_result, statistical_result)
            results['hybrid_diagnosis'] = aggregated

            print("✅ Diagnosi ibrida completata")

        except Exception as e:
            print(f"❌ Errore diagnosi: {e}")
            results['error'] = str(e)

        return results

    def aggregate_diagnoses(self, symbolic: Dict, statistical: Dict) -> Dict:
        """
        Aggrega risultati diagnostici simbolici e statistici

        Args:
            symbolic: Risultati Datalog/ASP
            statistical: Risultati SVM

        Returns:
            Diagnosi aggregata con confidenza combinata
        """
        aggregated = {
            'method': 'hybrid_symbolic_statistical',
            'diseases': [],
            'confidence_scores': {},
            'agreement': False,
            'recommendation': None
        }

        # Estrai malattie da diagnosi simbolica
        symbolic_diseases = []
        if symbolic.get('success', False) and symbolic.get('diagnoses'):
            symbolic_diseases = [diag['disease'] for diag in symbolic['diagnoses']]

        # Malattia da diagnosi statistica
        statistical_disease = statistical.get('predicted_disease', None)
        statistical_confidence = statistical.get('confidence', 0.0)

        # Verifica accordo
        agreement = statistical_disease in symbolic_diseases
        aggregated['agreement'] = agreement

        if agreement:
            # Accordo: alta confidenza
            aggregated['diseases'] = [statistical_disease]
            aggregated['confidence_scores'][statistical_disease] = min(0.95, statistical_confidence + 0.2)
            aggregated['recommendation'] = f"Diagnosi confermata da entrambi i metodi: {statistical_disease}"

        else:
            # Disaccordo: valuta entrambe
            all_diseases = set(symbolic_diseases + [statistical_disease])

            for disease in all_diseases:
                if disease == statistical_disease:
                    # Peso maggiore a SVM se non confermato da regole
                    aggregated['confidence_scores'][disease] = statistical_confidence * 0.7
                elif disease in symbolic_diseases:
                    # Peso alle regole simboliche
                    aggregated['confidence_scores'][disease] = 0.6

            # Malattia con confidenza più alta
            best_disease = max(aggregated['confidence_scores'].keys(), 
                             key=lambda x: aggregated['confidence_scores'][x])
            aggregated['diseases'] = [best_disease]
            aggregated['recommendation'] = f"Possibile {best_disease} (metodi divergenti - richiede verifica)"

        return aggregated

    def explain_diagnosis(self, diagnosis_result: Dict) -> str:
        """
        Genera spiegazione human-readable della diagnosi

        Args:
            diagnosis_result: Risultato diagnosi ibrida

        Returns:
            Spiegazione testuale
        """
        explanation = []
        explanation.append(f"🌱 **DIAGNOSI PLANT CARE KBS**")
        explanation.append(f"Pianta: {diagnosis_result['plant']}")
        explanation.append(f"Sintomi osservati: {', '.join(diagnosis_result['symptoms'])}")
        explanation.append("")

        if 'hybrid_diagnosis' in diagnosis_result:
            hybrid = diagnosis_result['hybrid_diagnosis']

            explanation.append("🔬 **RISULTATO IBRIDO:**")
            if hybrid['diseases']:
                disease = hybrid['diseases'][0]
                confidence = hybrid['confidence_scores'].get(disease, 0.0)
                explanation.append(f"Malattia identificata: **{disease}**")
                explanation.append(f"Confidenza: {confidence:.1%}")
                explanation.append(f"Accordo metodi: {'✅ Sì' if hybrid['agreement'] else '⚠️ No'}")
                explanation.append("")
                explanation.append(f"📋 Raccomandazione: {hybrid['recommendation']}")
            else:
                explanation.append("❓ Nessuna malattia identificata con certezza")

        if diagnosis_result.get('symbolic', {}).get('treatments'):
            explanation.append("")
            explanation.append("💊 **TRATTAMENTI SUGGERITI:**")
            for treatment in diagnosis_result['symbolic']['treatments']:
                explanation.append(f"• {treatment['treatment']} per {treatment['disease']}")

        return "\n".join(explanation)

def demo_hybrid_system():
    """
    Demo completa del sistema diagnostico ibrido
    """
    print("🔬 PLANT CARE KBS - SISTEMA IBRIDO SVM+DATALOG")
    print("=" * 65)

    # Inizializza sistema
    system = HybridDiagnosticSystem()

    # Addestra componente SVM
    print("\n📚 Phase 1: Training SVM Component")
    system.train_svm_component(n_samples=1500)

    # Test scenari diagnostici
    print("\n🧪 Phase 2: Test Scenari Diagnostici")

    # Scenario 1: Rosa con oidio (accordo atteso)
    print("\n" + "="*50)
    print("📋 SCENARIO 1: Rosa con sintomi oidio")
    result1 = system.hybrid_diagnosis(
        plant="rose",
        symptoms=["white_spots", "leaf_yellowing"],
        season="summer"
    )

    explanation1 = system.explain_diagnosis(result1)
    print("\n" + explanation1)

    # Scenario 2: Pomodoro con virus (test robustezza)
    print("\n" + "="*50)
    print("📋 SCENARIO 2: Pomodoro con deformazioni")
    result2 = system.hybrid_diagnosis(
        plant="tomato", 
        symptoms=["deformation", "yellow_spots", "stunted_growth"],
        season="summer"
    )

    explanation2 = system.explain_diagnosis(result2)
    print("\n" + explanation2)

    # Scenario 3: Caso ambiguo (test gestione incertezza)
    print("\n" + "="*50)
    print("📋 SCENARIO 3: Basilico con sintomi misti")
    result3 = system.hybrid_diagnosis(
        plant="basil",
        symptoms=["leaf_yellowing", "wilting"],
        season="spring"
    )

    explanation3 = system.explain_diagnosis(result3)
    print("\n" + explanation3)

    # Statistiche finali
    print("\n" + "="*65)
    print("📊 STATISTICHE SISTEMA IBRIDO")

    scenarios = [result1, result2, result3]
    agreements = sum(1 for r in scenarios 
                    if r.get('hybrid_diagnosis', {}).get('agreement', False))

    print(f"• Scenari testati: {len(scenarios)}")
    print(f"• Accordi simbolico-statistico: {agreements}/{len(scenarios)}")
    print(f"• Metodi integrati: Datalog/ASP + SVM")
    print(f"• Approccio: Neuro-simbolico ibrido")

    print("\n✅ Demo sistema ibrido completata!")

if __name__ == "__main__":
    demo_hybrid_system()
