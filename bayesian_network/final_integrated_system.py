# final_integrated_system.py
"""
Plant Care KBS - Sistema Integrato Finale Tri-Paradigma
Integrazione completa: Ontologia OWL + Datalog/ASP + SVM + Bayesian Network
Sistema diagnostico ibrido simbolico-statistico-probabilistico
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from ml_models.plant_svm_classifier import PlantSVMClassifier, PlantDiseaseDataGenerator
from datalog.plant_diagnostics_engine import PlantDiagnosticEngine
from plant_bayesian_network import PlantBayesianNetwork
from ml_models.svm_datalog_integration import HybridDiagnosticSystem
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import json

class FinalIntegratedKBS:
    """
    Sistema Knowledge-Based completo con integrazione tri-paradigma:
    1. Ragionamento Simbolico (OWL + Datalog/ASP)
    2. Apprendimento Statistico (SVM)  
    3. Ragionamento Probabilistico (Bayesian Network)

    Seguendo l'architettura completa dell'Ingegneria della Conoscenza
    """

    def __init__(self):
        self.datalog_engine = PlantDiagnosticEngine()
        self.svm_classifier = PlantSVMClassifier()
        self.bayesian_network = PlantBayesianNetwork()
        self.is_svm_trained = False

    def train_ml_component(self, n_samples: int = 1500):
        """
        Addestra la componente SVM del sistema

        Args:
            n_samples: Numero di campioni per il training
        """
        print("🤖 Training componente SVM del sistema integrato...")

        generator = PlantDiseaseDataGenerator()
        df = generator.generate_synthetic_dataset(n_samples)

        metrics = self.svm_classifier.train(df)
        self.is_svm_trained = True

        print(f"✅ SVM addestrato - Accuracy: {metrics['test_accuracy']:.3f}")
        return metrics

    def symbolic_diagnosis(self, plant: str, symptoms: List[str], 
                          season: str = "summer") -> Dict:
        """Diagnosi simbolica con Datalog/ASP"""
        print(f"🧠 Diagnosi Simbolica (Datalog/ASP)...")

        self.datalog_engine.clear_observations()
        self.datalog_engine.set_current_season(season)

        for symptom in symptoms:
            self.datalog_engine.add_observation(plant, symptom)

        return self.datalog_engine.run_diagnosis()

    def statistical_diagnosis(self, plant: str, symptoms: List[str],
                             season: str = "summer", organ: str = "leaf") -> Dict:
        """Diagnosi statistica con SVM"""
        if not self.is_svm_trained:
            raise ValueError("SVM non addestrato! Eseguire train_ml_component() prima.")

        print(f"📊 Diagnosi Statistica (SVM)...")

        # Conversione sintomi → features
        features = self._convert_symptoms_to_svm_features(
            plant, symptoms, season, organ
        )

        return self.svm_classifier.predict(features)

    def probabilistic_diagnosis(self, plant: str, symptoms: List[str],
                               season: str = "summer", 
                               humidity: str = "medium",
                               temperature: str = "warm") -> Dict:
        """Diagnosi probabilistica con Bayesian Network"""
        print(f"🔬 Diagnosi Probabilistica (Bayesian Network)...")

        # Mappa sintomi a variabili BN
        evidence = {
            'Season': season,
            'Humidity': humidity,
            'Temperature': temperature,
            'PlantType': plant
        }

        # Aggiungi evidenze sintomi
        symptom_mapping = {
            'white_spots': 'WhiteSpots',
            'black_spots': 'BlackSpots',
            'leaf_yellowing': 'Yellowing',
            'wilting': 'Wilting',
            'deformation': 'Deformation',
            'stunted_growth': 'StuntedGrowth'
        }

        for symptom in symptoms:
            if symptom in symptom_mapping:
                evidence[symptom_mapping[symptom]] = 'present'

        # Query sulla malattia
        result = self.bayesian_network.query(
            variables=['Disease'],
            evidence=evidence
        )

        return result

    def integrated_diagnosis(self, plant: str, symptoms: List[str],
                            season: str = "summer",
                            humidity: str = "medium",
                            temperature: str = "warm",
                            organ: str = "leaf") -> Dict:
        """
        Diagnosi integrata combinando tutti e tre i paradigmi

        Args:
            plant: Tipo pianta
            symptoms: Lista sintomi osservati
            season: Stagione corrente
            humidity: Livello umidità
            temperature: Livello temperatura
            organ: Organo colpito

        Returns:
            Risultati diagnosi integrata con fusion decisionale
        """
        print(f"🔬 DIAGNOSI INTEGRATA TRI-PARADIGMA")
        print(f"   Pianta: {plant}")
        print(f"   Sintomi: {symptoms}")
        print(f"   Contesto: {season}, humidity={humidity}, temp={temperature}")
        print("=" * 60)

        results = {
            'plant': plant,
            'symptoms': symptoms,
            'context': {
                'season': season,
                'humidity': humidity,
                'temperature': temperature,
                'organ': organ
            }
        }

        try:
            # 1. Diagnosi Simbolica (Datalog/ASP)
            symbolic_result = self.symbolic_diagnosis(plant, symptoms, season)
            results['symbolic'] = symbolic_result

            # 2. Diagnosi Statistica (SVM)
            statistical_result = self.statistical_diagnosis(
                plant, symptoms, season, organ
            )
            results['statistical'] = statistical_result

            # 3. Diagnosi Probabilistica (Bayesian Network)
            probabilistic_result = self.probabilistic_diagnosis(
                plant, symptoms, season, humidity, temperature
            )
            results['probabilistic'] = probabilistic_result

            # 4. Fusion decisionale tri-paradigma
            integrated = self.fuse_diagnoses(
                symbolic_result, statistical_result, probabilistic_result
            )
            results['integrated_diagnosis'] = integrated

            print("✅ Diagnosi integrata completata con successo")

        except Exception as e:
            print(f"❌ Errore diagnosi integrata: {e}")
            results['error'] = str(e)

        return results

    def fuse_diagnoses(self, symbolic: Dict, statistical: Dict, 
                      probabilistic: Dict) -> Dict:
        """
        Fusion decisionale combinando i tre approcci

        Strategia:
        - Bayesian Network fornisce probabilità base
        - SVM aggiunge confidenza statistica  
        - Datalog fornisce validazione logica

        Args:
            symbolic: Risultati Datalog/ASP
            statistical: Risultati SVM
            probabilistic: Risultati Bayesian Network

        Returns:
            Diagnosi fusa con confidenza combinata
        """
        fused = {
            'method': 'tri_paradigm_fusion',
            'diseases': [],
            'confidence_scores': {},
            'paradigm_agreement': {},
            'recommendation': None
        }

        # Estrai malattie da ogni paradigma
        symbolic_diseases = []
        if symbolic.get('success', False) and symbolic.get('diagnoses'):
            symbolic_diseases = [diag['disease'] for diag in symbolic['diagnoses']]

        statistical_disease = statistical.get('predicted_disease', None)
        statistical_confidence = statistical.get('confidence', 0.0)

        probabilistic_dist = probabilistic.get('distribution', {})
        probabilistic_disease = probabilistic.get('most_likely', None)
        probabilistic_confidence = probabilistic.get('confidence', 0.0)

        # Raccogli tutte le malattie candidate
        all_diseases = set()
        if symbolic_diseases:
            all_diseases.update(symbolic_diseases)
        if statistical_disease:
            all_diseases.add(statistical_disease)
        if probabilistic_disease:
            all_diseases.add(probabilistic_disease)

        # Calcola score fusion per ogni malattia
        for disease in all_diseases:
            # Peso per ogni paradigma
            weight_symbolic = 0.25
            weight_statistical = 0.35
            weight_probabilistic = 0.40

            # Score simbolico
            score_symbolic = 0.6 if disease in symbolic_diseases else 0.0

            # Score statistico
            score_statistical = statistical_confidence if disease == statistical_disease else 0.0

            # Score probabilistico
            score_probabilistic = probabilistic_dist.get(disease, 0.0)

            # Fusion score pesato
            fusion_score = (
                weight_symbolic * score_symbolic +
                weight_statistical * score_statistical +
                weight_probabilistic * score_probabilistic
            )

            fused['confidence_scores'][disease] = fusion_score

            # Traccia accordo paradigmi
            fused['paradigm_agreement'][disease] = {
                'symbolic': disease in symbolic_diseases,
                'statistical': disease == statistical_disease,
                'probabilistic': disease == probabilistic_disease
            }

        # Malattia con score più alto
        if fused['confidence_scores']:
            best_disease = max(fused['confidence_scores'].items(), 
                             key=lambda x: x[1])[0]
            best_score = fused['confidence_scores'][best_disease]

            fused['diseases'] = [best_disease]
            fused['final_confidence'] = best_score

            # Conta paradigmi in accordo
            agreement = fused['paradigm_agreement'][best_disease]
            n_agree = sum(agreement.values())

            # Raccomandazione basata su accordo
            if n_agree == 3:
                fused['recommendation'] = (
                    f"✅ Diagnosi confermata da tutti e 3 i paradigmi: {best_disease} "
                    f"(confidenza: {best_score:.1%})"
                )
            elif n_agree == 2:
                fused['recommendation'] = (
                    f"✓ Diagnosi probabile {best_disease} confermata da 2/3 paradigmi "
                    f"(confidenza: {best_score:.1%})"
                )
            else:
                fused['recommendation'] = (
                    f"⚠️ Diagnosi incerta {best_disease} - solo 1/3 paradigmi concorde "
                    f"(confidenza: {best_score:.1%}) - Raccogliere più evidenze"
                )
        else:
            fused['recommendation'] = "❓ Nessuna diagnosi identificata con certezza"

        return fused

    def explain_integrated_diagnosis(self, diagnosis_result: Dict) -> str:
        """
        Genera spiegazione dettagliata della diagnosi integrata

        Args:
            diagnosis_result: Risultato diagnosi integrata

        Returns:
            Spiegazione testuale completa
        """
        explanation = []
        explanation.append("🌱 **DIAGNOSI INTEGRATA PLANT CARE KBS**")
        explanation.append("=" * 60)
        explanation.append(f"Pianta: {diagnosis_result['plant']}")
        explanation.append(f"Sintomi osservati: {', '.join(diagnosis_result['symptoms'])}")
        explanation.append(f"Contesto: {diagnosis_result['context']}")
        explanation.append("")

        # Risultati per paradigma
        explanation.append("🔬 **RISULTATI PER PARADIGMA:**")
        explanation.append("")

        # Simbolico
        if 'symbolic' in diagnosis_result:
            explanation.append("1️⃣ **Ragionamento Simbolico (Datalog/ASP):**")
            symbolic = diagnosis_result['symbolic']
            if symbolic.get('success'):
                for diag in symbolic.get('diagnoses', []):
                    explanation.append(f"   • {diag['disease']} (confidenza: {diag.get('confidence_level', 'N/A')})")
            else:
                explanation.append("   • Nessuna diagnosi")
            explanation.append("")

        # Statistico  
        if 'statistical' in diagnosis_result:
            explanation.append("2️⃣ **Apprendimento Statistico (SVM):**")
            statistical = diagnosis_result['statistical']
            explanation.append(f"   • {statistical.get('predicted_disease', 'N/A')}")
            explanation.append(f"   • Confidenza: {statistical.get('confidence', 0):.1%}")
            explanation.append("")

        # Probabilistico
        if 'probabilistic' in diagnosis_result:
            explanation.append("3️⃣ **Ragionamento Probabilistico (Bayesian Network):**")
            prob = diagnosis_result['probabilistic']
            explanation.append(f"   • Malattia più probabile: {prob.get('most_likely', 'N/A')}")
            explanation.append(f"   • Confidenza: {prob.get('confidence', 0):.1%}")
            explanation.append("   • Distribuzione completa:")
            for disease, p in list(prob.get('distribution', {}).items())[:3]:
                explanation.append(f"      - {disease}: {p:.1%}")
            explanation.append("")

        # Diagnosi integrata
        if 'integrated_diagnosis' in diagnosis_result:
            explanation.append("🎯 **DIAGNOSI FINALE INTEGRATA:**")
            integrated = diagnosis_result['integrated_diagnosis']

            if integrated['diseases']:
                disease = integrated['diseases'][0]
                confidence = integrated['final_confidence']

                explanation.append(f"   • Malattia identificata: **{disease}**")
                explanation.append(f"   • Confidenza fusion: {confidence:.1%}")
                explanation.append("")

                # Accordo paradigmi
                agreement = integrated['paradigm_agreement'][disease]
                explanation.append("   • Accordo paradigmi:")
                explanation.append(f"      - Simbolico: {'✓' if agreement['symbolic'] else '✗'}")
                explanation.append(f"      - Statistico: {'✓' if agreement['statistical'] else '✗'}")
                explanation.append(f"      - Probabilistico: {'✓' if agreement['probabilistic'] else '✗'}")
                explanation.append("")

                # Raccomandazione
                explanation.append(f"📋 {integrated['recommendation']}")

        return "\n".join(explanation)

    def _convert_symptoms_to_svm_features(self, plant: str, symptoms: List[str],
                                         season: str, organ: str) -> Dict:
        """Conversione sintomi → features per SVM"""
        features = {}

        # One-hot encoding piante
        for p in ['rose', 'tomato', 'basil', 'olive']:
            features[f'plant_{p}'] = 1 if plant == p else 0

        # One-hot encoding stagioni
        for s in ['spring', 'summer', 'autumn', 'winter']:
            features[f'season_{s}'] = 1 if season == s else 0

        # One-hot encoding organi
        for o in ['leaf', 'stem', 'root', 'flower', 'fruit']:
            features[f'affected_organ_{o}'] = 1 if organ == o else 0

        # Features numeriche
        features['num_symptoms'] = len(symptoms)
        features['environmental_humidity'] = 0.7
        features['environmental_temperature'] = 25.0

        # Severity scores
        symptom_types = [
            'white_spots', 'black_spots', 'brown_spots', 'yellow_spots',
            'leaf_yellowing', 'wilting', 'deformation', 'stunted_growth',
            'leaf_drop', 'root_rot'
        ]

        for symptom_type in symptom_types:
            if symptom_type in symptoms:
                features[f'severity_{symptom_type}'] = 0.8
            else:
                features[f'severity_{symptom_type}'] = 0.1

        return features

def demo_final_integrated_system():
    """
    Demo completa del sistema integrato tri-paradigma
    """
    print("🌱 PLANT CARE KBS - SISTEMA INTEGRATO TRI-PARADIGMA")
    print("=" * 70)

    # Inizializza sistema
    system = FinalIntegratedKBS()

    # Training componente SVM
    print("\n📚 Phase 1: Training componente Machine Learning")
    system.train_ml_component(n_samples=1000)

    # Test scenari diagnostici integrati
    print("\n🧪 Phase 2: Test Scenari Diagnostici Integrati")

    # Scenario 1: Rosa con oidio (accordo atteso tra tutti paradigmi)
    print("\n" + "="*70)
    print("📋 SCENARIO 1: Rosa con sintomi oidio estivo")
    result1 = system.integrated_diagnosis(
        plant="rose",
        symptoms=["white_spots", "leaf_yellowing", "stunted_growth"],
        season="summer",
        humidity="high",
        temperature="warm"
    )

    explanation1 = system.explain_integrated_diagnosis(result1)
    print("\n" + explanation1)

    # Scenario 2: Pomodoro con virus
    print("\n" + "="*70)
    print("📋 SCENARIO 2: Pomodoro con deformazioni virali")
    result2 = system.integrated_diagnosis(
        plant="tomato",
        symptoms=["deformation", "leaf_yellowing", "stunted_growth"],
        season="summer",
        humidity="medium",
        temperature="hot"
    )

    explanation2 = system.explain_integrated_diagnosis(result2)
    print("\n" + explanation2)

    # Scenario 3: Caso ambiguo (test robustezza)
    print("\n" + "="*70)
    print("📋 SCENARIO 3: Basilico con sintomi misti ambigui")
    result3 = system.integrated_diagnosis(
        plant="basil",
        symptoms=["leaf_yellowing", "wilting"],
        season="spring",
        humidity="low",
        temperature="mild"
    )

    explanation3 = system.explain_integrated_diagnosis(result3)
    print("\n" + explanation3)

    # Statistiche finali
    print("\n" + "="*70)
    print("📊 STATISTICHE SISTEMA INTEGRATO FINALE")

    scenarios = [result1, result2, result3]
    full_agreements = sum(
        1 for r in scenarios 
        if r.get('integrated_diagnosis', {}).get('paradigm_agreement', {})
        and sum(list(r['integrated_diagnosis']['paradigm_agreement'].values())[0].values()) == 3
    )

    print(f"• Scenari testati: {len(scenarios)}")
    print(f"• Accordo completo (3/3 paradigmi): {full_agreements}/{len(scenarios)}")
    print(f"• Paradigmi integrati:")
    print(f"  1. Simbolico (OWL + Datalog/ASP)")
    print(f"  2. Statistico (SVM)")
    print(f"  3. Probabilistico (Bayesian Network)")
    print(f"• Metodo fusion: Weighted confidence fusion")
    print(f"• Architettura: Tri-paradigma ibrida")

    print("\n✅ Demo sistema integrato completata con successo!")
    print("🎉 PLANT CARE KBS: Sistema completo operativo!")

if __name__ == "__main__":
    demo_final_integrated_system()
