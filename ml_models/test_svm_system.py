# test_svm_system.py
"""
Test Suite per il sistema SVM Plant Care KBS
Validazione completa del modulo di apprendimento automatico
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(__file__))

from plant_svm_classifier import PlantSVMClassifier, PlantDiseaseDataGenerator
from svm_datalog_integration import HybridDiagnosticSystem

class TestPlantDiseaseDataGenerator(unittest.TestCase):
    """Test del generatore di dataset sintetici"""

    def setUp(self):
        self.generator = PlantDiseaseDataGenerator()

    def test_dataset_generation(self):
        """Test generazione dataset"""
        df = self.generator.generate_synthetic_dataset(100)

        # Verifica dimensioni
        self.assertEqual(len(df), 100)

        # Verifica colonne obbligatorie
        required_cols = ['plant', 'disease', 'season', 'num_symptoms']
        for col in required_cols:
            self.assertIn(col, df.columns)

        # Verifica valori validi
        valid_plants = set(self.generator.plants)
        self.assertTrue(df['plant'].isin(valid_plants).all())

        valid_diseases = set(self.generator.diseases)
        self.assertTrue(df['disease'].isin(valid_diseases).all())

        print("✅ Test generazione dataset: PASS")

    def test_symptom_disease_mapping(self):
        """Test mappature sintomi-malattie"""
        # Verifica che ogni malattia abbia sintomi associati
        for disease in self.generator.diseases:
            symptoms = self.generator.symptom_disease_map.get(disease, [])
            self.assertGreater(len(symptoms), 0, f"Malattia {disease} senza sintomi")

        print("✅ Test mappature sintomi-malattie: PASS")

class TestPlantSVMClassifier(unittest.TestCase):
    """Test del classificatore SVM"""

    def setUp(self):
        self.classifier = PlantSVMClassifier()
        self.generator = PlantDiseaseDataGenerator()

        # Dataset piccolo per test rapidi
        self.df_small = self.generator.generate_synthetic_dataset(200)

    def test_feature_preparation(self):
        """Test preparazione features"""
        X, y = self.classifier.prepare_features(self.df_small)

        # Verifica dimensioni
        self.assertEqual(X.shape[0], len(self.df_small))
        self.assertEqual(len(y), len(self.df_small))

        # Verifica che X sia numerico
        self.assertTrue(np.issubdtype(X.dtype, np.number))

        print("✅ Test preparazione features: PASS")

    def test_model_training(self):
        """Test training del modello"""
        # Training su dataset piccolo
        metrics = self.classifier.train(self.df_small, test_size=0.3)

        # Verifica che il modello sia addestrato
        self.assertIsNotNone(self.classifier.model)

        # Verifica metriche ragionevoli
        self.assertGreater(metrics['test_accuracy'], 0.3)  # Almeno 30%
        self.assertLess(metrics['test_accuracy'], 1.1)     # Al massimo 100%

        print(f"✅ Test training modello: PASS (Accuracy: {metrics['test_accuracy']:.3f})")

    def test_prediction(self):
        """Test predizioni del modello"""
        # Addestra il modello
        self.classifier.train(self.df_small, test_size=0.3)

        # Test predizione
        test_features = {
            'plant_rose': 1, 'plant_tomato': 0, 'plant_basil': 0, 'plant_olive': 0,
            'season_summer': 1, 'season_spring': 0, 'season_autumn': 0, 'season_winter': 0,
            'affected_organ_leaf': 1, 'affected_organ_stem': 0, 'affected_organ_root': 0,
            'affected_organ_flower': 0, 'affected_organ_fruit': 0,
            'num_symptoms': 2,
            'environmental_humidity': 0.7,
            'environmental_temperature': 25.0,
            'severity_white_spots': 0.8,
            'severity_leaf_yellowing': 0.6,
            'severity_stunted_growth': 0.4,
            'severity_black_spots': 0.1,
            'severity_brown_spots': 0.1,
            'severity_yellow_spots': 0.2,
            'severity_wilting': 0.1,
            'severity_deformation': 0.1,
            'severity_leaf_drop': 0.2,
            'severity_root_rot': 0.1
        }

        result = self.classifier.predict(test_features)

        # Verifica struttura risultato
        self.assertIn('predicted_disease', result)
        self.assertIn('confidence', result)
        self.assertIn('probabilities', result)

        # Verifica valori ragionevoli
        self.assertGreater(result['confidence'], 0.0)
        self.assertLessEqual(result['confidence'], 1.0)

        print(f"✅ Test predizione: PASS (Malattia: {result['predicted_disease']}, Conf: {result['confidence']:.3f})")

class TestHybridDiagnosticSystem(unittest.TestCase):
    """Test del sistema diagnostico ibrido"""

    def setUp(self):
        self.system = HybridDiagnosticSystem()

        # Addestra componente SVM
        print("Training SVM per test...")
        self.system.train_svm_component(n_samples=300)

    def test_feature_conversion(self):
        """Test conversione sintomi → features"""
        symptoms = ["white_spots", "leaf_yellowing"]
        features = self.system.convert_symptoms_to_features(
            plant="rose",
            symptoms=symptoms,
            season="summer"
        )

        # Verifica presenza features obbligatorie
        self.assertEqual(features['plant_rose'], 1)
        self.assertEqual(features['plant_tomato'], 0)
        self.assertEqual(features['season_summer'], 1)
        self.assertEqual(features['num_symptoms'], 2)

        # Verifica severity scores
        self.assertGreater(features['severity_white_spots'], 0.5)
        self.assertGreater(features['severity_leaf_yellowing'], 0.5)
        self.assertLess(features['severity_black_spots'], 0.5)

        print("✅ Test conversione features: PASS")

    def test_statistical_diagnosis(self):
        """Test diagnosi statistica"""
        features = {
            'plant_rose': 1, 'plant_tomato': 0, 'plant_basil': 0, 'plant_olive': 0,
            'season_summer': 1, 'season_spring': 0, 'season_autumn': 0, 'season_winter': 0,
            'affected_organ_leaf': 1, 'affected_organ_stem': 0, 'affected_organ_root': 0,
            'affected_organ_flower': 0, 'affected_organ_fruit': 0,
            'num_symptoms': 2,
            'environmental_humidity': 0.8,
            'environmental_temperature': 28.0,
            'severity_white_spots': 0.9,
            'severity_leaf_yellowing': 0.7,
            'severity_stunted_growth': 0.6,
            'severity_black_spots': 0.1,
            'severity_brown_spots': 0.1,
            'severity_yellow_spots': 0.2,
            'severity_wilting': 0.1,
            'severity_deformation': 0.1,
            'severity_leaf_drop': 0.2,
            'severity_root_rot': 0.1
        }

        result = self.system.statistical_diagnosis(features)

        # Verifica struttura risultato
        self.assertIn('predicted_disease', result)
        self.assertIn('confidence', result)

        print(f"✅ Test diagnosi statistica: PASS (Malattia: {result['predicted_disease']})")

    def test_hybrid_diagnosis(self):
        """Test diagnosi ibrida completa"""
        result = self.system.hybrid_diagnosis(
            plant="rose",
            symptoms=["white_spots", "leaf_yellowing"],
            season="summer"
        )

        # Verifica struttura risultato
        self.assertIn('plant', result)
        self.assertIn('symptoms', result)
        self.assertIn('statistical', result)

        # Verifica che non ci siano errori critici
        self.assertNotIn('error', result)

        print("✅ Test diagnosi ibrida: PASS")

def run_performance_benchmark():
    """
    Benchmark performance del sistema SVM
    """
    print("\n🏃 BENCHMARK PERFORMANCE SVM SYSTEM")
    print("=" * 50)

    import time

    # Setup
    generator = PlantDiseaseDataGenerator()
    classifier = PlantSVMClassifier()

    # Test 1: Generazione dataset
    print("\n📊 Test 1: Generazione Dataset")
    start_time = time.time()
    df = generator.generate_synthetic_dataset(1000)
    dataset_time = time.time() - start_time
    print(f"   Tempo generazione 1000 campioni: {dataset_time:.3f}s")

    # Test 2: Training SVM
    print("\n🤖 Test 2: Training SVM")
    start_time = time.time()
    metrics = classifier.train(df, test_size=0.2)
    training_time = time.time() - start_time
    print(f"   Tempo training: {training_time:.3f}s")
    print(f"   Accuracy: {metrics['test_accuracy']:.3f}")

    # Test 3: Predizioni multiple
    print("\n🔮 Test 3: Predizioni Multiple")
    test_features = {
        'plant_rose': 1, 'plant_tomato': 0, 'plant_basil': 0, 'plant_olive': 0,
        'season_summer': 1, 'season_spring': 0, 'season_autumn': 0, 'season_winter': 0,
        'affected_organ_leaf': 1, 'affected_organ_stem': 0, 'affected_organ_root': 0,
        'affected_organ_flower': 0, 'affected_organ_fruit': 0,
        'num_symptoms': 2,
        'environmental_humidity': 0.7,
        'environmental_temperature': 25.0,
        'severity_white_spots': 0.8,
        'severity_leaf_yellowing': 0.6,
        'severity_stunted_growth': 0.4,
        'severity_black_spots': 0.1,
        'severity_brown_spots': 0.1,
        'severity_yellow_spots': 0.2,
        'severity_wilting': 0.1,
        'severity_deformation': 0.1,
        'severity_leaf_drop': 0.2,
        'severity_root_rot': 0.1
    }

    start_time = time.time()
    predictions = []
    for _ in range(100):
        pred = classifier.predict(test_features)
        predictions.append(pred)
    prediction_time = time.time() - start_time

    print(f"   Tempo 100 predizioni: {prediction_time:.3f}s")
    print(f"   Tempo medio per predizione: {prediction_time/100*1000:.1f}ms")

    # Riepilogo benchmark
    print("\n📈 RIEPILOGO BENCHMARK")
    print(f"   • Dataset generation: {dataset_time:.3f}s per 1000 campioni")
    print(f"   • SVM training: {training_time:.3f}s")
    print(f"   • Prediction latency: {prediction_time/100*1000:.1f}ms")
    print(f"   • Model accuracy: {metrics['test_accuracy']:.1%}")

def main():
    """
    Esegue tutti i test del sistema SVM
    """
    print("🧪 PLANT CARE KBS - TEST SUITE SVM")
    print("=" * 60)

    # Test suite principale
    print("\n🔬 Esecuzione Test Suite...")

    # Test generatore dataset
    print("\n1️⃣ Test Dataset Generator")
    suite1 = unittest.TestLoader().loadTestsFromTestCase(TestPlantDiseaseDataGenerator)
    runner = unittest.TextTestRunner(verbosity=0)
    result1 = runner.run(suite1)

    # Test classificatore SVM
    print("\n2️⃣ Test SVM Classifier")
    suite2 = unittest.TestLoader().loadTestsFromTestCase(TestPlantSVMClassifier)
    result2 = runner.run(suite2)

    # Test sistema ibrido
    print("\n3️⃣ Test Sistema Ibrido")
    suite3 = unittest.TestLoader().loadTestsFromTestCase(TestHybridDiagnosticSystem)
    result3 = runner.run(suite3)

    # Benchmark performance
    run_performance_benchmark()

    # Risultati finali
    total_tests = result1.testsRun + result2.testsRun + result3.testsRun
    total_failures = len(result1.failures) + len(result2.failures) + len(result3.failures)
    total_errors = len(result1.errors) + len(result2.errors) + len(result3.errors)

    print("\n" + "=" * 60)
    print("🎯 RISULTATI TEST SUITE SVM")
    print(f"   • Test eseguiti: {total_tests}")
    print(f"   • Successi: {total_tests - total_failures - total_errors}")
    print(f"   • Fallimenti: {total_failures}")
    print(f"   • Errori: {total_errors}")

    if total_failures == 0 and total_errors == 0:
        print("\n🎉 TUTTI I TEST COMPLETATI CON SUCCESSO!")
        print("✅ Modulo SVM Plant Care KBS operativo e validato")
    else:
        print("\n⚠️ Alcuni test falliti - Verificare implementazione")

    return total_failures == 0 and total_errors == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
