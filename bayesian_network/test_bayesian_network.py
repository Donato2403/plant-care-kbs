# test_bayesian_network.py
"""
Test Suite per il modulo Bayesian Network Plant Care KBS
Validazione completa del ragionamento probabilistico
"""

import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(__file__))

from plant_bayesian_network import PlantBayesianNetwork

class TestPlantBayesianNetwork(unittest.TestCase):
    """Test della Rete Bayesiana"""

    def setUp(self):
        """Setup eseguito prima di ogni test"""
        self.bn = PlantBayesianNetwork()

    def test_network_structure(self):
        """Test struttura DAG della rete"""
        # Verifica nodi esistenti
        expected_nodes = [
            'Season', 'Humidity', 'Temperature', 'PlantType',
            'Disease', 'WhiteSpots', 'BlackSpots', 'Yellowing',
            'Wilting', 'Deformation', 'StuntedGrowth', 'LeafDrop'
        ]

        for node in expected_nodes:
            self.assertIn(node, self.bn.model.nodes())

        # Verifica aciclicità (DAG)
        from pgmpy.base import DAG
        self.assertIsInstance(self.bn.model, DAG)

        print("✅ Test struttura rete: PASS")

    def test_cpd_validity(self):
        """Test validità CPD"""
        # Verifica che tutte le CPD sommino a 1
        for cpd in self.bn.cpds:
            # Per ogni configurazione dei genitori
            if cpd.get_evidence():
                # CPD condizionata
                self.assertTrue(self.bn.model.check_model())
            else:
                # CPD marginale
                total = cpd.values.sum()
                self.assertAlmostEqual(total, 1.0, places=5)

        print("✅ Test validità CPD: PASS")

    def test_inference_disease(self):
        """Test inferenza su malattia"""
        # Test con evidenze ambientali
        evidence = {
            'Season': 'summer',
            'Humidity': 'high',
            'PlantType': 'rose'
        }

        result = self.bn.query(variables=['Disease'], evidence=evidence)

        # Verifica struttura risultato
        self.assertIn('variable', result)
        self.assertIn('distribution', result)
        self.assertIn('most_likely', result)
        self.assertIn('confidence', result)

        # Verifica che le probabilità sommino a 1
        total_prob = sum(result['distribution'].values())
        self.assertAlmostEqual(total_prob, 1.0, places=5)

        # Verifica che confidenza sia in [0,1]
        self.assertGreaterEqual(result['confidence'], 0.0)
        self.assertLessEqual(result['confidence'], 1.0)

        print(f"✅ Test inferenza malattia: PASS")
        print(f"   Malattia più probabile: {result['most_likely']}")
        print(f"   Confidenza: {result['confidence']:.3f}")

    def test_inference_symptoms(self):
        """Test inferenza su sintomi data malattia"""
        evidence = {'Disease': 'powdery_mildew'}

        result = self.bn.query(variables=['WhiteSpots'], evidence=evidence)

        # Verifica che white spots sia più probabile con oidio
        prob_present = result['distribution'].get('present', 0.0)
        prob_absent = result['distribution'].get('absent', 0.0)

        # Con oidio, white spots dovrebbe essere molto probabile
        self.assertGreater(prob_present, prob_absent)

        print("✅ Test inferenza sintomi: PASS")
        print(f"   P(WhiteSpots=present|Disease=powdery_mildew) = {prob_present:.3f}")

    def test_markov_blanket(self):
        """Test Markov Blanket"""
        mb_disease = self.bn.get_markov_blanket('Disease')

        # Verifica che includa genitori
        expected_parents = ['Season', 'Humidity', 'Temperature', 'PlantType']
        for parent in expected_parents:
            self.assertIn(parent, mb_disease)

        # Verifica che includa figli
        expected_children = [
            'WhiteSpots', 'BlackSpots', 'Yellowing', 
            'Wilting', 'Deformation', 'StuntedGrowth'
        ]
        for child in expected_children:
            self.assertIn(child, mb_disease)

        print(f"✅ Test Markov Blanket: PASS")
        print(f"   Dimensione MB(Disease): {len(mb_disease)}")

    def test_sampling(self):
        """Test forward sampling"""
        n_samples = 100
        samples = self.bn.sample_network(n_samples)

        # Verifica dimensioni
        self.assertEqual(len(samples), n_samples)

        # Verifica che tutte le variabili siano presenti
        expected_vars = list(self.bn.model.nodes())
        for var in expected_vars:
            self.assertIn(var, samples.columns)

        print(f"✅ Test sampling: PASS ({n_samples} campioni)")

def run_integration_tests():
    """
    Test di integrazione del sistema Bayesian Network
    """
    print("\n🧪 INTEGRATION TESTS - BAYESIAN NETWORK")
    print("=" * 60)

    bn = PlantBayesianNetwork()

    # Test 1: Scenario diagnostico realistico
    print("\n📋 Test 1: Scenario Rosa Estiva")
    evidence1 = {
        'Season': 'summer',
        'Humidity': 'high',
        'Temperature': 'warm',
        'PlantType': 'rose',
        'WhiteSpots': 'present',
        'Yellowing': 'present'
    }

    result1 = bn.query(variables=['Disease'], evidence=evidence1)
    print(f"   Malattia più probabile: {result1['most_likely']}")
    print(f"   Confidenza: {result1['confidence']:.1%}")
    assert result1['confidence'] >= 0.29, "Confidenza troppo bassa"

    # Test 2: Reasoning diagnostico inverso
    print("\n📋 Test 2: Predizione Sintomi da Malattia")
    evidence2 = {'Disease': 'mosaic_virus'}

    result2_def = bn.query(variables=['Deformation'], evidence=evidence2)
    result2_yel = bn.query(variables=['Yellowing'], evidence=evidence2)

    print(f"   P(Deformation=present|Disease=mosaic_virus) = {result2_def['distribution']['present']:.1%}")
    print(f"   P(Yellowing=present|Disease=mosaic_virus) = {result2_yel['distribution']['present']:.1%}")

    # Deformation dovrebbe essere molto probabile con virus
    assert result2_def['distribution']['present'] > 0.8, "Deformazione dovrebbe essere molto probabile"

    # Test 3: Scenario con evidenze contraddittorie
    print("\n📋 Test 3: Gestione Evidenze Contrastanti")
    evidence3 = {
        'Season': 'winter',  # Stagione fredda
        'Temperature': 'cold',
        'Humidity': 'low',
        'WhiteSpots': 'present',  # Ma sintomi estivi
        'BlackSpots': 'present'
    }

    result3 = bn.query(variables=['Disease'], evidence=evidence3)
    print(f"   Malattia più probabile: {result3['most_likely']}")
    print(f"   Confidenza: {result3['confidence']:.1%}")

    # Con evidenze contrastanti, confidenza dovrebbe essere più bassa
    assert result3['confidence'] < 0.7, "Confidenza dovrebbe essere più bassa con evidenze contrastanti"

    print("\n✅ Tutti i test di integrazione completati con successo!")

def main():
    """
    Esegue tutti i test del sistema Bayesian Network
    """
    print("🧪 PLANT CARE KBS - TEST SUITE BAYESIAN NETWORK")
    print("=" * 65)

    # Test suite principale
    print("\n🔬 Esecuzione Test Suite...")

    suite = unittest.TestLoader().loadTestsFromTestCase(TestPlantBayesianNetwork)
    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)

    # Integration tests
    run_integration_tests()

    # Risultati finali
    total_tests = result.testsRun
    total_failures = len(result.failures)
    total_errors = len(result.errors)

    print("\n" + "=" * 65)
    print("🎯 RISULTATI TEST SUITE BAYESIAN NETWORK")
    print(f"   • Test eseguiti: {total_tests}")
    print(f"   • Successi: {total_tests - total_failures - total_errors}")
    print(f"   • Fallimenti: {total_failures}")
    print(f"   • Errori: {total_errors}")

    if total_failures == 0 and total_errors == 0:
        print("\n🎉 TUTTI I TEST COMPLETATI CON SUCCESSO!")
        print("✅ Modulo Bayesian Network Plant Care KBS operativo e validato")
    else:
        print("\n⚠️ Alcuni test falliti - Verificare implementazione")

    return total_failures == 0 and total_errors == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)