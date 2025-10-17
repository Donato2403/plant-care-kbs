# plant_bayesian_network.py
"""
Plant Care KBS - Rete Bayesiana per Ragionamento Probabilistico
Implementazione Bayesian Network per gestione incertezza diagnostica
Seguendo i principi del corso di Ingegneria della Conoscenza - Modelli Probabilistici
"""

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.sampling import BayesianModelSampling
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json
import warnings
warnings.filterwarnings('ignore')

class PlantBayesianNetwork:
    """
    Rete Bayesiana per diagnosi probabilistica malattie piante
    Implementa modello causale malattie → sintomi con gestione incertezza
    """

    def __init__(self):
        """Inizializza la Rete Bayesiana con struttura DAG"""
        self.model = None
        self.inference_engine = None
        self._build_network()

    def _build_network(self):
        """
        Costruisce la struttura DAG della Bayesian Network

        Struttura causale:
        - Season/Environment → Disease
        - Disease → Symptoms  
        - Plant susceptibility influences Disease
        """
        print("🔬 Costruzione Rete Bayesiana Plant Care...")

        # Definizione struttura DAG (Directed Acyclic Graph)
        # Seguendo principi belief networks del corso
        edges = [
            # Fattori ambientali → Malattie
            ('Season', 'Disease'),
            ('Humidity', 'Disease'),
            ('Temperature', 'Disease'),
            ('PlantType', 'Disease'),

            # Malattie → Sintomi (modello causale)
            ('Disease', 'WhiteSpots'),
            ('Disease', 'BlackSpots'),
            ('Disease', 'Yellowing'),
            ('Disease', 'Wilting'),
            ('Disease', 'Deformation'),
            ('Disease', 'StuntedGrowth'),

            # Sintomi correlati (dipendenze secondarie)
            ('Yellowing', 'LeafDrop'),
            ('Wilting', 'LeafDrop')
        ]

        # Crea il modello
        self.model = DiscreteBayesianNetwork(edges)

        # Definisci CPD (Conditional Probability Distributions)
        self._define_cpds()

        # Aggiungi CPDs al modello
        self.model.add_cpds(*self.cpds)

        # Verifica correttezza del modello
        assert self.model.check_model(), "❌ Modello Bayesiano non valido!"

        print("✅ Rete Bayesiana costruita con successo")
        print(f"   • Nodi: {len(self.model.nodes())}")
        print(f"   • Archi: {len(self.model.edges())}")

        # Inizializza motore inferenza
        self.inference_engine = VariableElimination(self.model)

    def _define_cpds(self):
        """
        Definisce le Conditional Probability Distributions (CPD)
        Probabilità basate su conoscenza esperta del dominio botanico
        """

        # CPD per variabili ambientali (nodi radice, senza genitori)

        # Season: {spring, summer, autumn, winter}
        cpd_season = TabularCPD(
            variable='Season',
            variable_card=4,
            values=[[0.25], [0.25], [0.25], [0.25]],  # Uniforme
            state_names={'Season': ['spring', 'summer', 'autumn', 'winter']}
        )

        # Humidity: {low, medium, high}
        cpd_humidity = TabularCPD(
            variable='Humidity',
            variable_card=3,
            values=[[0.2], [0.5], [0.3]],  # Più probabile media
            state_names={'Humidity': ['low', 'medium', 'high']}
        )

        # Temperature: {cold, mild, warm, hot}
        cpd_temperature = TabularCPD(
            variable='Temperature',
            variable_card=4,
            values=[[0.15], [0.35], [0.35], [0.15]],  # Più probabile temperata
            state_names={'Temperature': ['cold', 'mild', 'warm', 'hot']}
        )

        # PlantType: {rose, tomato, basil, olive}
        cpd_plant = TabularCPD(
            variable='PlantType',
            variable_card=4,
            values=[[0.25], [0.25], [0.25], [0.25]],  # Uniforme
            state_names={'PlantType': ['rose', 'tomato', 'basil', 'olive']}
        )

        # CPD per Disease (dipende da Season, Humidity, Temperature, PlantType)
        # Disease: {powdery_mildew, black_spot, mosaic_virus, bacterial_canker, 
        #           iron_chlorosis, healthy}

        # Matrice 6 x (4*3*4*4) = 6 x 192
        # Semplificazione: aggreghiamo per casi tipici
        # Implementazione completa richiederebbe 6 x 192 = 1152 probabilità

        # Usiamo approccio semplificato: probabilità marginali condizionate
        disease_probs = self._compute_disease_cpd()

        cpd_disease = TabularCPD(
            variable='Disease',
            variable_card=6,
            values=disease_probs,
            evidence=['Season', 'Humidity', 'Temperature', 'PlantType'],
            evidence_card=[4, 3, 4, 4],
            state_names={
                'Disease': ['powdery_mildew', 'black_spot', 'mosaic_virus', 
                           'bacterial_canker', 'iron_chlorosis', 'healthy'],
                'Season': ['spring', 'summer', 'autumn', 'winter'],
                'Humidity': ['low', 'medium', 'high'],
                'Temperature': ['cold', 'mild', 'warm', 'hot'],
                'PlantType': ['rose', 'tomato', 'basil', 'olive']
            }
        )

        # CPD per sintomi (dipendono dalla malattia)
        # Noisy-OR model: sintomo appare se malattia lo causa (con rumore)

        # WhiteSpots (causato principalmente da powdery_mildew)
        cpd_white_spots = TabularCPD(
            variable='WhiteSpots',
            variable_card=2,
            values=[
                [0.05, 0.95, 0.1, 0.1, 0.1, 0.02],  # absent
                [0.95, 0.05, 0.9, 0.9, 0.9, 0.98]   # present
            ],
            evidence=['Disease'],
            evidence_card=[6],
            state_names={
                'WhiteSpots': ['absent', 'present'],
                'Disease': ['powdery_mildew', 'black_spot', 'mosaic_virus',
                           'bacterial_canker', 'iron_chlorosis', 'healthy']
            }
        )

        # BlackSpots (causato da black_spot)
        cpd_black_spots = TabularCPD(
            variable='BlackSpots',
            variable_card=2,
            values=[
                [0.1, 0.05, 0.15, 0.1, 0.15, 0.02],   # absent
                [0.9, 0.95, 0.85, 0.9, 0.85, 0.98]    # present
            ],
            evidence=['Disease'],
            evidence_card=[6],
            state_names={
                'BlackSpots': ['absent', 'present'],
                'Disease': ['powdery_mildew', 'black_spot', 'mosaic_virus',
                           'bacterial_canker', 'iron_chlorosis', 'healthy']
            }
        )

        # Yellowing (comune a molte malattie)
        cpd_yellowing = TabularCPD(
            variable='Yellowing',
            variable_card=2,
            values=[
                [0.3, 0.2, 0.2, 0.3, 0.05, 0.01],     # absent
                [0.7, 0.8, 0.8, 0.7, 0.95, 0.99]      # present
            ],
            evidence=['Disease'],
            evidence_card=[6],
            state_names={
                'Yellowing': ['absent', 'present'],
                'Disease': ['powdery_mildew', 'black_spot', 'mosaic_virus',
                           'bacterial_canker', 'iron_chlorosis', 'healthy']
            }
        )

        # Wilting (batterico e stress)
        cpd_wilting = TabularCPD(
            variable='Wilting',
            variable_card=2,
            values=[
                [0.2, 0.3, 0.4, 0.05, 0.3, 0.01],     # absent
                [0.8, 0.7, 0.6, 0.95, 0.7, 0.99]      # present
            ],
            evidence=['Disease'],
            evidence_card=[6],
            state_names={
                'Wilting': ['absent', 'present'],
                'Disease': ['powdery_mildew', 'black_spot', 'mosaic_virus',
                           'bacterial_canker', 'iron_chlorosis', 'healthy']
            }
        )

        # Deformation (virale)
        cpd_deformation = TabularCPD(
            variable='Deformation',
            variable_card=2,
            values=[
                [0.2, 0.15, 0.05, 0.15, 0.2, 0.01],   # absent
                [0.8, 0.85, 0.95, 0.85, 0.8, 0.99]    # present
            ],
            evidence=['Disease'],
            evidence_card=[6],
            state_names={
                'Deformation': ['absent', 'present'],
                'Disease': ['powdery_mildew', 'black_spot', 'mosaic_virus',
                           'bacterial_canker', 'iron_chlorosis', 'healthy']
            }
        )

        # StuntedGrowth (comune)
        cpd_stunted = TabularCPD(
            variable='StuntedGrowth',
            variable_card=2,
            values=[
                [0.3, 0.3, 0.2, 0.2, 0.1, 0.02],      # absent
                [0.7, 0.7, 0.8, 0.8, 0.9, 0.98]       # present
            ],
            evidence=['Disease'],
            evidence_card=[6],
            state_names={
                'StuntedGrowth': ['absent', 'present'],
                'Disease': ['powdery_mildew', 'black_spot', 'mosaic_virus',
                           'bacterial_canker', 'iron_chlorosis', 'healthy']
            }
        )

        # LeafDrop (dipende da Yellowing e Wilting)
        cpd_leaf_drop = TabularCPD(
            variable='LeafDrop',
            variable_card=2,
            values=[
                # Yellowing: absent, absent, present, present
                # Wilting:   absent, present, absent, present
                [0.95, 0.6, 0.7, 0.2],                # absent
                [0.05, 0.4, 0.3, 0.8]                 # present
            ],
            evidence=['Yellowing', 'Wilting'],
            evidence_card=[2, 2],
            state_names={
                'LeafDrop': ['absent', 'present'],
                'Yellowing': ['absent', 'present'],
                'Wilting': ['absent', 'present']
            }
        )

        # Salva tutte le CPDs
        self.cpds = [
            cpd_season, cpd_humidity, cpd_temperature, cpd_plant,
            cpd_disease,
            cpd_white_spots, cpd_black_spots, cpd_yellowing,
            cpd_wilting, cpd_deformation, cpd_stunted, cpd_leaf_drop
        ]

    def _compute_disease_cpd(self) -> np.ndarray:
        """
        Computa CPD per Disease data la complessità delle evidenze
        Usa modello semplificato con logica esperta

        Returns:
            Array 6 x 192 con probabilità normalizzate
        """
        # 6 disease states x 192 parental configurations
        n_diseases = 6
        n_configs = 4 * 3 * 4 * 4  # Season x Humidity x Temp x Plant

        # Inizializza con probabilità base uniforme
        probs = np.ones((n_diseases, n_configs)) / n_diseases

        # Modifica probabilità per configurazioni tipiche
        # Logica semplificata: modifichiamo solo scenari principali

        for config_idx in range(n_configs):
            # Decodifica configurazione
            season_idx = config_idx // (3 * 4 * 4)
            humidity_idx = (config_idx // (4 * 4)) % 3
            temp_idx = (config_idx // 4) % 4
            plant_idx = config_idx % 4

            # Estate + alta umidità → più probabile powdery_mildew
            if season_idx == 1 and humidity_idx == 2:  # summer, high humidity
                probs[0, config_idx] = 0.4  # powdery_mildew
                probs[5, config_idx] = 0.2  # healthy ridotto

            # Primavera + rosa → black_spot
            if season_idx == 0 and plant_idx == 0:  # spring, rose
                probs[1, config_idx] = 0.35  # black_spot

            # Estate + pomodoro → mosaic_virus
            if season_idx == 1 and plant_idx == 1:  # summer, tomato
                probs[2, config_idx] = 0.3  # mosaic_virus

            # Umidità alta → bacterial_canker
            if humidity_idx == 2:
                probs[3, config_idx] = 0.25  # bacterial_canker

            # Normalizza
            probs[:, config_idx] /= probs[:, config_idx].sum()

        return probs

    def query(self, variables: List[str], evidence: Dict[str, str] = None) -> Dict:
        """
        Esegue inferenza probabilistica (Variable Elimination)

        Args:
            variables: Lista variabili di query
            evidence: Dizionario evidenze osservate

        Returns:
            Distribuzione a posteriori delle variabili
        """
        if evidence is None:
            evidence = {}

        print(f"🔮 Inferenza Bayesiana:")
        print(f"   Query: {variables}")
        print(f"   Evidenze: {evidence}")

        try:
            # Exact inference con Variable Elimination
            result = self.inference_engine.query(
                variables=variables,
                evidence=evidence,
                show_progress=False
            )

            return self._parse_inference_result(result, variables[0])

        except Exception as e:
            print(f"❌ Errore inferenza: {e}")
            return {'error': str(e)}

    def _parse_inference_result(self, result, variable: str) -> Dict:
        """Parse risultato inferenza in formato leggibile"""
        states = result.state_names[variable]
        values = result.values

        distribution = {}
        for i, state in enumerate(states):
            distribution[state] = float(values[i])

        # Ordina per probabilità decrescente
        sorted_dist = dict(sorted(distribution.items(), 
                                 key=lambda x: x[1], reverse=True))

        return {
            'variable': variable,
            'distribution': sorted_dist,
            'most_likely': max(sorted_dist.items(), key=lambda x: x[1])[0],
            'confidence': max(sorted_dist.values())
        }

    def map_query(self, evidence: Dict[str, str]) -> Dict:
        """
        Maximum A Posteriori (MAP) query: configurazione più probabile

        Args:
            evidence: Evidenze osservate

        Returns:
            Configurazione più probabile per variabili non osservate
        """
        print(f"🎯 MAP Query con evidenze: {evidence}")

        try:
            result = self.inference_engine.map_query(
                variables=['Disease'],
                evidence=evidence,
                show_progress=False
            )

            return {'map_assignment': result}

        except Exception as e:
            print(f"❌ Errore MAP: {e}")
            return {'error': str(e)}

    def sample_network(self, n_samples: int = 100) -> pd.DataFrame:
        """
        Genera campioni dalla distribuzione congiunta (Forward Sampling)

        Args:
            n_samples: Numero di campioni da generare

        Returns:
            DataFrame con campioni generati
        """
        sampler = BayesianModelSampling(self.model)
        samples = sampler.forward_sample(size=n_samples)

        print(f"🎲 Generati {n_samples} campioni dalla BN")
        return samples

    def get_markov_blanket(self, node: str) -> List[str]:
        """
        Restituisce il Markov Blanket di un nodo
        (genitori, figli, e altri genitori dei figli)

        Args:
            node: Nome del nodo

        Returns:
            Lista nodi nel Markov blanket
        """
        mb = []

        # Genitori
        parents = list(self.model.get_parents(node))
        mb.extend(parents)

        # Figli
        children = list(self.model.get_children(node))
        mb.extend(children)

        # Altri genitori dei figli (co-parents)
        for child in children:
            co_parents = self.model.get_parents(child)
            mb.extend([p for p in co_parents if p != node and p not in mb])

        return mb

    def explain_inference(self, disease_result: Dict, evidence: Dict) -> str:
        """
        Genera spiegazione human-readable dell'inferenza

        Args:
            disease_result: Risultato query su Disease
            evidence: Evidenze fornite

        Returns:
            Spiegazione testuale
        """
        explanation = []
        explanation.append("🔬 **DIAGNOSI BAYESIANA PLANT CARE KBS**\n")

        explanation.append("**Evidenze osservate:**")
        for var, value in evidence.items():
            explanation.append(f"  • {var}: {value}")

        explanation.append("\n**Distribuzione probabilità malattie:**")
        for disease, prob in disease_result['distribution'].items():
            explanation.append(f"  • {disease}: {prob:.1%}")

        explanation.append(f"\n**Diagnosi più probabile:** {disease_result['most_likely']}")
        explanation.append(f"**Confidenza:** {disease_result['confidence']:.1%}")

        # Interpretazione
        if disease_result['confidence'] > 0.7:
            explanation.append("\n✅ **Alta confidenza** - Diagnosi affidabile")
        elif disease_result['confidence'] > 0.4:
            explanation.append("\n⚠️ **Media confidenza** - Raccogliere più evidenze")
        else:
            explanation.append("\n❓ **Bassa confidenza** - Diagnosi incerta, serve esame approfondito")

        return "\n".join(explanation)

def demo_bayesian_network():
    """
    Demo completa della Rete Bayesiana per diagnosi piante
    """
    print("🌱 PLANT CARE KBS - DEMO BAYESIAN NETWORK")
    print("=" * 65)

    # Inizializza rete
    bn = PlantBayesianNetwork()

    print("\n📊 Struttura della Rete:")
    print(f"   Nodi: {list(bn.model.nodes())}")
    print(f"   Genitori di Disease: {list(bn.model.get_parents('Disease'))}")
    print(f"   Figli di Disease: {list(bn.model.get_children('Disease'))}")

    # Scenario 1: Inferenza con evidenze ambientali
    print("\n" + "="*65)
    print("📋 SCENARIO 1: Diagnosi con evidenze ambientali")
    evidence1 = {
        'Season': 'summer',
        'Humidity': 'high',
        'PlantType': 'rose'
    }

    result1 = bn.query(variables=['Disease'], evidence=evidence1)
    explanation1 = bn.explain_inference(result1, evidence1)
    print("\n" + explanation1)

    # Scenario 2: Inferenza con sintomi osservati
    print("\n" + "="*65)
    print("📋 SCENARIO 2: Diagnosi con sintomi osservati")
    evidence2 = {
        'WhiteSpots': 'present',
        'Yellowing': 'present',
        'StuntedGrowth': 'present'
    }

    result2 = bn.query(variables=['Disease'], evidence=evidence2)
    explanation2 = bn.explain_inference(result2, evidence2)
    print("\n" + explanation2)

    # Scenario 3: Inferenza combinata (ambiente + sintomi)
    print("\n" + "="*65)
    print("📋 SCENARIO 3: Diagnosi combinata ambiente+sintomi")
    evidence3 = {
        'Season': 'summer',
        'Humidity': 'high',
        'PlantType': 'tomato',
        'Deformation': 'present',
        'Yellowing': 'present'
    }

    result3 = bn.query(variables=['Disease'], evidence=evidence3)
    explanation3 = bn.explain_inference(result3, evidence3)
    print("\n" + explanation3)

    # Scenario 4: Query sintomo dato malattia (reasoning inverso)
    print("\n" + "="*65)
    print("📋 SCENARIO 4: Predizione sintomi data malattia")
    evidence4 = {'Disease': 'powdery_mildew'}

    result4_white = bn.query(variables=['WhiteSpots'], evidence=evidence4)
    result4_yellow = bn.query(variables=['Yellowing'], evidence=evidence4)

    print(f"\nData malattia: {evidence4['Disease']}")
    print(f"  • Probabilità WhiteSpots: {result4_white['distribution']}")
    print(f"  • Probabilità Yellowing: {result4_yellow['distribution']}")

    # Test Markov Blanket
    print("\n" + "="*65)
    print("📋 MARKOV BLANKET")
    mb_disease = bn.get_markov_blanket('Disease')
    print(f"Markov Blanket di 'Disease': {mb_disease}")

    # Sampling
    print("\n" + "="*65)
    print("📋 FORWARD SAMPLING")
    samples = bn.sample_network(n_samples=10)
    print("\nPrimi 5 campioni:")
    print(samples.head())

    print("\n" + "="*65)
    print("✅ Demo Bayesian Network completata con successo!")

    # Statistiche finali
    print("\n📊 STATISTICHE RETE BAYESIANA")
    print(f"  • Nodi totali: {len(bn.model.nodes())}")
    print(f"  • Archi: {len(bn.model.edges())}")
    print(f"  • CPDs definite: {len(bn.cpds)}")
    print(f"  • Metodo inferenza: Variable Elimination")
    print(f"  • Gestione incertezza: Probabilità Bayesiana")

if __name__ == "__main__":
    demo_bayesian_network()
