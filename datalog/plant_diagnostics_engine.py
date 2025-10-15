# plant_diagnostics_engine.py
"""
Plant Care KBS - Motore Diagnostico Datalog/ASP
Integrazione con clingo per ragionamento automatico su malattie delle piante
Implementazione seguendo i principi del corso di Ingegneria della Conoscenza
"""

import subprocess
import tempfile
import os
import json
from typing import List, Dict, Tuple, Optional
import sys

class PlantDiagnosticEngine:
    """
    Motore di inferenza diagnostica basato su Answer Set Programming (ASP)
    Utilizza clingo per il ragionamento automatico sulle regole diagnostiche
    """

    def __init__(self, asp_file_path: str = "datalog/plant_diagnostics.lp"):
        """
        Inizializza il motore diagnostico

        Args:
            asp_file_path: Path al file con le regole ASP
        """
        self.asp_file_path = asp_file_path
        self.facts = []  # Fatti dinamici aggiunti dall'utente
        self.diagnoses = []  # Risultati delle diagnosi

    def add_observation(self, plant: str, symptom: str):
        """
        Aggiunge un'osservazione sintomatica

        Args:
            plant: Nome della pianta osservata
            symptom: Sintomo osservato
        """
        fact = f"observes_symptom({plant}, {symptom})."
        if fact not in self.facts:
            self.facts.append(fact)
            print(f"✅ Aggiunta osservazione: {plant} mostra {symptom}")

    def set_current_season(self, season: str):
        """
        Imposta la stagione corrente per il contesto diagnostico

        Args:
            season: Stagione (spring, summer, autumn, winter)
        """
        # Rimuovi stagioni precedenti
        self.facts = [f for f in self.facts if not f.startswith("current_season")]

        fact = f"current_season({season})."
        self.facts.append(fact)
        print(f"🕐 Stagione impostata: {season}")

    def add_environmental_factor(self, factor: str, level: str):
        """
        Aggiunge fattori ambientali per diagnosi più accurate

        Args:
            factor: Fattore ambientale (humidity, temperature, etc.)
            level: Livello (high, medium, low)
        """
        fact = f"environmental_factor({factor}, {level})."
        if fact not in self.facts:
            self.facts.append(fact)
            print(f"🌡️ Fattore ambientale: {factor} = {level}")

    def run_diagnosis(self, show_all_models: bool = False) -> Dict:
        """
        Esegue la diagnosi utilizzando clingo

        Args:
            show_all_models: Se True, mostra tutti i modelli possibili

        Returns:
            Dizionario con risultati della diagnosi
        """
        try:
            # Crea file temporaneo con fatti aggiuntivi
            with tempfile.NamedTemporaryFile(mode='w', suffix='.lp', delete=False) as temp_file:
                # Scrivi i fatti dinamici
                temp_file.write("\n".join(self.facts))
                temp_file_path = temp_file.name

            # Comando clingo
            num_models = 0 if show_all_models else 1
            cmd = [
                'clingo', 
                self.asp_file_path, 
                temp_file_path,
                f'-n {num_models}',
                '--quiet=1',
                '--project'
            ]

            # Esegui clingo
            result = subprocess.run(
                ' '.join(cmd), 
                shell=True, 
                capture_output=True, 
                text=True,
                timeout=30
            )

            # Cleanup
            os.unlink(temp_file_path)

            if result.returncode != 0:
                return {
                    'success': False, 
                    'error': f"Clingo error: {result.stderr}",
                    'suggestions': self._get_error_suggestions(result.stderr)
                }

            # Parsing dei risultati
            return self._parse_clingo_output(result.stdout)

        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': "Timeout: La diagnosi ha richiesto troppo tempo",
                'suggestions': ["Ridurre il numero di osservazioni", "Verificare la coerenza dei sintomi"]
            }
        except FileNotFoundError:
            return {
                'success': False,
                'error': "Clingo non trovato. Installare clingo/potassco",
                'suggestions': [
                    "Installare clingo: conda install -c potassco clingo",
                    "O scaricare da: https://potassco.org/"
                ]
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Errore inaspettato: {str(e)}",
                'suggestions': ["Verificare la sintassi delle regole ASP"]
            }

    def _parse_clingo_output(self, output: str) -> Dict:
        """
        Parsing dell'output di clingo per estrarre diagnosi

        Args:
            output: Output testuale di clingo

        Returns:
            Struttura dati con diagnosi parsate
        """
        results = {
            'success': True,
            'diagnoses': [],
            'treatments': [],
            'preventions': [],
            'certainty_levels': [],
            'raw_atoms': []
        }

        # Estrai atomi dalle answer sets
        lines = output.strip().split('\n')
        current_answer_set = []

        for line in lines:
            if line.startswith('Answer:'):
                if current_answer_set:
                    results['raw_atoms'].extend(current_answer_set)
                current_answer_set = []
            elif line and not line.startswith('SATISFIABLE') and not line.startswith('Models'):
                atoms = line.split()
                current_answer_set.extend(atoms)

        # Aggiungi l'ultimo answer set
        if current_answer_set:
            results['raw_atoms'].extend(current_answer_set)

        # Parse degli atomi specifici
        for atom in results['raw_atoms']:
            if atom.startswith('has_disease('):
                diagnosis = self._parse_atom(atom, 'has_disease', 2)
                if diagnosis:
                    results['diagnoses'].append({
                        'plant': diagnosis[0],
                        'disease': diagnosis[1]
                    })

            elif atom.startswith('recommend_treatment('):
                treatment = self._parse_atom(atom, 'recommend_treatment', 3)
                if treatment:
                    results['treatments'].append({
                        'plant': treatment[0],
                        'disease': treatment[1],
                        'treatment': treatment[2]
                    })

            elif atom.startswith('prevention_advice('):
                prevention = self._parse_atom(atom, 'prevention_advice', 3)
                if prevention:
                    results['preventions'].append({
                        'plant': prevention[0],
                        'disease': prevention[1],
                        'advice': prevention[2]
                    })

            elif atom.startswith('certainty_level('):
                certainty = self._parse_atom(atom, 'certainty_level', 3)
                if certainty:
                    results['certainty_levels'].append({
                        'plant': certainty[0],
                        'disease': certainty[1],
                        'level': certainty[2]
                    })

        return results

    def _parse_atom(self, atom: str, predicate: str, arity: int) -> Optional[List[str]]:
        """
        Parse di un atomo ASP per estrarre i parametri

        Args:
            atom: Atomo da parsare (es. "has_disease(rose,powdery_mildew)")
            predicate: Nome del predicato
            arity: Numero di argomenti

        Returns:
            Lista degli argomenti o None se parsing fallisce
        """
        try:
            if not atom.startswith(f"{predicate}(") or not atom.endswith(")"):
                return None

            # Estrai contenuto tra parentesi
            content = atom[len(predicate)+1:-1]

            # Split per virgola (semplificato, non gestisce parentesi annidate)
            args = [arg.strip() for arg in content.split(',')]

            if len(args) == arity:
                return args
            return None

        except Exception:
            return None

    def _get_error_suggestions(self, error_text: str) -> List[str]:
        """
        Genera suggerimenti basati sugli errori di clingo

        Args:
            error_text: Testo dell'errore

        Returns:
            Lista di suggerimenti per risolvere l'errore
        """
        suggestions = []

        if "syntax error" in error_text.lower():
            suggestions.append("Verificare la sintassi delle regole Datalog")
            suggestions.append("Controllare parentesi e punti nelle clausole")

        if "unsafe" in error_text.lower():
            suggestions.append("Variabili non sicure: ogni variabile deve apparire in un atomo positivo")

        if "inconsistent" in error_text.lower():
            suggestions.append("Base di conoscenza inconsistente: verificare vincoli contraddittori")

        if not suggestions:
            suggestions.append("Consultare documentazione clingo per dettagli specifici")

        return suggestions

    def get_diagnosis_summary(self, diagnosis_result: Dict) -> str:
        """
        Genera un riassunto human-readable della diagnosi

        Args:
            diagnosis_result: Risultato del metodo run_diagnosis()

        Returns:
            Stringa con riassunto diagnostico
        """
        if not diagnosis_result['success']:
            return f"❌ Diagnosi fallita: {diagnosis_result['error']}"

        summary = []
        summary.append("🔬 **RISULTATI DIAGNOSI PLANT CARE KBS**\n")

        # Diagnosi principali
        if diagnosis_result['diagnoses']:
            summary.append("**Malattie identificate:**")
            for diag in diagnosis_result['diagnoses']:
                summary.append(f"• {diag['plant']} → {diag['disease']}")
        else:
            summary.append("• Nessuna malattia identificata con certezza")

        # Livelli di certezza
        if diagnosis_result['certainty_levels']:
            summary.append("\n**Livelli di certezza:**")
            for cert in diagnosis_result['certainty_levels']:
                summary.append(f"• {cert['plant']} - {cert['disease']}: {cert['level']}")

        # Trattamenti raccomandati
        if diagnosis_result['treatments']:
            summary.append("\n**Trattamenti raccomandati:**")
            for treat in diagnosis_result['treatments']:
                summary.append(f"• {treat['plant']}: {treat['treatment']} per {treat['disease']}")

        # Prevenzioni
        if diagnosis_result['preventions']:
            summary.append("\n**Consigli di prevenzione:**")
            for prev in diagnosis_result['preventions']:
                summary.append(f"• {prev['plant']}: {prev['advice']}")

        return "\n".join(summary)

    def clear_observations(self):
        """Pulisce tutte le osservazioni correnti"""
        self.facts = []
        print("🧹 Osservazioni cancellate")

    def save_case_study(self, filename: str, diagnosis_result: Dict):
        """
        Salva un caso di studio per future referenze

        Args:
            filename: Nome del file di output
            diagnosis_result: Risultati della diagnosi
        """
        case_data = {
            'facts': self.facts,
            'diagnosis': diagnosis_result,
            'timestamp': __import__('datetime').datetime.now().isoformat()
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(case_data, f, indent=2, ensure_ascii=False)

        print(f"💾 Caso di studio salvato: {filename}")

def demo_diagnosis():
    """
    Funzione di demo per testare il sistema diagnostico
    """
    print("🌱 DEMO PLANT CARE KBS - MOTORE DIAGNOSTICO DATALOG")
    print("=" * 60)

    # Inizializza motore
    engine = PlantDiagnosticEngine()

    # Scenario 1: Rosa con oidio in estate
    print("\n📋 **SCENARIO 1: Rosa con sintomi estivi**")
    engine.set_current_season("summer")
    engine.add_environmental_factor("humidity", "high")
    engine.add_observation("my_rose", "white_spots")
    engine.add_observation("my_rose", "leaf_yellowing")

    # Esegui diagnosi
    result = engine.run_diagnosis()
    print(engine.get_diagnosis_summary(result))

    # Scenario 2: Pomodoro con virus
    print("\n" + "=" * 60)
    print("\n📋 **SCENARIO 2: Pomodoro con deformazioni**")
    engine.clear_observations()
    engine.set_current_season("summer")
    engine.add_observation("my_tomato", "deformation")
    engine.add_observation("my_tomato", "yellow_spots")
    engine.add_observation("my_tomato", "stunted_growth")

    result2 = engine.run_diagnosis()
    print(engine.get_diagnosis_summary(result2))

    # Salva casi di studio
    engine.save_case_study("case_study_rose.json", result)
    engine.save_case_study("case_study_tomato.json", result2)

    print("\n✅ Demo completata con successo!")

if __name__ == "__main__":
    demo_diagnosis()
