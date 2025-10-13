# plant_care_ontology.py
"""
Plant Care KBS - Ontologia OWL per diagnosi malattie delle piante
Implementazione con Owlready2 seguendo i principi delle logiche descrittive
"""

from owlready2 import *
import tempfile
import os

# Creazione dell'ontologia
onto = get_ontology("http://www.plantcare.org/ontology.owl")

with onto:
    # CLASSI PRINCIPALI

    # Classe radice per organismi viventi
    class LivingOrganism(Thing):
        """Classe base per tutti gli organismi viventi"""
        pass

    # PIANTE
    class Plant(LivingOrganism):
        """Classe base per tutte le piante"""
        pass

    class Rose(Plant):
        """Rosa - pianta ornamentale con spine"""
        pass

    class Tomato(Plant):
        """Pomodoro - pianta da frutto della famiglia solanaceae"""
        pass

    class Basil(Plant):
        """Basilico - erba aromatica"""
        pass

    class Olive(Plant):
        """Olivo - albero da frutto mediterraneo"""
        pass

    # MALATTIE
    class Disease(Thing):
        """Classe base per tutte le malattie delle piante"""
        pass

    class FungalDisease(Disease):
        """Malattie causate da funghi"""
        pass

    class BacterialDisease(Disease):
        """Malattie causate da batteri"""
        pass

    class ViralDisease(Disease):
        """Malattie causate da virus"""
        pass

    class NutritionalDeficiency(Disease):
        """Carenze nutritive"""
        pass

    # Malattie fungine specifiche
    class PowderyMildew(FungalDisease):
        """Oidio - malattia fungina che causa macchie bianche"""
        pass

    class BlackSpot(FungalDisease):
        """Macchia nera - malattia fungina comune nelle rose"""
        pass

    class Rust(FungalDisease):
        """Ruggine - malattia fungina che causa macchie arancioni"""
        pass

    # SINTOMI
    class Symptom(Thing):
        """Classe base per tutti i sintomi"""
        pass

    class VisibleSymptom(Symptom):
        """Sintomi visibili ad occhio nudo"""
        pass

    class GrowthSymptom(Symptom):
        """Sintomi relativi alla crescita"""
        pass

    # Sintomi visibili specifici
    class LeafYellowing(VisibleSymptom):
        """Ingiallimento delle foglie"""
        pass

    class WhiteSpots(VisibleSymptom):
        """Macchie bianche sulle piante"""
        pass

    class BrownSpots(VisibleSymptom):
        """Macchie marroni sulle piante"""
        pass

    class Wilting(VisibleSymptom):
        """Appassimento della pianta"""
        pass

    # Sintomi di crescita specifici
    class StuntedGrowth(GrowthSymptom):
        """Crescita rallentata"""
        pass

    class LeafDrop(GrowthSymptom):
        """Caduta delle foglie"""
        pass

    class RootRot(GrowthSymptom):
        """Marciume delle radici"""
        pass

    # ORGANI VEGETALI
    class PlantOrgan(Thing):
        """Classe base per gli organi delle piante"""
        pass

    class Leaf(PlantOrgan):
        """Foglia"""
        pass

    class Stem(PlantOrgan):
        """Fusto/Stelo"""
        pass

    class Root(PlantOrgan):
        """Radice"""
        pass

    class Flower(PlantOrgan):
        """Fiore"""
        pass

    class Fruit(PlantOrgan):
        """Frutto"""
        pass

    # STAGIONI
    class Season(Thing):
        """Classe base per le stagioni"""
        pass

    class Spring(Season):
        """Primavera"""
        pass

    class Summer(Season):
        """Estate"""
        pass

    class Autumn(Season):
        """Autunno"""
        pass

    class Winter(Season):
        """Inverno"""
        pass

    # CONDIZIONI AMBIENTALI
    class EnvironmentalCondition(Thing):
        """Condizioni ambientali che influenzano le malattie"""
        pass

    class TemperatureRange(EnvironmentalCondition):
        """Range di temperatura"""
        pass

    class HumidityLevel(EnvironmentalCondition):
        """Livello di umidità"""
        pass

    class LightConditions(EnvironmentalCondition):
        """Condizioni di luce"""
        pass

    # OBJECT PROPERTIES (Relazioni tra individui)

    class hasSymptom(ObjectProperty):
        """Una pianta ha un sintomo"""
        domain = [Plant]
        range = [Symptom]

    class hasDisease(ObjectProperty):
        """Una pianta ha una malattia"""
        domain = [Plant]
        range = [Disease]

    class causesSymptom(ObjectProperty):
        """Una malattia causa un sintomo"""
        domain = [Disease]
        range = [Symptom]

    class affectsOrgan(ObjectProperty):
        """Una malattia colpisce un organo"""
        domain = [Disease]
        range = [PlantOrgan]

    class occursInSeason(ObjectProperty):
        """Una malattia si manifesta in una stagione"""
        domain = [Disease]
        range = [Season]

    class requiresCondition(ObjectProperty):
        """Una malattia richiede certe condizioni ambientali"""
        domain = [Disease]
        range = [EnvironmentalCondition]

    class hasOrgan(ObjectProperty):
        """Una pianta ha un organo"""
        domain = [Plant]
        range = [PlantOrgan]

    # Proprietà inversa
    class isSymptomOf(ObjectProperty):
        """Un sintomo è sintomo di una malattia (inversa di causesSymptom)"""
        domain = [Symptom]
        range = [Disease]
        inverse_property = causesSymptom

    # DATA PROPERTIES (Relazioni con valori letterali)

    class severity(DataProperty, FunctionalProperty):
        """Gravità di un sintomo (0.0 - 1.0)"""
        domain = [Symptom]
        range = [float]

    class confidence(DataProperty, FunctionalProperty):
        """Confidenza di una diagnosi (0.0 - 1.0)"""
        range = [float]

    class description(DataProperty):
        """Descrizione testuale di un'entità"""
        range = [str]

    class scientificName(DataProperty, FunctionalProperty):
        """Nome scientifico di una pianta"""
        domain = [Plant]
        range = [str]

    class commonName(DataProperty):
        """Nome comune di una pianta"""
        domain = [Plant]
        range = [str]

    class temperatureMin(DataProperty, FunctionalProperty):
        """Temperatura minima per una condizione"""
        domain = [TemperatureRange]
        range = [float]

    class temperatureMax(DataProperty, FunctionalProperty):
        """Temperatura massima per una condizione"""
        domain = [TemperatureRange]
        range = [float]

# CREAZIONE DI INDIVIDUI DI ESEMPIO (ABox)

with onto:
    # Piante specifiche
    my_rose = Rose("my_rose")
    my_rose.scientificName = "Rosa gallica"
    my_rose.commonName = ["Rosa francese", "Gallic Rose"]
    my_rose.description = ["Rosa ornamentale con fiori profumati"]

    # Organi della rosa
    rose_leaves = Leaf("rose_leaves")  
    rose_stem = Stem("rose_stem")
    rose_flowers = Flower("rose_flowers")

    my_rose.hasOrgan = [rose_leaves, rose_stem, rose_flowers]

    # Malattia specifica
    powdery_mildew_case = PowderyMildew("powdery_mildew_case")
    powdery_mildew_case.description = ["Malattia fungina che causa macchie bianche polverose"]

    # Sintomo specifico  
    white_spots_leaves = WhiteSpots("white_spots_leaves")
    white_spots_leaves.severity = 0.7
    white_spots_leaves.description = ["Macchie bianche polverose sulle foglie"]

    # Stagioni
    summer = Summer("summer")
    autumn = Autumn("autumn")

    # Collegamenti
    powdery_mildew_case.causesSymptom = [white_spots_leaves]
    powdery_mildew_case.affectsOrgan = [rose_leaves]
    powdery_mildew_case.occursInSeason = [summer, autumn]

    my_rose.hasSymptom = [white_spots_leaves]
    my_rose.hasDisease = [powdery_mildew_case]

# RESTRIZIONI DI CLASSE OWL (Equivalenze e Inclusioni)

with onto:
    # Definizioni equivalenti usando costruttori OWL
    # Ogni pianta deve avere almeno un organo
    Plant.is_a.append(hasOrgan.some(PlantOrgan))

    # Ogni malattia deve causare almeno un sintomo  
    Disease.is_a.append(causesSymptom.some(Symptom))

    # Le malattie fungine sono malattie causate da funghi (semplificato)
    # FungalDisease.is_a.append(Disease)  # Già definito tramite ereditarietà

    # Rose devono avere fiori
    Rose.is_a.append(hasOrgan.some(Flower))

def save_ontology(filename="plant_care_ontology.owl"):
    """Salva l'ontologia in formato OWL/XML"""
    onto.save(file=filename, format="rdfxml")
    print(f"Ontologia salvata come: {filename}")

def load_reasoner():
    """Carica il reasoner HermiT per l'inferenza"""
    try:
        # Sincronizzazione con il reasoner HermiT
        with onto:
            sync_reasoner_hermit()
        print("Reasoner HermiT caricato con successo")
        return True
    except Exception as e:
        print(f"Errore nel caricamento del reasoner: {e}")
        return False

def query_ontology():
    """Esegue alcune query di esempio sull'ontologia"""
    print("\n=== QUERY ONTOLOGIA ===")

    # Query 1: Tutte le piante
    print("\n1. Tutte le piante:")
    for plant in Plant.instances():
        print(f"   - {plant.name}: {plant.commonName}")

    # Query 2: Tutti i sintomi con gravità
    print("\n2. Sintomi con gravità:")
    for symptom in Symptom.instances():
        severity_val = symptom.severity if symptom.severity else "Non specificata"
        print(f"   - {symptom.name}: gravità {severity_val}")

    # Query 3: Malattie che colpiscono le foglie
    print("\n3. Malattie che colpiscono le foglie:")
    for disease in Disease.instances():
        affected_organs = disease.affectsOrgan
        if any(isinstance(organ, Leaf) for organ in affected_organs):
            print(f"   - {disease.name}")

    # Query 4: Relazioni malattia-sintomo
    print("\n4. Relazioni malattia → sintomo:")
    for disease in Disease.instances():
        symptoms = disease.causesSymptom
        for symptom in symptoms:
            print(f"   - {disease.name} → {symptom.name}")

def validate_ontology():
    """Valida la consistenza dell'ontologia"""
    try:
        print("\n=== VALIDAZIONE ONTOLOGIA ===")

        # Controlla classi inconsistenti
        inconsistent = list(onto.inconsistent_classes())
        if inconsistent:
            print(f"ATTENZIONE: Classi inconsistenti trovate: {inconsistent}")
        else:
            print("✅ Nessuna classe inconsistente trovata")

        # Statistiche ontologia
        num_classes = len(list(onto.classes()))
        num_properties = len(list(onto.properties())) 
        num_individuals = len(list(onto.individuals()))

        print(f"\nStatistiche Ontologia:")
        print(f"  - Classi: {num_classes}")
        print(f"  - Proprietà: {num_properties}")
        print(f"  - Individui: {num_individuals}")

        return True

    except Exception as e:
        print(f"❌ Errore durante la validazione: {e}")
        return False

if __name__ == "__main__":
    print("PLANT CARE KBS - ONTOLOGIA OWL")
    print("=" * 40)

    # Salva ontologia
    save_ontology()

    # Valida ontologia  
    validate_ontology()

    # Carica reasoner (opzionale, richiede Java)
    reasoner_loaded = load_reasoner()

    # Esegui query di esempio
    query_ontology()

    print("\n✅ Ontologia creata e testata con successo!")
