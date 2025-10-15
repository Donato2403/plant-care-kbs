
% PLANT CARE KBS - REGOLE DIAGNOSTICHE DATALOG/ASP
% Implementazione motore inferenza per diagnosi malattie piante
% Basato su Rappresentazione e Ragionamento Relazionale

%% FATTI BASE (dal modulo ontologia OWL)

% Piante disponibili nel sistema
plant(rose).
plant(tomato).
plant(basil).
plant(olive).

% Organi delle piante
organ(leaf).
organ(stem).
organ(root).
organ(flower).
organ(fruit).

% Stagioni
season(spring).
season(summer).
season(autumn).
season(winter).

% Malattie fungine
disease(powdery_mildew).
disease(black_spot).
disease(rust).
disease(downy_mildew).

% Malattie batteriche  
disease(bacterial_canker).
disease(fire_blight).

% Malattie virali
disease(mosaic_virus).
disease(yellow_dwarf).

% Carenze nutrizionali
disease(nitrogen_deficiency).
disease(iron_chlorosis).

% Sintomi visibili
symptom(white_spots).
symptom(brown_spots).
symptom(black_spots).
symptom(yellow_spots).
symptom(leaf_yellowing).
symptom(wilting).
symptom(deformation).

% Sintomi di crescita
symptom(stunted_growth).
symptom(leaf_drop).
symptom(root_rot).

%% REGOLE DI ASSOCIAZIONE MALATTIA-SINTOMO

% Oidio (Powdery Mildew) - Macchie bianche polverose
causes_symptom(powdery_mildew, white_spots).
causes_symptom(powdery_mildew, leaf_yellowing).
causes_symptom(powdery_mildew, stunted_growth).

% Macchia nera (Black Spot) - Macchie nere su foglie
causes_symptom(black_spot, black_spots).
causes_symptom(black_spot, leaf_yellowing).
causes_symptom(black_spot, leaf_drop).

% Ruggine (Rust) - Macchie arancioni/marroni
causes_symptom(rust, brown_spots).
causes_symptom(rust, yellow_spots).
causes_symptom(rust, leaf_drop).

% Peronospora (Downy Mildew) - Macchie gialle con muffa
causes_symptom(downy_mildew, yellow_spots).
causes_symptom(downy_mildew, wilting).
causes_symptom(downy_mildew, stunted_growth).

% Cancro batterico - Lesioni necrotiche
causes_symptom(bacterial_canker, brown_spots).
causes_symptom(bacterial_canker, wilting).
causes_symptom(bacterial_canker, stunted_growth).

% Colpo di fuoco - Appassimento rapido
causes_symptom(fire_blight, wilting).
causes_symptom(fire_blight, brown_spots).
causes_symptom(fire_blight, leaf_drop).

% Virus del mosaico - Deformazioni e mosaico
causes_symptom(mosaic_virus, deformation).
causes_symptom(mosaic_virus, yellow_spots).
causes_symptom(mosaic_virus, stunted_growth).

% Virus del nanismo giallo - Ingiallimento e nanismo
causes_symptom(yellow_dwarf, leaf_yellowing).
causes_symptom(yellow_dwarf, stunted_growth).

% Carenza di azoto - Ingiallimento foglie inferiori
causes_symptom(nitrogen_deficiency, leaf_yellowing).
causes_symptom(nitrogen_deficiency, stunted_growth).

% Clorosi ferrica - Ingiallimento con nervature verdi
causes_symptom(iron_chlorosis, leaf_yellowing).

%% REGOLE DI ASSOCIAZIONE MALATTIA-ORGANO

% Quali organi colpisce ogni malattia
affects_organ(powdery_mildew, leaf).
affects_organ(powdery_mildew, stem).

affects_organ(black_spot, leaf).

affects_organ(rust, leaf).

affects_organ(downy_mildew, leaf).

affects_organ(bacterial_canker, stem).
affects_organ(bacterial_canker, leaf).

affects_organ(fire_blight, flower).
affects_organ(fire_blight, fruit).
affects_organ(fire_blight, leaf).

affects_organ(mosaic_virus, leaf).

affects_organ(yellow_dwarf, leaf).

affects_organ(nitrogen_deficiency, leaf).

affects_organ(iron_chlorosis, leaf).

%% REGOLE DI ASSOCIAZIONE MALATTIA-STAGIONE

% In quali stagioni si manifestano le malattie
occurs_in_season(powdery_mildew, summer).
occurs_in_season(powdery_mildew, autumn).

occurs_in_season(black_spot, spring).
occurs_in_season(black_spot, summer).

occurs_in_season(rust, autumn).
occurs_in_season(rust, winter).

occurs_in_season(downy_mildew, spring).
occurs_in_season(downy_mildew, autumn).

occurs_in_season(bacterial_canker, spring).

occurs_in_season(fire_blight, spring).
occurs_in_season(fire_blight, summer).

occurs_in_season(mosaic_virus, summer).

occurs_in_season(yellow_dwarf, summer).
occurs_in_season(yellow_dwarf, autumn).

occurs_in_season(nitrogen_deficiency, spring).

occurs_in_season(iron_chlorosis, spring).
occurs_in_season(iron_chlorosis, summer).

%% REGOLE DI ASSOCIAZIONE PIANTA-MALATTIA

% Suscettibilità delle piante alle malattie
susceptible_to(rose, powdery_mildew).
susceptible_to(rose, black_spot).
susceptible_to(rose, rust).

susceptible_to(tomato, mosaic_virus).
susceptible_to(tomato, bacterial_canker).
susceptible_to(tomato, nitrogen_deficiency).

susceptible_to(basil, downy_mildew).
susceptible_to(basil, iron_chlorosis).

susceptible_to(olive, fire_blight).
susceptible_to(olive, bacterial_canker).

%% REGOLE DIAGNOSTICHE PRINCIPALI

% Regola base: una pianta ha una malattia se mostra i sintomi
% e la malattia può colpire quella pianta in quella stagione
has_disease(Plant, Disease) :-
    plant(Plant),
    disease(Disease),
    susceptible_to(Plant, Disease),
    causes_symptom(Disease, Symptom),
    observes_symptom(Plant, Symptom),
    current_season(Season),
    occurs_in_season(Disease, Season).

% Regola di confidenza alta: se una pianta mostra almeno 2 sintomi 
% specifici di una malattia
high_confidence_diagnosis(Plant, Disease) :-
    has_disease(Plant, Disease),
    observes_symptom(Plant, Symptom1),
    observes_symptom(Plant, Symptom2),
    causes_symptom(Disease, Symptom1),
    causes_symptom(Disease, Symptom2),
    Symptom1 != Symptom2.

% Regola per diagnosi differenziale: esclude malattie incompatibili
possible_disease(Plant, Disease) :-
    has_disease(Plant, Disease),
    not conflicting_symptoms(Plant, Disease).

% Conflitto se la pianta mostra sintomi non associati alla malattia
conflicting_symptoms(Plant, Disease) :-
    observes_symptom(Plant, Symptom),
    disease(Disease),
    not causes_symptom(Disease, Symptom),
    specific_symptom(Symptom, OtherDisease),
    Disease != OtherDisease.

% Sintomi specifici per diagnosi differenziale
specific_symptom(white_spots, powdery_mildew).
specific_symptom(black_spots, black_spot).
specific_symptom(deformation, mosaic_virus).

%% REGOLE DI INFERENZA AVANZATE

% Propagazione di sintomi: se un organo è colpito, i sintomi si manifestano
manifests_on_organ(Plant, Symptom, Organ) :-
    has_disease(Plant, Disease),
    causes_symptom(Disease, Symptom),
    affects_organ(Disease, Organ).

% Severità della malattia basata su numero di sintomi
severe_infection(Plant, Disease) :-
    has_disease(Plant, Disease),
    #count{Symptom : causes_symptom(Disease, Symptom), 
                     observes_symptom(Plant, Symptom)} >= 3.

% Infezione lieve con 1-2 sintomi
mild_infection(Plant, Disease) :-
    has_disease(Plant, Disease),
    #count{Symptom : causes_symptom(Disease, Symptom), 
                     observes_symptom(Plant, Symptom)} <= 2,
    #count{Symptom : causes_symptom(Disease, Symptom), 
                     observes_symptom(Plant, Symptom)} >= 1.

% Regola per raccomandazioni di trattamento
recommend_treatment(Plant, Disease, Treatment) :-
    high_confidence_diagnosis(Plant, Disease),
    treatment_for(Disease, Treatment).

% Trattamenti associati alle malattie
treatment_for(powdery_mildew, fungicide_spray).
treatment_for(black_spot, copper_fungicide).
treatment_for(rust, systemic_fungicide).
treatment_for(downy_mildew, organic_fungicide).
treatment_for(bacterial_canker, copper_bactericide).
treatment_for(fire_blight, antibiotic_spray).
treatment_for(mosaic_virus, remove_infected_parts).
treatment_for(yellow_dwarf, vector_control).
treatment_for(nitrogen_deficiency, nitrogen_fertilizer).
treatment_for(iron_chlorosis, iron_chelate).

%% REGOLE PER LA GESTIONE DELL'INCERTEZZA

% Livelli di certezza basati sui sintomi osservati
certainty_level(Plant, Disease, high) :-
    high_confidence_diagnosis(Plant, Disease),
    severe_infection(Plant, Disease).

certainty_level(Plant, Disease, medium) :-
    has_disease(Plant, Disease),
    not high_confidence_diagnosis(Plant, Disease),
    not mild_infection(Plant, Disease).

certainty_level(Plant, Disease, low) :-
    has_disease(Plant, Disease),
    mild_infection(Plant, Disease).

%% REGOLE DI PREVENZIONE

% Condizioni favorevoli allo sviluppo delle malattie
favorable_conditions(Disease) :-
    disease(Disease),
    current_season(Season),
    occurs_in_season(Disease, Season),
    environmental_factor(humidity, high).

% Raccomandazioni preventive
prevention_advice(Plant, Disease, Advice) :-
    susceptible_to(Plant, Disease),
    favorable_conditions(Disease),
    prevention_for(Disease, Advice).

prevention_for(powdery_mildew, increase_air_circulation).
prevention_for(black_spot, avoid_overhead_watering).
prevention_for(rust, remove_fallen_leaves).
prevention_for(downy_mildew, improve_drainage).
prevention_for(bacterial_canker, prune_infected_branches).
prevention_for(fire_blight, sterilize_tools).

%% VINCOLI DI INTEGRITA' (ASP Constraints)

% Una pianta non può avere sintomi contraddittori
:- observes_symptom(Plant, white_spots), 
   observes_symptom(Plant, black_spots),
   not mixed_infection(Plant).

% Non più di 3 malattie contemporaneamente per pianta
:- #count{Disease : has_disease(Plant, Disease)} > 3.

% I sintomi devono essere coerenti con la stagione
:- observes_symptom(Plant, Symptom),
   causes_symptom(Disease, Symptom),
   has_disease(Plant, Disease),
   current_season(Season),
   not occurs_in_season(Disease, Season).
