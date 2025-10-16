# plant_svm_classifier.py
"""
Plant Care KBS - Modulo SVM per Classificazione Malattie Piante
Implementazione Support Vector Machine per diagnosi automatica
Seguendo i principi del corso di Ingegneria della Conoscenza - Apprendimento Supervisionato
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class PlantDiseaseDataGenerator:
    """
    Generatore dataset sintetico per training SVM
    Basato sui dati dell'ontologia OWL e regole Datalog
    """

    def __init__(self):
        self.diseases = [
            'powdery_mildew', 'black_spot', 'rust', 'downy_mildew',
            'bacterial_canker', 'fire_blight', 'mosaic_virus', 
            'yellow_dwarf', 'nitrogen_deficiency', 'iron_chlorosis'
        ]

        self.symptoms = [
            'white_spots', 'black_spots', 'brown_spots', 'yellow_spots',
            'leaf_yellowing', 'wilting', 'deformation', 'stunted_growth',
            'leaf_drop', 'root_rot'
        ]

        self.plants = ['rose', 'tomato', 'basil', 'olive']
        self.seasons = ['spring', 'summer', 'autumn', 'winter']
        self.organs = ['leaf', 'stem', 'root', 'flower', 'fruit']

        # Mappature sintomi-malattie basate su regole Datalog
        self.symptom_disease_map = {
            'powdery_mildew': ['white_spots', 'leaf_yellowing', 'stunted_growth'],
            'black_spot': ['black_spots', 'leaf_yellowing', 'leaf_drop'],
            'rust': ['brown_spots', 'yellow_spots', 'leaf_drop'],
            'downy_mildew': ['yellow_spots', 'wilting', 'stunted_growth'],
            'bacterial_canker': ['brown_spots', 'wilting', 'stunted_growth'],
            'fire_blight': ['wilting', 'brown_spots', 'leaf_drop'],
            'mosaic_virus': ['deformation', 'yellow_spots', 'stunted_growth'],
            'yellow_dwarf': ['leaf_yellowing', 'stunted_growth'],
            'nitrogen_deficiency': ['leaf_yellowing', 'stunted_growth'],
            'iron_chlorosis': ['leaf_yellowing']
        }

        # Suscettibilità piante-malattie
        self.plant_disease_map = {
            'rose': ['powdery_mildew', 'black_spot', 'rust'],
            'tomato': ['mosaic_virus', 'bacterial_canker', 'nitrogen_deficiency'],
            'basil': ['downy_mildew', 'iron_chlorosis'],
            'olive': ['fire_blight', 'bacterial_canker']
        }

    def generate_synthetic_dataset(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Genera dataset sintetico per training SVM

        Args:
            n_samples: Numero di campioni da generare

        Returns:
            DataFrame con features e target
        """
        np.random.seed(42)  # Riproducibilità
        data = []

        for _ in range(n_samples):
            # Scegli pianta casuale
            plant = np.random.choice(self.plants)

            # Scegli malattia compatibile con la pianta
            compatible_diseases = self.plant_disease_map.get(plant, self.diseases)
            disease = np.random.choice(compatible_diseases)

            # Genera sintomi basati sulla malattia
            disease_symptoms = self.symptom_disease_map.get(disease, [])

            # Feature numeriche (simulate severity scores)
            severity_scores = {}
            for symptom in self.symptoms:
                if symptom in disease_symptoms:
                    # Sintomi correlati alla malattia hanno score più alto
                    severity_scores[f'severity_{symptom}'] = np.random.uniform(0.6, 1.0)
                else:
                    # Altri sintomi hanno score più basso
                    severity_scores[f'severity_{symptom}'] = np.random.uniform(0.0, 0.3)

            # Feature categoriche
            sample = {
                'plant': plant,
                'season': np.random.choice(self.seasons),
                'affected_organ': np.random.choice(self.organs),
                'num_symptoms': len([s for s in disease_symptoms if np.random.random() > 0.3]),
                'environmental_humidity': np.random.uniform(0.3, 0.9),
                'environmental_temperature': np.random.uniform(15.0, 35.0),
                **severity_scores,
                'disease': disease  # Target
            }

            # Aggiungi rumore realistico
            if np.random.random() < 0.1:  # 10% di campioni rumorosi
                wrong_disease = np.random.choice(self.diseases)
                sample['disease'] = wrong_disease

            data.append(sample)

        return pd.DataFrame(data)

class PlantSVMClassifier:
    """
    Classificatore SVM per diagnosi malattie delle piante
    Implementazione seguendo principi apprendimento supervisionato
    """

    def __init__(self, kernel: str = 'rbf', random_state: int = 42):
        """
        Inizializza il classificatore SVM

        Args:
            kernel: Tipo di kernel SVM ('linear', 'poly', 'rbf', 'sigmoid')
            random_state: Seed per riproducibilità
        """
        self.kernel = kernel
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.training_history = {}

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepara features per training SVM
        Conversione categoriche e normalizzazione

        Args:
            df: DataFrame con dati

        Returns:
            X: Features preparate, y: Target encodato
        """
        df_processed = df.copy()

        # Encoding features categoriche con one-hot
        categorical_cols = ['plant', 'season', 'affected_organ']
        df_encoded = pd.get_dummies(df_processed, columns=categorical_cols, prefix=categorical_cols)

        # Separa features e target
        X = df_encoded.drop('disease', axis=1).astype(float)
        y = df_processed['disease']

        # Salva nomi features
        self.feature_names = X.columns.tolist()

        return X.values, y.values

    def train(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict:
        """
        Addestra il modello SVM con cross-validation

        Args:
            df: Dataset di training
            test_size: Frazione per test set

        Returns:
            Dizionario con metriche di training
        """
        print("🤖 TRAINING SVM CLASSIFIER PLANT CARE KBS")
        print("=" * 55)

        # Prepara dati
        X, y = self.prepare_features(df)

        # Encode target
        y_encoded = self.label_encoder.fit_transform(y)

        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=self.random_state, 
            stratify=y_encoded
        )

        # Pipeline con normalizzazione
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(random_state=self.random_state, probability=True))
        ])

        # Grid Search per ottimizzazione iperparametri
        param_grid = {
            'svm__kernel': ['linear', 'rbf', 'poly'],
            'svm__C': [0.1, 1, 10, 100],
            'svm__gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
        }

        print("🔍 Ottimizzazione iperparametri con Grid Search...")
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=5, scoring='accuracy', 
            n_jobs=-1, verbose=1
        )

        grid_search.fit(X_train, y_train)

        # Miglior modello
        self.model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        print(f"✅ Migliori parametri: {best_params}")

        # Valutazione su test set
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)

        # Metriche
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, 
                                     target_names=self.label_encoder.classes_,
                                     output_dict=True)

        # Cross-validation score
        cv_scores = cross_val_score(self.model, X, y_encoded, cv=5)

        # Salva storia training
        self.training_history = {
            'best_params': best_params,
            'test_accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': report,
            'feature_names': self.feature_names
        }

        print(f"📊 Accuracy test set: {accuracy:.3f}")
        print(f"📊 CV Score: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")

        return self.training_history

    def predict(self, features: Dict) -> Dict:
        """
        Predice malattia da features input

        Args:
            features: Dizionario con features della pianta

        Returns:
            Predizione con confidenza
        """
        if self.model is None:
            raise ValueError("Modello non addestrato! Eseguire train() prima.")

        # Converti features in formato DataFrame
        df_input = pd.DataFrame([features])

        # Assicurati che tutte le colonne siano presenti
        for feature in self.feature_names:
            if feature not in df_input.columns:
                df_input[feature] = 0

        # Riordina colonne come in training
        df_input = df_input[self.feature_names]

        # Predizione
        X_input = df_input.values
        prediction = self.model.predict(X_input)[0]
        probabilities = self.model.predict_proba(X_input)[0]

        # Decodifica risultato
        predicted_disease = self.label_encoder.inverse_transform([prediction])[0]
        confidence = max(probabilities)

        # Probabilità per tutte le classi
        disease_probabilities = {}
        for i, disease in enumerate(self.label_encoder.classes_):
            disease_probabilities[disease] = probabilities[i]

        return {
            'predicted_disease': predicted_disease,
            'confidence': confidence,
            'probabilities': disease_probabilities
        }

    def evaluate_feature_importance(self) -> Dict:
        """
        Analizza importanza features per modello lineare

        Returns:
            Dizionario con importanza features
        """
        if self.model is None or self.model.named_steps['svm'].kernel != 'linear':
            return {'message': 'Feature importance disponibile solo per kernel lineare'}

        # Coefficienti modello lineare
        coef = self.model.named_steps['svm'].coef_[0]
        importance = np.abs(coef)

        # Crea dizionario importanza
        feature_importance = {}
        for i, feature in enumerate(self.feature_names):
            feature_importance[feature] = importance[i]

        # Ordina per importanza
        sorted_features = sorted(feature_importance.items(), 
                               key=lambda x: x[1], reverse=True)

        return {
            'feature_importance': dict(sorted_features[:10]),  # Top 10
            'message': 'Feature importance calcolata per modello lineare'
        }

    def plot_confusion_matrix(self, df_test: pd.DataFrame, save_path: str = None):
        """
        Genera confusion matrix visualizzata

        Args:
            df_test: Dataset di test
            save_path: Path per salvare il plot
        """
        X_test, y_test = self.prepare_features(df_test)
        y_test_encoded = self.label_encoder.transform(y_test)
        y_pred = self.model.predict(X_test)

        cm = confusion_matrix(y_test_encoded, y_pred)

        plt.figure(figsize=(12, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title('Plant Disease Classification - Confusion Matrix')
        plt.ylabel('True Disease')
        plt.xlabel('Predicted Disease')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def save_model(self, filepath: str):
        """Salva modello addestrato"""
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'training_history': self.training_history
        }
        joblib.dump(model_data, filepath)
        print(f"💾 Modello salvato: {filepath}")

    def load_model(self, filepath: str):
        """Carica modello salvato"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.training_history = model_data['training_history']
        print(f"📂 Modello caricato: {filepath}")

def demo_svm_classification():
    """
    Demo completa del sistema SVM per diagnosi piante
    """
    print("🌱 PLANT CARE KBS - DEMO SVM CLASSIFICATION")
    print("=" * 60)

    # Genera dataset
    print("\n📊 Generazione dataset sintetico...")
    generator = PlantDiseaseDataGenerator()
    df = generator.generate_synthetic_dataset(n_samples=2000)

    print(f"Dataset generato: {len(df)} campioni")
    print(f"Malattie: {df['disease'].unique()}")
    print(f"Distribuzione classi:\n{df['disease'].value_counts()}")

    # Inizializza e addestra SVM
    print("\n🤖 Training SVM Classifier...")
    svm_classifier = PlantSVMClassifier()

    # Training
    metrics = svm_classifier.train(df, test_size=0.3)

    # Valutazione feature importance
    importance = svm_classifier.evaluate_feature_importance()
    if 'feature_importance' in importance:
        print("\n🔍 Top Feature Importanti:")
        for feature, score in list(importance['feature_importance'].items())[:5]:
            print(f"  • {feature}: {score:.3f}")

    # Test predizioni
    print("\n🧪 Test Predizioni:")

    # Scenario 1: Rosa con oidio
    test_case_1 = {
        'plant_rose': 1, 'plant_tomato': 0, 'plant_basil': 0, 'plant_olive': 0,
        'season_summer': 1, 'season_spring': 0, 'season_autumn': 0, 'season_winter': 0,
        'affected_organ_leaf': 1, 'affected_organ_stem': 0, 'affected_organ_root': 0,
        'affected_organ_flower': 0, 'affected_organ_fruit': 0,
        'num_symptoms': 3,
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

    result_1 = svm_classifier.predict(test_case_1)
    print(f"\n🌹 Caso Rosa Estiva:")
    print(f"   Malattia predetta: {result_1['predicted_disease']}")
    print(f"   Confidenza: {result_1['confidence']:.3f}")

    # Scenario 2: Pomodoro con virus
    test_case_2 = {
        'plant_rose': 0, 'plant_tomato': 1, 'plant_basil': 0, 'plant_olive': 0,
        'season_summer': 1, 'season_spring': 0, 'season_autumn': 0, 'season_winter': 0,
        'affected_organ_leaf': 1, 'affected_organ_stem': 0, 'affected_organ_root': 0,
        'affected_organ_flower': 0, 'affected_organ_fruit': 0,
        'num_symptoms': 3,
        'environmental_humidity': 0.6,
        'environmental_temperature': 25.0,
        'severity_deformation': 0.9,
        'severity_yellow_spots': 0.8,
        'severity_stunted_growth': 0.7,
        'severity_white_spots': 0.1,
        'severity_black_spots': 0.1,
        'severity_brown_spots': 0.1,
        'severity_wilting': 0.2,
        'severity_leaf_yellowing': 0.3,
        'severity_leaf_drop': 0.2,
        'severity_root_rot': 0.1
    }

    result_2 = svm_classifier.predict(test_case_2)
    print(f"\n🍅 Caso Pomodoro con Deformazioni:")
    print(f"   Malattia predetta: {result_2['predicted_disease']}")
    print(f"   Confidenza: {result_2['confidence']:.3f}")

    # Salva modello
    svm_classifier.save_model('plant_svm_model.pkl')

    print("\n✅ Demo SVM completata con successo!")
    print("\n📈 Risultati Finali:")
    print(f"   • Accuracy: {metrics['test_accuracy']:.3f}")
    print(f"   • CV Score: {metrics['cv_mean']:.3f} (±{metrics['cv_std']:.3f})")
    print(f"   • Modello salvato: plant_svm_model.pkl")

if __name__ == "__main__":
    demo_svm_classification()
