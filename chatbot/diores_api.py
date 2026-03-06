"""
Client API pour les modèles DIORES (Classifier et Lasso)
Appelle les modèles de prédiction pour obtenir les probabilités d'orientation et de réussite
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging
import pickle

# Ajouter le chemin des modèles
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

logger = logging.getLogger(__name__)

# Importer le client V4
V4_AVAILABLE = False
DioresAPIClientV4 = None
try:
    from chatbot.diores_api_v4 import DioresAPIClientV4
    V4_AVAILABLE = True
except ImportError as e:
    logger.warning(f"DioresAPIClientV4 non disponible: {e}. Utilisation de l'ancienne version si disponible.")
    V4_AVAILABLE = False
    DioresAPIClientV4 = None

# Les imports pour l'ancienne version seront faits de manière lazy
DioresPredictor = None
DataFrameProcessor = None


class DioresAPIClient:
    """
    Client pour appeler les modèles DIORES
    
    Utilise:
    - DIORES-Classifier: pour prédire admission, session, mention
    - DIORES-Lasso: pour prédire les scores et probabilités
    """
    
    def __init__(self, use_v4: bool = True, doc_version: Optional[str] = None, auto_select_best: bool = True):
        """
        Initialise le client avec les modèles DIORES
        
        Args:
            use_v4: Si True, utilise les modèles V4 (recommandé). Sinon, utilise V2
            doc_version: Version du document à utiliser pour V4 ("Doc1", "Doc2", ou "Doc3")
                        Si None et auto_select_best=True, sélectionne automatiquement le meilleur modèle
            auto_select_best: Si True, sélectionne automatiquement le meilleur modèle par formation
                             basé sur le RMSE le plus faible (recommandé)
        """
        self.use_v4 = use_v4 and V4_AVAILABLE
        
        if self.use_v4:
            # Utiliser le client V4
            try:
                self.v4_client = DioresAPIClientV4(doc_version=doc_version, auto_select_best=auto_select_best)
                if hasattr(self.v4_client, 'best_models') and self.v4_client.best_models:
                    logger.info(f"Client DIORES V4 initialise avec selection automatique des meilleurs modeles")
                else:
                    doc_used = doc_version or "Doc1"
                    logger.info(f"Client DIORES V4 initialise avec {doc_used}")
            except Exception as e:
                logger.error(f"Erreur lors de l'initialisation du client V4: {str(e)}")
                logger.warning("Fallback vers l'ancienne version")
                self.use_v4 = False
                self.v4_client = None
        
        if not self.use_v4:
            # Fallback vers l'ancienne version V2
            self.models_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "Models"
            )
            
            # Essayer d'importer les modèles de manière lazy
            global DioresPredictor, DataFrameProcessor
            
            if DioresPredictor is None:
                try:
                    import importlib.util
                    spec_path = os.path.join(os.path.dirname(__file__), "..", "Models", "Utils", "diores_predictor.py")
                    spec = importlib.util.spec_from_file_location("diores_predictor", spec_path)
                    if spec and spec.loader:
                        diores_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(diores_module)
                        DioresPredictor = diores_module.DioresPredictor
                        DataFrameProcessor = diores_module.DataFrameProcessor
                        logger.info("DioresPredictor importe avec succes")
                    else:
                        raise ImportError("Impossible de charger le module diores_predictor")
                except Exception as e:
                    logger.warning(f"Impossible d'importer DioresPredictor: {e}. Les predictions DIORES ne seront pas disponibles.")
                    DioresPredictor = None
                    DataFrameProcessor = None
            
            # Initialiser le prédicteur DIORES
            if DioresPredictor is None or DataFrameProcessor is None:
                logger.warning("DioresPredictor non disponible. Les predictions DIORES seront desactivees.")
                self.predictor = None
                self.processor = None
                return
            
            try:
                # Changer le répertoire de travail pour charger les modèles
                original_cwd = os.getcwd()
                os.chdir(os.path.join(os.path.dirname(__file__), ".."))
                
                self.predictor = DioresPredictor()
                self.processor = DataFrameProcessor(pd.DataFrame())
                
                os.chdir(original_cwd)
                
                logger.info("Modeles DIORES V2 charges avec succes")
            except Exception as e:
                logger.error(f"Erreur lors du chargement des modeles DIORES: {str(e)}")
                logger.warning("Le chatbot fonctionnera sans les predictions DIORES")
                self.predictor = None
                self.processor = None
    
    async def predict(
        self,
        profile: Dict,
        formation: str
    ) -> Dict:
        """
        Effectue les prédictions DIORES pour un profil d'étudiant
        
        Args:
            profile: Dictionnaire avec le profil complet de l'étudiant
            formation: Formation souhaitée (L1MPI, L1BCGS, L1PCSM)
        
        Returns:
            Dictionnaire avec:
            - prob_orientation: Probabilité d'être orienté vers la formation
            - prob_reussite: Probabilité de réussite en L1
            - admission: Prédiction d'admission (AUTORISE, PASSE, NON ADMIS)
            - session: Session prédite (Première Session, Deuxième Session)
            - mention: Mention prédite (Passable, Assez-Bien, Bien, Très-Bien)
        """
        # Utiliser le client V4 si disponible
        if self.use_v4 and self.v4_client:
            try:
                return await self.v4_client.predict(profile, formation)
            except Exception as e:
                logger.error(f"Erreur avec le client V4: {str(e)}")
                logger.warning("Fallback vers l'ancienne version")
                # Continuer avec l'ancienne version
        
        # Fallback vers l'ancienne version V2
        if self.predictor is None:
            logger.warning("Les modeles DIORES ne sont pas disponibles. Retour de predictions par defaut.")
            return {
                "prob_orientation": 50.0,
                "prob_reussite": 50.0,
                "admission": "AUTORISE",
                "session": "Premiere Session",
                "mention": "Assez-Bien",
                "formation": formation
            }
        
        try:
            # Préparer les données pour la prédiction
            df = self._profile_to_dataframe(profile, formation)
            
            # Traiter les données
            processed_df = self.processor.process_all()
            
            # Prédire avec DIORES-Classifier
            predictions = self.predictor.predict(processed_df)
            
            if not predictions or len(predictions) == 0:
                raise ValueError("Aucune prediction retournee")
            
            pred = predictions[0]
            
            # Calculer les probabilités
            prob_orientation = self._calculate_orientation_probability(
                profile, formation, pred
            )
            
            prob_reussite = self._calculate_success_probability(pred)
            
            return {
                "prob_orientation": prob_orientation,
                "prob_reussite": prob_reussite,
                "admission": pred.get("admission"),
                "session": pred.get("session"),
                "mention": pred.get("mention"),
                "formation": formation
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la prediction: {str(e)}", exc_info=True)
            raise
    
    def _profile_to_dataframe(
        self,
        profile: Dict,
        formation: str
    ) -> pd.DataFrame:
        """
        Convertit le profil utilisateur en DataFrame pour les modèles
        
        Args:
            profile: Profil de l'étudiant
            formation: Formation souhaitée
        
        Returns:
            DataFrame avec les features nécessaires
        """
        notes = profile.get("notes", {})
        
        # Créer un DataFrame avec les données du profil
        data = {
            "Année BAC": [2024],  # Année actuelle ou à adapter
            "Nbre Fois au BAC": [1],
            "Groupe Résultat": [0],
            "Moy. nde": [0],
            "Moy. ère": [0],
            "Moy. S Term.": [0],
            "Moy. S Term..1": [0],
            "MATH": [notes.get("MATH", 0)],
            "SCPH": [notes.get("SCPH", 0)],
            "FR": [notes.get("FR", 0)],
            "PHILO": [notes.get("PHILO", 0)],
            "AN": [notes.get("AN", 0)],
            "Tot. Pts au Grp.": [0],
            "Moyenne au Grp.": [0],
            "Moy. Gle": [sum(notes.values()) / len(notes) if notes else 0],
            "Moy. sur Mat.Fond.": [0],
            "Age en Décembre 2018": [profile.get("age", 18)],
            "Sexe_F": [1 if profile.get("sexe") == "F" else 0],
            "Sexe_M": [1 if profile.get("sexe") == "M" else 0],
            "Série_S1": [1 if profile.get("serie") == "S1" else 0],
            "Série_S2": [1 if profile.get("serie") == "S2" else 0],
            "Série_S3": [1 if profile.get("serie") == "S3" else 0],
            "Mention_ABien": [0],
            "Mention_Bien": [0],
            "Mention_Pass": [0],
            "Résidence": [profile.get("residence", "Dakar")],
            "Ets. de provenance": [""],
            "Centre d'Ec.": [""],
            "Académie de l'Ets. Prov.": [profile.get("academie", "Dakar")],
            "REGION_DE_NAISSANCE": [""],
            "Academie perf.": [0]
        }
        
        df = pd.DataFrame(data)
        
        # Réinitialiser le processeur avec ce DataFrame
        self.processor = DataFrameProcessor(df)
        
        return df
    
    def _calculate_orientation_probability(
        self,
        profile: Dict,
        formation: str,
        prediction: Dict
    ) -> float:
        """
        Calcule la probabilité d'être orienté vers la formation
        
        Utilise la formule basée sur le rang (voir mémoire, équation 4.5)
        
        Args:
            profile: Profil de l'étudiant
            formation: Formation souhaitée
            prediction: Prédictions du modèle
        
        Returns:
            Probabilité en pourcentage (0-100)
        """
        # Si non admis, probabilité faible
        if prediction.get("admission") == "NON ADMIS":
            return 20.0
        
        # Calculer un score basé sur les notes et la prédiction
        notes = profile.get("notes", {})
        moyenne = sum(notes.values()) / len(notes) if notes else 0
        
        # Score de base basé sur la moyenne
        base_score = (moyenne / 20.0) * 100
        
        # Ajustements selon la prédiction
        if prediction.get("admission") == "PASSE":
            base_score += 20
        elif prediction.get("admission") == "AUTORISE":
            base_score += 10
        
        if prediction.get("session") == "Première Session":
            base_score += 10
        
        # Limiter entre 0 et 100
        return min(100.0, max(0.0, base_score))
    
    def _calculate_success_probability(self, prediction: Dict) -> float:
        """
        Calcule la probabilité de réussite en L1
        
        Utilise la formule du mémoire (équation 4.3):
        P(R) = w_A * P(A) + w_S * P(S|A) + w_M * P(M|A, S)
        
        Args:
            prediction: Prédictions du modèle
        
        Returns:
            Probabilité en pourcentage (0-100)
        """
        # Poids selon le mémoire
        w_A = 0.5
        w_S = 0.25
        w_M = 0.25
        
        # P(A): Probabilité d'admission
        if prediction.get("admission") == "NON ADMIS":
            P_A = 0
        else:
            P_A = 1
        
        # P(S|A): Probabilité de première session conditionnelle à l'admission
        if prediction.get("session") == "Première Session":
            P_S_given_A = 1
        else:
            P_S_given_A = 0
        
        # P(M|A, S): Probabilité de mention conditionnelle
        mention = prediction.get("mention", "Passable")
        if mention == "Passable":
            P_M_given_A_S = 0
        elif mention == "Assez-Bien":
            P_M_given_A_S = 0.5
        elif mention == "Bien":
            P_M_given_A_S = 0.75
        else:  # Très-Bien
            P_M_given_A_S = 1
        
        # Calcul de la probabilité de réussite
        P_R = w_A * P_A + w_S * P_S_given_A + w_M * P_M_given_A_S
        
        # Convertir en pourcentage
        return P_R * 100

