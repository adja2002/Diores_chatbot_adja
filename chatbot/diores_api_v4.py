"""
Client API pour les modèles DIORES V4 (Classifier et Lasso)
Utilise les modèles V4 organisés par formation (L1MPI, L1BCGS, L1PCSM) et Doc (Doc1, Doc2, Doc3)
Implémente les formules de probabilité selon le mémoire:
- P(R) = w_A * P(A) + w_S * P(S|A) + w_M * P(M|A, S)  (probabilité de réussite)
- P(x) = 0.5 + (n - rang(x)) / (2n)  (probabilité d'orientation)
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging
import pickle

logger = logging.getLogger(__name__)

# Pour le calcul du rang avec distribution normale
try:
    from scipy.stats import norm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy non disponible, utilisation d'une approximation simple pour le rang")

# Ajouter le chemin des modèles
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

logger = logging.getLogger(__name__)

# Les imports seront faits de manière lazy
DataFrameProcessor = None


class DioresAPIClientV4:
    """
    Client pour appeler les modèles DIORES V4
    
    Utilise:
    - DIORES-Classifier V4: pour prédire admission, session, mention (par formation)
    - DIORES-Lasso V4: pour prédire les scores et calculer le rang (par formation)
    """
    
    def __init__(self, doc_version: Optional[str] = None, auto_select_best: bool = True):
        """
        Initialise le client avec les modèles DIORES V4
        
        Args:
            doc_version: Version du document à utiliser ("Doc1", "Doc2", ou "Doc3")
                        Si None et auto_select_best=True, sélectionne automatiquement le meilleur modèle
            auto_select_best: Si True, sélectionne automatiquement le meilleur modèle par formation
                             basé sur le RMSE le plus faible
        """
        self.models_base_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "Models",
            "V4"
        )
        
        # Sélectionner automatiquement les meilleurs modèles si demandé
        if auto_select_best and doc_version is None:
            self.best_models = DioresAPIClientV4._select_best_models_static(self.models_base_path)
            logger.info(f"Meilleurs modeles selectionnes: {self.best_models}")
        else:
            self.best_models = None
            self.doc_version = doc_version or "Doc1"
        
        # Essayer d'importer DataFrameProcessor (original ou version simplifiée)
        global DataFrameProcessor
        
        if DataFrameProcessor is None:
            # Essayer d'abord la version originale
            try:
                import importlib.util
                spec_path = os.path.join(os.path.dirname(__file__), "..", "Models", "Utils", "diores_predictor.py")
                spec = importlib.util.spec_from_file_location("diores_predictor", spec_path)
                if spec and spec.loader:
                    diores_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(diores_module)
                    DataFrameProcessor = diores_module.DataFrameProcessor
                    logger.info("DataFrameProcessor original importe avec succes")
            except Exception as e:
                logger.warning(f"Impossible d'importer DataFrameProcessor original: {e}")
                # Essayer la version simplifiée
                try:
                    from chatbot.dataframe_processor_simple import DataFrameProcessorSimple
                    DataFrameProcessor = DataFrameProcessorSimple
                    logger.info("DataFrameProcessorSimple importe avec succes (version simplifiee)")
                except Exception as e2:
                    logger.error(f"Impossible d'importer DataFrameProcessorSimple: {e2}")
                    DataFrameProcessor = None
        
        if DataFrameProcessor is None:
            logger.error("Aucun DataFrameProcessor disponible. Les predictions DIORES seront desactivees.")
            self.processor = None
            self.models = {}
            self.lasso_models = {}
            self.lasso_scalers = {}
            self.lasso_info = {}
            return
        
        # Charger les modèles pour chaque formation
        self.models = {}
        self.lasso_models = {}
        self.lasso_scalers = {}
        self.lasso_info = {}
        
        formations = ["L1MPI", "L1BCGS", "L1PCSM"]
        
        for formation in formations:
            # Utiliser le meilleur modèle pour cette formation si disponible
            if self.best_models and formation in self.best_models:
                doc_version_to_use = self.best_models[formation]
            else:
                doc_version_to_use = self.doc_version if hasattr(self, 'doc_version') else "Doc1"
            
            try:
                # Charger les classificateurs
                classifier_path = os.path.join(
                    self.models_base_path,
                    "Classifiers",
                    formation,
                    doc_version_to_use
                )
                
                # Modèle admission
                admi_path = os.path.join(classifier_path, "admi_non_admi_best_model.pkl")
                if os.path.exists(admi_path):
                    with open(admi_path, 'rb') as f:
                        self.models[f"{formation}_admi"] = pickle.load(f)
                
                # Modèle session
                session_path = os.path.join(classifier_path, "session_best_model.pkl")
                if os.path.exists(session_path):
                    with open(session_path, 'rb') as f:
                        self.models[f"{formation}_session"] = pickle.load(f)
                
                # Modèle mention
                mention_path = os.path.join(classifier_path, "mention_best_model.pkl")
                if os.path.exists(mention_path):
                    with open(mention_path, 'rb') as f:
                        self.models[f"{formation}_mention"] = pickle.load(f)
                
                # Charger les modèles Lasso
                lasso_path = os.path.join(
                    self.models_base_path,
                    "LassoGlobal",
                    formation,
                    doc_version_to_use
                )
                
                # Modèle Lasso
                lasso_model_path = os.path.join(lasso_path, "lasso_globale_model.pkl")
                if os.path.exists(lasso_model_path):
                    with open(lasso_model_path, 'rb') as f:
                        self.lasso_models[formation] = pickle.load(f)
                
                # Scaler Lasso
                lasso_scaler_path = os.path.join(lasso_path, "lasso_globale_scaler.pkl")
                if os.path.exists(lasso_scaler_path):
                    with open(lasso_scaler_path, 'rb') as f:
                        self.lasso_scalers[formation] = pickle.load(f)
                
                # Info Lasso (contient peut-être des données de référence pour le calcul du rang)
                lasso_info_path = os.path.join(lasso_path, "lasso_globale_info.pkl")
                if os.path.exists(lasso_info_path):
                    with open(lasso_info_path, 'rb') as f:
                        self.lasso_info[formation] = pickle.load(f)
                
                logger.info(f"Modeles V4 charges pour {formation} ({doc_version_to_use})")
                
            except Exception as e:
                logger.error(f"Erreur lors du chargement des modeles pour {formation}: {str(e)}")
        
        # Mappings pour interpréter les prédictions
        self.resultat_map = {0: 'NON ADMIS', 1: 'AUTORISE', 2: 'PASSE'}
        self.session_map = {0: 'Deuxième Session', 1: 'Première Session'}
        self.mention_map = {0: 'Passable', 1: 'Assez-Bien', 2: 'Bien', 3: 'Très-Bien'}
        
        # Initialiser le processeur
        self.processor = DataFrameProcessor(pd.DataFrame()) if DataFrameProcessor else None
        
        logger.info("Client DIORES V4 initialise avec succes")
    
    async def predict(
        self,
        profile: Dict,
        formation: str
    ) -> Dict:
        """
        Effectue les prédictions DIORES V4 pour un profil d'étudiant
        
        Args:
            profile: Dictionnaire avec le profil complet de l'étudiant
            formation: Formation souhaitée (L1MPI, L1BCGS, L1PCSM)
        
        Returns:
            Dictionnaire avec:
            - prob_orientation: Probabilité d'être orienté vers la formation (formule 4.5)
            - prob_reussite: Probabilité de réussite en L1 (formule 4.3)
            - admission: Prédiction d'admission (AUTORISE, PASSE, NON ADMIS)
            - session: Session prédite (Première Session, Deuxième Session)
            - mention: Mention prédite (Passable, Assez-Bien, Bien, Très-Bien)
        """
        if not self.models or not self.processor:
            logger.warning("Les modeles DIORES V4 ne sont pas disponibles. Retour de predictions par defaut.")
            return {
                "prob_orientation": 50.0,
                "prob_reussite": 50.0,
                "admission": "AUTORISE",
                "session": "Première Session",
                "mention": "Assez-Bien",
                "formation": formation
            }
        
        try:
            # Normaliser le nom de la formation
            formation = formation.upper()
            if formation not in ["L1MPI", "L1BCGS", "L1PCSM"]:
                # Essayer de convertir
                if "MPI" in formation:
                    formation = "L1MPI"
                elif "BCGS" in formation:
                    formation = "L1BCGS"
                elif "PCSM" in formation:
                    formation = "L1PCSM"
                else:
                    raise ValueError(f"Formation non reconnue: {formation}")
            
            # Préparer les données pour la prédiction
            df = self._profile_to_dataframe(profile, formation)
            
            # Traiter les données
            self.processor = DataFrameProcessor(df)
            processed_df = self.processor.process_all()
            
            # Prédire avec les classificateurs V4
            predictions = self._predict_classifiers(processed_df, formation)
            
            if not predictions:
                raise ValueError("Aucune prediction retournee")
            
            pred = predictions[0]
            
            # Calculer le score avec le modèle Lasso pour le rang
            lasso_score = self._calculate_lasso_score(processed_df, formation)
            
            # Calculer les probabilités selon les formules du mémoire
            prob_orientation = self._calculate_orientation_probability(
                lasso_score, formation, pred
            )
            
            prob_reussite = self._calculate_success_probability(pred)
            
            # Déterminer quel modèle a été utilisé
            model_used = "Inconnu"
            if self.best_models and formation in self.best_models:
                model_used = self.best_models[formation]
            elif hasattr(self, 'doc_version'):
                model_used = self.doc_version
            else:
                model_used = "Doc1"
            
            return {
                "prob_orientation": round(prob_orientation, 1),
                "prob_reussite": round(prob_reussite, 1),
                "admission": pred.get("admission"),
                "session": pred.get("session"),
                "mention": pred.get("mention"),
                "formation": formation,
                "model_used": model_used  # Ajouter l'info sur le modèle utilisé
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la prediction: {str(e)}", exc_info=True)
            raise
    
    def _predict_classifiers(
        self,
        processed_df: pd.DataFrame,
        formation: str
    ) -> List[Dict]:
        """
        Prédit avec les classificateurs V4
        
        Args:
            processed_df: DataFrame traité
            formation: Formation (L1MPI, L1BCGS, L1PCSM)
        
        Returns:
            Liste de dictionnaires avec les prédictions
        """
        resultats = []
        
        # Prédire l'admission
        admi_key = f"{formation}_admi"
        if admi_key not in self.models:
            raise ValueError(f"Modele d'admission non trouve pour {formation}")
        
        # Préparer le DataFrame pour la prédiction avec les bonnes features
        admi_model = self.models[admi_key]
        classifier_df = self._prepare_dataframe_for_classifier(processed_df, admi_model, formation, "admission")
        
        admissions = admi_model.predict(classifier_df)
        
        for i, est_admis in enumerate(admissions):
            if est_admis == 0:  # NON ADMIS
                resultats.append({
                    'admission': self.resultat_map[0],
                    'session': None,
                    'mention': None
                })
            else:  # AUTORISE ou PASSE
                # Prédire la session
                session_key = f"{formation}_session"
                if session_key in self.models:
                    session_model = self.models[session_key]
                    session_df = self._prepare_dataframe_for_classifier(processed_df.iloc[[i]], session_model, formation, "session")
                    session_pred = session_model.predict(session_df)[0]
                else:
                    session_pred = 1  # Par défaut première session
                
                if session_pred == 0:  # Deuxième Session
                    resultats.append({
                        'admission': self.resultat_map[est_admis],
                        'session': self.session_map[0],
                        'mention': None
                    })
                else:  # Première Session
                    # Prédire la mention
                    mention_key = f"{formation}_mention"
                    if mention_key in self.models:
                        mention_model = self.models[mention_key]
                        mention_df = self._prepare_dataframe_for_classifier(processed_df.iloc[[i]], mention_model, formation, "mention")
                        mention_pred = mention_model.predict(mention_df)[0]
                    else:
                        mention_pred = 1  # Par défaut Assez-Bien
                    
                    resultats.append({
                        'admission': self.resultat_map[est_admis],
                        'session': self.session_map[1],
                        'mention': self.mention_map.get(mention_pred, "Assez-Bien")
                    })
        
        return resultats
    
    def _prepare_dataframe_for_classifier(
        self,
        processed_df: pd.DataFrame,
        model,
        formation: str,
        model_type: str
    ) -> pd.DataFrame:
        """
        Prépare un DataFrame avec les features attendues par un classificateur
        
        Args:
            processed_df: DataFrame traité
            model: Modèle sklearn
            formation: Formation
            model_type: Type de modèle ("admission", "session", "mention")
        
        Returns:
            DataFrame avec les features dans le bon ordre
        """
        # Obtenir les features attendues par le modèle
        if hasattr(model, 'feature_names_in_'):
            expected_features = model.feature_names_in_
        else:
            # Fallback: utiliser les features du processed_df
            return processed_df
        
        # Créer un DataFrame avec uniquement les features attendues
        classifier_df = pd.DataFrame(index=processed_df.index)
        
        for feat in expected_features:
            if feat in processed_df.columns:
                classifier_df[feat] = processed_df[feat]
            else:
                # Feature manquante, mettre à 0
                logger.warning(f"Feature {feat} manquante pour le modèle {model_type} {formation}, mise à 0")
                classifier_df[feat] = 0
        
        # S'assurer que l'ordre des colonnes correspond à celui attendu par le modèle
        classifier_df = classifier_df[expected_features]
        
        return classifier_df
    
    def _calculate_lasso_score(
        self,
        processed_df: pd.DataFrame,
        formation: str
    ) -> float:
        """
        Calcule le score avec le modèle Lasso pour déterminer le rang
        
        Args:
            processed_df: DataFrame traité
            formation: Formation (L1MPI, L1BCGS, L1PCSM)
        
        Returns:
            Score calculé par le modèle Lasso
        """
        if formation not in self.lasso_models:
            logger.warning(f"Modele Lasso non trouve pour {formation}, utilisation d'un score par defaut")
            # Calculer un score basé sur la moyenne
            if 'Moy. Gle' in processed_df.columns:
                return float(processed_df['Moy. Gle'].iloc[0])
            return 12.0  # Score par défaut
        
        try:
            # Obtenir les features attendues par le scaler Lasso
            if formation not in self.lasso_scalers:
                logger.warning(f"Scaler Lasso non trouve pour {formation}")
                if 'Moy. Gle' in processed_df.columns:
                    return float(processed_df['Moy. Gle'].iloc[0])
                return 12.0
            
            scaler = self.lasso_scalers[formation]
            
            # Obtenir les features attendues par le scaler
            if hasattr(scaler, 'feature_names_in_'):
                expected_features = scaler.feature_names_in_
            else:
                # Fallback: utiliser les features du processed_df
                expected_features = processed_df.columns.tolist()
            
            # Créer un DataFrame avec uniquement les features attendues
            lasso_df = pd.DataFrame(index=processed_df.index)
            
            for feat in expected_features:
                # Gérer les features spéciales comme S1, S2, S3
                if feat == 'S1':
                    # Créer S1 à partir de Série_S1 ou Série_Encode
                    if 'Série_S1' in processed_df.columns:
                        lasso_df['S1'] = processed_df['Série_S1']
                    elif 'Série_Encode' in processed_df.columns:
                        lasso_df['S1'] = (processed_df['Série_Encode'] == 0).astype(int)
                    else:
                        lasso_df['S1'] = 0
                elif feat == 'S2':
                    # Créer S2 à partir de Série_S2 ou Série_Encode
                    if 'Série_S2' in processed_df.columns:
                        lasso_df['S2'] = processed_df['Série_S2']
                    elif 'Série_Encode' in processed_df.columns:
                        lasso_df['S2'] = (processed_df['Série_Encode'] == 1).astype(int)
                    else:
                        lasso_df['S2'] = 0
                elif feat == 'S3':
                    # Créer S3 à partir de Série_S3 ou Série_Encode
                    if 'Série_S3' in processed_df.columns:
                        lasso_df['S3'] = processed_df['Série_S3']
                    elif 'Série_Encode' in processed_df.columns:
                        lasso_df['S3'] = (processed_df['Série_Encode'] == 2).astype(int)
                    else:
                        lasso_df['S3'] = 0
                elif feat in processed_df.columns:
                    lasso_df[feat] = processed_df[feat]
                else:
                    # Feature manquante, mettre à 0
                    logger.warning(f"Feature {feat} manquante pour le modèle Lasso {formation}, mise à 0")
                    lasso_df[feat] = 0
            
            # S'assurer que l'ordre des colonnes correspond à celui attendu par le scaler
            lasso_df = lasso_df[expected_features]
            
            # Normaliser les données avec le scaler
            X_scaled = scaler.transform(lasso_df)
            
            # Prédire avec le modèle Lasso
            score = self.lasso_models[formation].predict(X_scaled)[0]
            return float(score)
        except Exception as e:
            logger.error(f"Erreur lors du calcul du score Lasso: {str(e)}", exc_info=True)
            # Fallback: utiliser la moyenne générale
            if 'Moy. Gle' in processed_df.columns:
                return float(processed_df['Moy. Gle'].iloc[0])
            return 12.0
    
    def _calculate_orientation_probability(
        self,
        lasso_score: float,
        formation: str,
        prediction: Dict
    ) -> float:
        """
        Calcule la probabilité d'être orienté vers la formation
        
        Utilise la formule du mémoire (équation 4.5):
        P(x) = 0.5 + (n - rang(x)) / (2n)
        
        Args:
            lasso_score: Score calculé par le modèle Lasso
            formation: Formation souhaitée
            prediction: Prédictions du classificateur
        
        Returns:
            Probabilité en pourcentage (0-100)
        """
        # Si non admis, probabilité faible
        if prediction.get("admission") == "NON ADMIS":
            return 20.0
        
        # Pour calculer le rang selon la formule 4.4 du mémoire:
        # rang(x) = 1 + sum(1_{x_i > x})
        # On a besoin des scores de référence de la population
        
        # Estimation du nombre d'observations (basé sur les données typiques)
        # Pour une formation, on peut estimer n entre 200-500 étudiants par an
        n = 300  # Estimation raisonnable
        
        # Approximation du rang basé sur le score Lasso
        # Les scores Lasso sont généralement dans une plage de 8-18
        # On utilise une distribution normale approximative:
        # - Score moyen estimé: 12.5
        # - Écart-type estimé: 2.5
        # - Plus le score est élevé, meilleur est le rang (plus petit)
        
        score_moyen_estime = 12.5
        ecart_type_estime = 2.5
        
        # Calculer le z-score
        z_score = (lasso_score - score_moyen_estime) / ecart_type_estime
        
        # Convertir z-score en percentile (0-1) puis en rang
        if SCIPY_AVAILABLE:
            try:
                percentile = norm.cdf(z_score)
                # Le rang est l'inverse du percentile (plus le percentile est élevé, meilleur est le rang)
                rang = max(1, int((1 - percentile) * n))
            except:
                # Fallback si erreur
                rang = self._estimate_rank_from_score(lasso_score, n)
        else:
            # Fallback si scipy n'est pas disponible
            rang = self._estimate_rank_from_score(lasso_score, n)
        
        # Appliquer la formule 4.5 du mémoire: P(x) = 0.5 + (n - rang(x)) / (2n)
        P_x = 0.5 + (n - rang) / (2 * n)
        
        # Convertir en pourcentage
        prob = P_x * 100
        
        # Ajustements selon la prédiction du classificateur
        if prediction.get("admission") == "PASSE":
            prob += 15
        elif prediction.get("admission") == "AUTORISE":
            prob += 5
        
        if prediction.get("session") == "Première Session":
            prob += 10
        
        # Limiter entre 0 et 100
        return min(100.0, max(0.0, prob))
    
    def _calculate_success_probability(self, prediction: Dict) -> float:
        """
        Calcule la probabilité de réussite en L1
        
        Utilise la formule du mémoire (équation 4.3):
        P(R) = w_A * P(A) + w_S * P(S|A) + w_M * P(M|A, S)
        
        Avec:
        - w_A = 0.5
        - w_S = 0.25
        - w_M = 0.25
        
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
        
        # Calcul de la probabilité de réussite selon la formule 4.3
        P_R = w_A * P_A + w_S * P_S_given_A + w_M * P_M_given_A_S
        
        # Convertir en pourcentage
        return P_R * 100
    
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
        
        # Calculer la moyenne générale
        moyenne_gle = sum(notes.values()) / len(notes) if notes else 0
        
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
            "Moy. Gle": [moyenne_gle],
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
        
        # Ajouter SVT et HG si disponibles
        if "SVT" in notes:
            data["SVT"] = [notes.get("SVT", 0)]
        if "HG" in notes:
            data["HG"] = [notes.get("HG", 0)]
        
        df = pd.DataFrame(data)
        
        return df
    
    @staticmethod
    def _select_best_models_static(models_base_path: str) -> Dict[str, str]:
        """
        Sélectionne automatiquement les meilleurs modèles pour chaque formation
        basé sur le RMSE le plus faible
        
        Args:
            models_base_path: Chemin de base vers les modèles V4
        
        Returns:
            Dictionnaire {formation: doc_version} avec les meilleurs modèles
        """
        formations = ["L1MPI", "L1BCGS", "L1PCSM"]
        docs = ["Doc1", "Doc2", "Doc3"]
        best_models = {}
        
        for formation in formations:
            best_doc = None
            best_rmse = float('inf')
            
            for doc in docs:
                info_path = os.path.join(
                    models_base_path,
                    "LassoGlobal",
                    formation,
                    doc,
                    "lasso_globale_info.pkl"
                )
                
                if os.path.exists(info_path):
                    try:
                        with open(info_path, 'rb') as f:
                            info = pickle.load(f)
                        
                        rmse = info.get('val_rmse', None)
                        if rmse is not None and rmse < best_rmse:
                            best_rmse = rmse
                            best_doc = doc
                    except Exception as e:
                        logger.warning(f"Erreur lors de la lecture de {info_path}: {e}")
            
            if best_doc:
                best_models[formation] = best_doc
                logger.info(f"Meilleur modele pour {formation}: {best_doc} (RMSE: {best_rmse:.4f})")
            else:
                # Fallback vers Doc1 si aucun modèle valide
                best_models[formation] = "Doc1"
                logger.warning(f"Aucun modele valide trouve pour {formation}, utilisation de Doc1")
        
        return best_models
    
    def _estimate_rank_from_score(self, score: float, n: int) -> int:
        """
        Estime le rang à partir du score (approximation simple)
        
        Args:
            score: Score Lasso
            n: Nombre total d'observations
        
        Returns:
            Rang estimé
        """
        if score >= 15:
            return max(1, int(n * 0.1))  # Top 10%
        elif score >= 13:
            return max(1, int(n * 0.25))  # Top 25%
        elif score >= 12:
            return max(1, int(n * 0.5))  # Médiane
        elif score >= 10:
            return max(1, int(n * 0.75))  # Bottom 25%
        else:
            return max(1, int(n * 0.9))  # Bottom 10%

