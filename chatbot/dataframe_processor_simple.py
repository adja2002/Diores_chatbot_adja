"""
Version simplifiée de DataFrameProcessor qui ne dépend pas de xgboost
Utilisée pour le traitement des données pour les modèles DIORES V4
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import logging

logger = logging.getLogger(__name__)


class DataFrameProcessorSimple:
    """
    Processeur de DataFrame simplifié pour les modèles DIORES
    Ne nécessite pas xgboost
    """
    
    def __init__(self, df):
        self.df = df.copy()
        self.features = [
            'Année BAC', 'Nbre Fois au BAC', 'Groupe Résultat', 'Moy. nde',
            'Moy. ère', 'Moy. S Term.', 'Moy. S Term..1', 'MATH', 'SCPH', 'FR',
            'PHILO', 'AN', 'Tot. Pts au Grp.', 'Moyenne au Grp.', 'Moy. Gle',
            'Moy. sur Mat.Fond.', 'Age en Décembre 2018', 'Sexe_F', 'Sexe_M',
            'Série_S1', 'Série_S2', 'Série_S3', 'Mention_ABien', 'Mention_Bien',
            'Mention_Pass', 'Résidence', 'Ets. de provenance', 'Centre d\'Ec.',
            'Académie de l\'Ets. Prov.', 'REGION_DE_NAISSANCE', 'Academie perf.'
        ]
    
    def encode_categorical_features(self, categorical_columns):
        self.df = pd.get_dummies(self.df, columns=categorical_columns, prefix=categorical_columns)
        return self
    
    def label_encode_columns(self, columns):
        """
        Encode les colonnes catégorielles avec LabelEncoder
        et renomme les colonnes avec le suffixe '_Encode' pour correspondre
        aux modèles V4
        """
        le = LabelEncoder()
        for col in columns:
            if col in self.df.columns:
                try:
                    # Encoder la colonne
                    self.df[col] = le.fit_transform(self.df[col].astype(str))
                    # Renommer avec le suffixe '_Encode'
                    new_col_name = f"{col}_Encode"
                    self.df = self.df.rename(columns={col: new_col_name})
                except Exception as e:
                    # Si l'encodage échoue, créer une colonne avec _Encode à 0
                    new_col_name = f"{col}_Encode"
                    if new_col_name not in self.df.columns:
                        self.df[new_col_name] = 0
        return self
    
    def calculate_academie_performance(self):
        if "Académie de l'Ets. Prov." in self.df.columns and 'Moy. Gle' in self.df.columns:
            def get_academie_moyenne():
                return pd.Series(
                    self.df['Moy. Gle'].values,
                    index=self.df["Académie de l'Ets. Prov."]
                ).to_dict()
            
            dic = get_academie_moyenne()
            self.df['Academie perf.'] = self.df.apply(
                lambda row: dic.get(row["Académie de l'Ets. Prov."], row['Moy. Gle']),
                axis=1
            )
        return self
    
    def convert_to_numeric(self):
        non_numeric_cols = self.df.select_dtypes(include=['object']).columns
        for col in non_numeric_cols:
            try:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            except:
                pass
        return self
    
    def convert_columns_to_int(self, columns):
        existing_cols = [col for col in columns if col in self.df.columns]
        if existing_cols:
            self.df[existing_cols] = self.df[existing_cols].astype(int)
        return self
    
    def clean_data(self):
        if 'MOYENNE ANNUELLE' in self.df.columns:
            self.df = self.df.drop(['MOYENNE ANNUELLE'], axis=1)
        self.df.fillna(0, inplace=True)
        return self
    
    def ensure_features(self):
        for feature in self.features:
            if feature not in self.df.columns:
                self.df[feature] = 0
        return self
    
    def process_all(self):
        try:
            # Suppression des doublons
            self.df = self.df.loc[:, ~self.df.columns.duplicated()]
            
            # Préparation des colonnes
            dummy_columns = ['Sexe_F', 'Sexe_M', 'Série_S1', 'Série_S2', 'Série_S3',
                            'Mention_ABien', 'Mention_Bien', 'Mention_Pass']
            for col in dummy_columns:
                if col not in self.df.columns:
                    self.df[col] = 0
            
            # Encodage catégoriel
            if 'Mention' in self.df.columns:
                self.df = pd.get_dummies(self.df, columns=['Mention'], prefix='Mention')
            if 'Sexe' in self.df.columns:
                self.df = pd.get_dummies(self.df, columns=['Sexe'], prefix='Sexe')
            if 'Série' in self.df.columns:
                self.df = pd.get_dummies(self.df, columns=['Série'], prefix='Série')
            
            # Encoder Série et Sexe avec LabelEncoder (pas get_dummies)
            # Car les modèles V4 attendent Série_Encode et Sexe_Encode
            if 'Série' in self.df.columns:
                le_serie = LabelEncoder()
                self.df['Série'] = le_serie.fit_transform(self.df['Série'].astype(str))
                self.df = self.df.rename(columns={'Série': 'Série_Encode'})
            elif 'Série_Encode' not in self.df.columns:
                # Créer Série_Encode à partir des colonnes dummy
                if 'Série_S1' in self.df.columns:
                    self.df['Série_Encode'] = (
                        self.df['Série_S1'] * 0 + 
                        self.df.get('Série_S2', 0) * 1 + 
                        self.df.get('Série_S3', 0) * 2
                    )
                else:
                    self.df['Série_Encode'] = 0
            
            if 'Sexe' in self.df.columns:
                le_sexe = LabelEncoder()
                self.df['Sexe'] = le_sexe.fit_transform(self.df['Sexe'].astype(str))
                self.df = self.df.rename(columns={'Sexe': 'Sexe_Encode'})
            elif 'Sexe_Encode' not in self.df.columns:
                # Créer Sexe_Encode à partir des colonnes dummy
                if 'Sexe_M' in self.df.columns:
                    self.df['Sexe_Encode'] = self.df['Sexe_M']
                else:
                    self.df['Sexe_Encode'] = 0
            
            # Supprimer les colonnes dummy qui ne sont plus nécessaires
            dummy_cols_to_remove = ['Sexe_F', 'Sexe_M', 'Série_S1', 'Série_S2', 'Série_S3',
                                   'Mention_ABien', 'Mention_Bien', 'Mention_Pass']
            for col in dummy_cols_to_remove:
                if col in self.df.columns:
                    self.df = self.df.drop(columns=[col])
            
            # Autres traitements
            label_cols = ['Résidence', "Ets. de provenance", "Centre d'Ec.",
                          "Académie de l'Ets. Prov.", "REGION_DE_NAISSANCE"]
            self.label_encode_columns(label_cols)
            self.calculate_academie_performance()
            self.convert_to_numeric()
            self.clean_data()
            
            # Vérification finale - les features attendues par les modèles V4
            expected_features = [
                'Année BAC', 'Nbre Fois au BAC', 'Groupe Résultat', 'Moy. nde',
                'Moy. ère', 'Moy. S Term.', 'Moy. S Term..1', 'MATH', 'SCPH', 'FR',
                'PHILO', 'AN', 'Tot. Pts au Grp.', 'Moyenne au Grp.', 'Moy. Gle',
                'Moy. sur Mat.Fond.', 'Age en Décembre 2018', 'Série_Encode', 'Sexe_Encode',
                'Academie perf.', 'Résidence_Encode', 'Ets. de provenance_Encode',
                "Centre d'Ec._Encode", "Académie de l'Ets. Prov._Encode", 'REGION_DE_NAISSANCE_Encode'
            ]
            
            missing_features = set(expected_features) - set(self.df.columns)
            if missing_features:
                for feature in missing_features:
                    self.df[feature] = 0
            
            # Retourner uniquement les features attendues par les modèles
            return self.df[expected_features]
        
        except Exception as e:
            logger.error(f"Erreur dans process_all: {str(e)}")
            raise

