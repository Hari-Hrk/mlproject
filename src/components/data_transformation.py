# this file for data transformation like categorical to numerical and encoding data etc

import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_objects(self):
        '''
            This funciton responsible for the data transformation
        '''
        try:
            numerical_columns = ["writing_score","reading_score"]
            categorical_columns = [
                "gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"
            ]

            num_pipeline = Pipeline(steps=[("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())])
            cat_pipeline = Pipeline(steps=[("imputer",SimpleImputer(strategy="most_frequent")),
            ("one_hot_encoder",OneHotEncoder(sparse_output=True)),("scaler",StandardScaler(with_mean=False))
            ])
            logging.info(f"Numerical columns: {categorical_columns}")
            logging.info(f"Categorical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                [("num_pipeline",num_pipeline,numerical_columns),("cat_pipelines",cat_pipeline,categorical_columns)]
            )

            return preprocessor

        except Exception as e :
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")
            preprocesiing_obj = self.get_data_transformer_objects()
            target_column_name = "math_score"
            numerical_columns = ["writing_score","reading_score"]

            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            input_feature_train_arr= preprocesiing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocesiing_obj.transform(input_feature_test_df)
            # Verify dimensions
            # if input_feature_train_arr.shape[0] != target_feature_train_df.shape[0]:
            #     raise ValueError("Mismatch in training data dimensions.")
            # if input_feature_test_arr.shape[0] != target_feature_test_df.shape[0]:
            #     raise ValueError("Mismatch in testing data dimensions.")


            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            # print(f"Shape of input_feature_train_arr: {input_feature_train_arr.shape}")
            # logging.info(f"Shape of target_feature_train_df: {target_feature_train_df.shape}")
            # logging.info(f"Shape of input_feature_test_arr: {input_feature_test_arr.shape}")
            # logging.info(f"Shape of target_feature_test_df: {target_feature_test_df.shape}")


            logging.info(f"Saved preprocessing objects.")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocesiing_obj
            )

            return(
                train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)