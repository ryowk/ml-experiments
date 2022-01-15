from typing import Literal

import numpy as np
import pandas as pd
import psutil
from lightfm import LightFM
from scipy.sparse import csr_matrix, lil_matrix


class ValidationError(Exception):
    pass


class LightFMWrapper:
    """
    LightFMをpandas.DataFrameベースで使うためのラッパー
    """

    def __init__(self, user_column: str, item_column: str, target_column: str, lightfm_params: dict):
        """
        Parameters
        ----------
        user_column: user idのカラム名
        item_column: item idのカラム名
        target_column: targetのカラム名
        lightfm_params: LightFMに渡すパラメーター
        """
        self._user_column = user_column
        self._item_column = item_column
        self._target_column = target_column
        self._model = LightFM(**lightfm_params)
        self._user_features_df = None
        self._item_features_df = None

    @property
    def user_features_df(self) -> pd.DataFrame:
        return self._user_features_df

    @user_features_df.setter
    def user_features_df(self, user_features_df: pd.DataFrame):
        self._validate_user_features_df(user_features_df)
        self._user_features_df = user_features_df

    @property
    def item_features_df(self) -> pd.DataFrame:
        return self._item_features_df

    @item_features_df.setter
    def item_features_df(self, item_features_df: pd.DataFrame):
        self._validate_item_features_df(item_features_df)
        self._item_features_df = item_features_df

    @staticmethod
    def _validate_features_df(features_df: pd.DataFrame, side: Literal['user', 'item'], column_name: str):
        if column_name not in features_df.columns:
            raise ValidationError(f"'{column_name}' column is not in {side} feature dataframe")
        if len(features_df[column_name]) != len(set(features_df[column_name])):
            raise ValidationError(f"duplicated {side}s in {side} feature dataframe")
        if features_df.isnull().sum().sum() > 0:
            raise ValidationError(f"{side} feature dataframe contains null")

    def _validate_user_features_df(self, user_features_df: pd.DataFrame):
        self._validate_features_df(user_features_df, 'user', self._user_column)

    def _validate_item_features_df(self, item_features_df: pd.DataFrame):
        self._validate_features_df(item_features_df, 'item', self._item_column)

    def _validate_interactions_df(self, interactions_df: pd.DataFrame):
        for column_name in [self._user_column, self._item_column, self._target_column]:
            if column_name not in interactions_df.columns:
                raise ValidationError(f"'{column_name}' column is not in interaction dataframe")
        if len(interactions_df[[self._user_column, self._item_column]].drop_duplicates()) != len(interactions_df):
            raise ValidationError("duplicated (user, item) in interaction dataframe")
        if len(set(interactions_df[self._user_column]) - set(self._user_features_df[self._user_column])) > 0:
            raise ValidationError("some users in interaction dataframe do not exist in user feature dataframe")
        if len(set(interactions_df[self._item_column]) - set(self._item_features_df[self._item_column])) > 0:
            raise ValidationError("some items in interaction dataframe do not exist in item feature dataframe")
        if interactions_df.isnull().sum().sum() > 0:
            raise ValidationError("interaction dataframe contains null")

    def _transform_to_id_format(self, interactions_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        interaction_dfに存在するuser, itemのみに絞り、番号を振った形式に変換する

        Returns
        -------
        user_features: user特徴の2次元配列(user_idx, user_feature_idxが添え字)
        item_features: item特徴の2次元配列(item_idx, item_feature_idxが添え字)
        interaction_users: interactionのuserの1次元配列
        interaction_items: interactionのitemの1次元配列
        """
        users_df = interactions_df[[self._user_column]].sort_values(by=self._user_column).drop_duplicates(ignore_index=True)
        users_df = users_df.merge(self._user_features_df, on=self._user_column, validate='one_to_one')
        user_features = users_df.drop(self._user_column, axis=1).values

        items_df = interactions_df[[self._item_column]].sort_values(by=self._item_column).drop_duplicates(ignore_index=True)
        items_df = items_df.merge(self._item_features_df, on=self._item_column, validate='one_to_one')
        item_features = items_df.drop(self._item_column, axis=1).values

        user_to_idx = {v: i for i, v in enumerate(users_df[self._user_column].values)}
        item_to_idx = {v: i for i, v in enumerate(items_df[self._item_column].values)}

        interaction_users = interactions_df[self._user_column].apply(lambda x: user_to_idx[x]).values
        interaction_items = interactions_df[self._item_column].apply(lambda x: item_to_idx[x]).values
        return user_features, item_features, interaction_users, interaction_items

    def fit(self, interactions_df: pd.DataFrame, epochs: int, verbose: bool = False):
        """
        Parameters
        ----------
        interactions_df: user id, item id, targetを持つデータフレーム
        epcohs: LightFM.fitのepochs
        verbose: LightFM.fitのverbose
        """
        self._validate_interactions_df(interactions_df)

        user_features, item_features, interaction_users, interaction_items = self._transform_to_id_format(interactions_df)
        num_users = len(user_features)
        num_items = len(item_features)
        interaction = lil_matrix((num_users, num_items))
        interaction[interaction_users, interaction_items] = interactions_df[self._target_column].values
        self._model.fit(
            interaction,
            user_features=csr_matrix(user_features),
            item_features=csr_matrix(item_features),
            epochs=epochs,
            verbose=verbose,
            num_threads=psutil.cpu_count(logical=False),
        )

    def predict(self, interactions_df: pd.DataFrame) -> np.ndarray:
        """
        Parameters
        ----------
        interactions_df: user id, item idを持つデータフレーム

        Returns
        -------
        predictions: 推論結果
        """
        self._validate_interactions_df(interactions_df)

        user_features, item_features, interaction_users, interaction_items = self._transform_to_id_format(interactions_df)
        return self._model.predict(
            user_ids=interaction_users,
            item_ids=interaction_items,
            user_features=csr_matrix(user_features),
            item_features=csr_matrix(item_features),
            num_threads=psutil.cpu_count(logical=False),
        )


if __name__ == '__main__':
    USER_COLUMN = 'user'
    ITEM_COLUMN = 'item'
    TARGET_COLUMN = 'target'
    N_USER_FEATURES = 5
    N_ITEM_FEATURES = 10
    N_ALL_USERS = 100
    N_ALL_ITEMS = 100

    def _create_dataset(
        min_interaction_users: int,
        max_interaction_users: int,
        min_interaction_items: int,
        max_interaction_items: int,
        n_records: int
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        user_features = np.random.uniform(0.0, 1.0, (N_ALL_USERS, N_USER_FEATURES))
        user_features_df = pd.DataFrame(user_features)
        user_features_df[USER_COLUMN] = np.arange(N_ALL_USERS).astype(str)
        user_features_df[USER_COLUMN] = user_features_df[USER_COLUMN].apply(lambda x: 'user_' + x)

        item_features = np.random.uniform(0.0, 1.0, (N_ALL_ITEMS, N_ITEM_FEATURES))
        item_features_df = pd.DataFrame(item_features)
        item_features_df[ITEM_COLUMN] = np.arange(N_ALL_ITEMS).astype(str)
        item_features_df[ITEM_COLUMN] = item_features_df[ITEM_COLUMN].apply(lambda x: 'item_' + x)

        interactions_df = pd.DataFrame({
            USER_COLUMN: np.random.randint(min_interaction_users, max_interaction_users, n_records),
            ITEM_COLUMN: np.random.randint(min_interaction_items, max_interaction_items, n_records),
        }).astype(str)
        interactions_df[USER_COLUMN] = interactions_df[USER_COLUMN].apply(lambda x: 'user_' + x)
        interactions_df[ITEM_COLUMN] = interactions_df[ITEM_COLUMN].apply(lambda x: 'item_' + x)
        interactions_df = interactions_df.drop_duplicates(ignore_index=True)
        interactions_df[TARGET_COLUMN] = np.random.randint(0, 2, len(interactions_df))
        return user_features_df, item_features_df, interactions_df

    train_user_features_df, train_item_features_df, train_interactions_df = _create_dataset(0, 50, 0, 50, 100)
    valid_user_features_df, valid_item_features_df, valid_interactions_df = _create_dataset(50, 100, 50, 100, 100)

    lightfm_params = {
        'no_components': 8,
        'learning_schedule': 'adadelta',
        'loss': 'warp',
        'learning_rate': 0.001,
        'random_state': 42,
    }
    lightfm_wrapper = LightFMWrapper(USER_COLUMN, ITEM_COLUMN, TARGET_COLUMN, lightfm_params)
    lightfm_wrapper.user_features_df = train_user_features_df
    lightfm_wrapper.item_features_df = train_item_features_df
    lightfm_wrapper.fit(train_interactions_df, 100)

    lightfm_wrapper.user_features_df = valid_user_features_df
    lightfm_wrapper.item_features_df = valid_item_features_df
    lightfm_wrapper.predict(valid_interactions_df)
