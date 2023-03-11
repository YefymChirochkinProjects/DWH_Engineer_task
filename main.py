import pandas as pd

test_df = pd.read_csv('data/test.csv', index_col=None)
train_df = pd.read_csv('data/train.csv', index_col=None)

transformed_df = pd.concat([train_df, test_df]).reset_index(drop=True)
transformed_df = transformed_df.rename(columns=
                                       {'Unnamed: 0': '',
                                        'feature_type_1_0': 'feature_type_1_stand_0',
                                        'feature_type_1_1': 'feature_type_1_stand_1',
                                        'feature_type_1_2': 'feature_type_1_stand_2',
                                        'feature_type_1_3': 'feature_type_1_stand_3',
                                        'feature_type_1_4': 'feature_type_1_stand_4',
                                        'feature_type_1_5': 'feature_type_1_stand_5',
                                        'feature_type_1_6': 'feature_type_1_stand_6',
                                        'feature_type_1_7': 'feature_type_1_stand_7',
                                        'feature_type_1_8': 'feature_type_1_stand_8',
                                        'feature_type_1_9': 'feature_type_1_stand_9'
                                        })

cols_to_normalize = list(transformed_df.columns[2:])

mean = transformed_df[cols_to_normalize].mean()
std = transformed_df[cols_to_normalize].std()

transformed_df[cols_to_normalize] = (transformed_df[cols_to_normalize] - mean) / std

df_for_max_value = transformed_df.iloc[:, 2:]
df_for_max_value = df_for_max_value.astype('float')
column_with_max_value = df_for_max_value.apply(lambda x: x[2:].idxmax(), axis=1)

transformed_df['max_feature_type_1_index'] = column_with_max_value.str[-1]

transformed_df.to_csv(r'test_transformed.csv', index=False)

