import numpy as np


def transform(df, columns=None, pattern_columns=None, transform_fn=None, pattern_fn=None, pattern_normalize=False):
    transformed_df = df.copy().fillna(method='bfill')

    if columns is None:
        transformed_df = transform_fn(transformed_df)
    else:
        if pattern_normalize:
            if pattern_columns is not None:
                for column in columns:
                    if column in pattern_columns:
                        transformed_df[column] = pattern_fn(transformed_df[column])
                    else:
                        transformed_df[column] = transform_fn(transformed_df[column])
            else:
                raise ValueError('pattern_columns is None')
        else:
            for column in columns:
                transformed_df[column] = transform_fn(transformed_df[column])

    return transformed_df


def max_min_normalize(df, columns=None, pattern_columns=None, pattern_normalize: bool = False):
    return transform(df=df, columns=columns, pattern_columns=pattern_columns,
                     transform_fn=lambda t_df: (t_df - t_df.min()) / (t_df.max() - t_df.min()),
                     pattern_fn=lambda t_df: (t_df - int(-100)) / (int(100) - int(-100)),
                     pattern_normalize=pattern_normalize)


def difference(df, columns=None):
    return transform(df=df, columns=columns, transform_fn=lambda t_df: t_df - t_df.shift(1))


def log_and_difference(df, columns=None):
    return transform(df=df, columns=columns, transform_fn=lambda t_df: np.log(t_df) - np.log(t_df).shift(1))
