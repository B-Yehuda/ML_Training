import pandas as pd
import pytest
import numpy as np


@pytest.fixture
def input_df():
    df = pd.read_pickle("sample_dataset")

    return df


@pytest.fixture
def input_config():
    config = {'Data_Processing': {'outliers_to_be_removed': {'ccv_30_d': {'threshold': 700, 'is_above': True}},
                                  'features_to_bucket': {'most_played_game': 25, 'country': 10, 'language': 10},
                                  'cols_to_drop': ['application_id', 'country', 'language', 'most_played_game'],
                                  'numeric_to_category': ['is_weekend_invite', 'is_tipping_panel',
                                                          'is_bot_command_usage', 'is_overlay', 'is_website_visit',
                                                          'is_se_live', 'is_alert_box_fired', 'is_sesp_page_visit',
                                                          'is_open_stream_report'],
                                  'is_train_val_test': "NO",
                                  'target': "'is_accept'"
                                  },
              'Dataset': {'filename': 'acceptance_rate_model_training_dataset',
                          'location': 'LOCAL'}
              }

    return config
