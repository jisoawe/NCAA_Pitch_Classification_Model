import pandas as pd

def load_data(file_path):
    """
    Loading in the data from the CSV
    Args: file_path (str): path to the CSV
    Returns: pd.DataFrame containing the data from the CSV
    """

    return pd.read_csv(file_path)

def clean_data(df):
    """
    Cleaning the data
    Args: df (pd.DataFrame): data frame of the inital loaded data
    Returns: pd.DataFrame: Cleaned Data
    """

    df = df.dropna()
    df = df.drop_duplicates()
    return df

def left_to_right_movement(df):
    """
    Normalizing movement profiles of LHP to be consistent with the movement profiles of RHP
    Args: df (pd.DataFrame): data frame of the cleaned data
    Returns: pd.DataFrame: normalized to RHP movement profile
    """
    df['Tilt'] = df['Tilt'] + ':00'

    tilt_mapping = {
        '11:45:00':'12:15:00','11:30:00': '12:30:00','11:15:00':'12:45:00','11:00:00':'01:00:00', 
        '10:45:00':'01:15:00', '10:30:00':'01:30:00','10:15:00':'01:45:00','10:00:00':'02:00:00',
        '09:45:00':'02:15:00','09:30:00':'02:30:00','09:15:00':'02:45:00','09:00:00':'03:00:00',
        '08:45:00':'03:15:00','08:30:00':'03:30:00','08:15:00':'03:45:00','08:00:00':'04:00:00',
        '07:45:00':'04:15:00','07:30:00':'04:30:00','07:15:00':'04:45:00','07:00:00':'05:00:00',
        '06:45:00':'05:15:00','06:30:00':'05:30:00','06:15:00':'05:45:00',
        '05:45:00':'06:15:00','05:30:00':'06:30:00','05:15:00':'06:45:00','05:00:00':'07:00:00',
        '04:45:00':'07:15:00','04:30:00':'07:30:00','04:15:00':'07:45:00','04:00:00':'08:00:00',
        '03:45:00':'08:15:00','03:30:00':'08:30:00','03:15:00':'08:45:00','03:00:00':'09:00:00',
        '02:45:00':'09:15:00','02:30:00':'09:30:00','02:15:00':'09:45:00','02:00:00':'10:00:00',
        '01:45:00':'10:15:00','01:30:00':'10:30:00','01:15:00':'10:45:00','01:00:00':'11:00:00'}
    
    df.loc[df['PitcherThrows'] == 'Left', 'Tilt'] = df.loc[df['PitcherThrows'] == 'Left', 'Tilt'].map(tilt_mapping)
        
    adjust_cols = ['HorzBreak', 'RelSide', 'HorzRelAngle', 'HorzApprAngle', 'PitchTrajectoryZc0', 'PitchTrajectoryZc1', 'PitchTrajectoryZc2']
    for col in adjust_cols:    
        if col in df.columns:
            df.loc[df['PitcherThrows'] == 'Left', col] = df.loc[df['PitcherThrows'] == 'Left', col] *-1
            
    df = df.drop('PitcherThrows', axis=1)
    return df

def pitch_type_adjust(df):
    """
    Adjusting pitch types (Classifying Sinkers as Fastballs & removing uncommon pitch types)
    Args: df (pd.DataFrame): data frame of the normalized data
    Returns: pd.DataFrame: Cleaned Data
    """

    df['TaggedPitchType'] = df['TaggedPitchType'].replace('Sinker', 'Fastball')
        
    nope = ['Other', 'Splitter', 'Undefined', 'Knuckleball']
    df[~df['TaggedPitchType'].isin(nope)]
        
    return df

def tilt_to_minutes(df):
    """
    Transforming the 'Tilt' variable from hours to minutes (optimized for modeling)
    Args: df (pd.DataFrame): data frame of the cleaned data
    Returns: pd.DataFrame: Transformed Data with tilt minutes var instead of tilt
    """
    df['Tilt_minutes'] = (pd.to_timedelta(df['Tilt']).dt.total_seconds())/60
    df = df.drop('Tilt', axis = 1)
    return df

def transform_data(df):
    """
    Transforming the data
    Args: df (pd.DataFrame): data frame of the cleaned data
    Returns: pd.DataFrame: Transformed Data optimized for modeling
    """
    cols = ['PitcherThrows', 'RelSpeed', 'VertRelAngle', 'HorzRelAngle', 'SpinRate', 'SpinAxis',
            'RelHeight', 'RelSide', 'Extension', 'VertBreak', 'InducedVertBreak', 'HorzBreak', 'ZoneSpeed',
           'VertApprAngle', 'HorzApprAngle', 'PitchTrajectoryXc0','PitchTrajectoryXc1', 'PitchTrajectoryXc2',
            'PitchTrajectoryYc0', 'PitchTrajectoryYc1', 'PitchTrajectoryYc2', 'PitchTrajectoryZc0',
            'PitchTrajectoryZc1','PitchTrajectoryZc2', 'TaggedPitchType', 'Tilt']
    
    df = df[cols]
    
    df = left_to_right_movement(df)
    df = pitch_type_adjust(df)
    df = tilt_to_minutes(df)
    return df


def save_data(df, output_path):
    """
    Saving the transformed data to a CSV
    Args: 
    df (pd.DataFrame): data frame of the transformed data
    output_path (str): file path to save the CSV
    """
    df.to_csv(output_path, index = False)