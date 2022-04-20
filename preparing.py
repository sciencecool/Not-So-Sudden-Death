import os
import pandas as pd
import random
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from joblib import dump



def prepare_datasets_for_ml(path_windowed_data, path_mlready_data, train_split=0.6, window_length=3, steps_forecast=2):
    """
    
    """
    
    steps = steps_forecast
    folder = f"winSize-{window_length}_forecSteps-{steps_forecast}"

    path_matches = os.path.join(path_windowed_data, folder)
    matches = set(os.listdir(path_matches))
    
    random.seed(4)
    train_matches = set(random.sample(matches, round(len(matches)*train_split)))
    remaining_matches = matches - train_matches
    val_matches = set(random.sample(remaining_matches, round(len(remaining_matches)*0.5)))
    remaining_matches = remaining_matches - val_matches
    test_matches = remaining_matches

    
    df = pd.DataFrame()
    # load matches for MinMaxScaler

    MAX_MATCHES = 50
    if len(train_matches) > MAX_MATCHES:
        scaler_matches = random.sample(train_matches, MAX_MATCHES)
    else:
        scaler_matches = train_matches
    
    print("FIT SCALER")
    df_list = []
    for match in scaler_matches:
        path_match = os.path.join(path_matches, match)
        rounds = os.listdir(path_match)

        for r in rounds:
            path_round = os.path.join(path_match, r)
            files = os.listdir(path_round)
            for file in files:
                fullfile = os.path.join(path_round, file)
                
                df_new = pd.read_parquet(fullfile)
                df_new = create_target_var(df_new, steps)
                df_new = remove_cols(df_new)
                
                df_list.append(df_new)

                
    df = pd.concat(df_list, ignore_index=True)
    scaler = MinMaxScaler()
    scaler.fit(df.to_numpy())

    # make new folders for train, test and validation datasets

    path_mlready_data = os.path.join(path_mlready_data, folder)
    if not os.path.exists(path_mlready_data):
        os.mkdir(path_mlready_data)

    path_mlready_train_data = os.path.join(path_mlready_data, 'train')
    if not os.path.exists(path_mlready_train_data):
        os.mkdir(path_mlready_train_data)

    path_mlready_val_data = os.path.join(path_mlready_data, 'val')
    if not os.path.exists(path_mlready_val_data):
        os.mkdir(path_mlready_val_data)

    path_mlready_test_data = os.path.join(path_mlready_data, 'test')
    if not os.path.exists(path_mlready_test_data):
        os.mkdir(path_mlready_test_data)


    print("TRAIN MATCHES")
    match_counter = 0
    for match in train_matches:
        path_match = os.path.join(path_matches, match)
        
        match_counter += 1
        print(f"{match_counter}/{len(train_matches)}\t{path_match}")        
        
        rounds = os.listdir(path_match)
        
        path_mlready_match = os.path.join(path_mlready_train_data, match)
        if not os.path.exists(path_mlready_match):
            os.mkdir(path_mlready_match)

        for r in rounds:
            path_round = os.path.join(path_match, r)
            files = os.listdir(path_round)

            path_mlready_round = os.path.join(path_mlready_match, r)
            if not os.path.exists(path_mlready_round):
                os.mkdir(path_mlready_round)


            for file in files:
                fullfile = os.path.join(path_round, file)
                df = pd.read_parquet(fullfile)
                df = create_target_var(df, steps)
                df = remove_cols(df)

                df.loc[:,:] = scaler.transform(df.to_numpy())
                
                fullfile = os.path.join(path_mlready_round, file)
                df.to_parquet(fullfile) 


    print("VALIDATION MATCHES")
    match_counter = 0
    for match in val_matches:

        path_match = os.path.join(path_matches, match)

        match_counter += 1
        print(f"{match_counter}/{len(val_matches)}\t{path_match}")

        rounds = os.listdir(path_match)

        ### make new folder
        path_mlready_match = os.path.join(path_mlready_val_data, match)
        if not os.path.exists(path_mlready_match):
            os.mkdir(path_mlready_match)

        for r in rounds:
            path_round = os.path.join(path_match, r)
            files = os.listdir(path_round)

            path_mlready_round = os.path.join(path_mlready_match, r)
            if not os.path.exists(path_mlready_round):
                os.mkdir(path_mlready_round)


            for file in files:
                fullfile = os.path.join(path_round, file)
                df = pd.read_parquet(fullfile)
                df = create_target_var(df, steps)
                df = remove_cols(df)

                df.loc[:,:] = scaler.transform(df.to_numpy())

                ##### set path
                fullfile = os.path.join(path_mlready_round, file)
                df.to_parquet(fullfile)

    
    print("TEST MATCHES")
    match_counter = 0
    for match in test_matches:
        path_match = os.path.join(path_matches, match)

        match_counter += 1
        print(f"{match_counter}/{len(val_matches)}\t{path_match}")

        rounds = os.listdir(path_match)

        path_mlready_match = os.path.join(path_mlready_test_data, match)
        if not os.path.exists(path_mlready_match):
            os.mkdir(path_mlready_match)

        for r in rounds:
            path_round = os.path.join(path_match, r)
            files = os.listdir(path_round)
            
            path_mlready_round = os.path.join(path_mlready_match, r)
            if not os.path.exists(path_mlready_round):
                os.mkdir(path_mlready_round)
            
            for file in files:
                fullfile = os.path.join(path_round, file)
                df = pd.read_parquet(fullfile)
                df = create_target_var(df, steps)
                df = remove_cols(df)

                df.loc[:,:] = scaler.transform(df.to_numpy())

                fullfile = os.path.join(path_mlready_round, file)
                df.to_parquet(fullfile)



def create_target_var(df, steps_forecast):
    
    var_name = f"forecast_dead-{steps_forecast}"
    
    forecast_var = f"isDead (t+{steps_forecast})"
    
    df.loc[:, var_name] = df['isAlive (t-0)'] * df[forecast_var]
    return df



def remove_cols(df):
    """
    
    """
    
    round_num_cols = [col for col in df.columns if 'roundNum' in col]
    df.drop(round_num_cols, axis=1, inplace=True)

    tick_cols = [col for col in df.columns if 'tick' in col]
    df.drop(tick_cols, axis=1, inplace=True)

    is_dead_cols = [col for col in df.columns if 'isDead' in col]    
    df.drop(is_dead_cols, axis=1, inplace=True)

    alive_cols = [col for col in df.columns if 'isAlive' in col]
    df.drop(alive_cols, axis=1, inplace=True)    
    
    
    return df



def prepare_data(path_splited_data, path_windowed_data, window_length, steps_forcast, path_models):
    """
    
    """
    
    match_list = os.listdir(path_splited_data)
    new_folder_name = f"winSize-{window_length}_forecSteps-{steps_forcast}"

    path_winsize_windowed = os.path.join(path_windowed_data, new_folder_name)
    if not os.path.exists(path_winsize_windowed):
        os.mkdir(path_winsize_windowed)

    match_counter = 0
    for folder in match_list:
        path_match = os.path.join(path_splited_data, folder)
        
        match_counter += 1        
        print(f"{match_counter}/{len(match_list)}\t{path_match}")
        
        rounds_list = os.listdir(path_match)
        
        path_match_windowed = os.path.join(path_winsize_windowed, folder)
        if not os.path.exists(path_match_windowed):
            os.mkdir(path_match_windowed)        

        for r in rounds_list:
            path_round = os.path.join(path_match, r)
            file_list = os.listdir(path_round)
            
            path_round_windowed = os.path.join(path_match_windowed, r)
            if not os.path.exists(path_round_windowed):
                os.mkdir(path_round_windowed)

            for file in file_list:
                
                fullfile = os.path.join(path_round, file)
                #print(f"IN:  {fullfile}")
                data = pd.read_parquet(fullfile)
                try:
                    data = encode_weapon(data, save_encoder=True, path=path_models)
                    data = encode_bool(data)
                    df = window_timeseries(data, window_length=window_length, steps_forcast=steps_forcast, forecast_target=['isDead'], forecast_sequence=False, dropnan=True)
                    
                
                    fullfile_out = os.path.join(path_round_windowed, file)
                    #print(f"OUT: {fullfile_out}")
                    df.to_parquet(fullfile_out)
                except Exception as e:
                    print(e)



def encode_weapon(df, save_encoder=False, path=None):
    """
    
    """
    weapon_types = ["Melee Kills", "Pistol Kills", "Shotgun Kills", "SMG Kills", "Assault Rifle Kills", "Machine Gun Kills", "Sniper Rifle Kills", "Utility Kills"]

    if ('activeWeapon' not in df.columns) & ('weapon' not in df.columns) & ('weapon_closest_enemy' not in df.columns): 
        print('no weapon column')
        return df

    
    if 'activeWeapon' in df.columns:
        weapon_encoder = OneHotEncoder(handle_unknown='ignore')
        weapon_encoder.fit(pd.DataFrame({"activeWeapon": weapon_types}))

        active_weapon = weapon_encoder.transform(df[['activeWeapon']]).toarray()
        col_names = ['active_weapon_'+col_name for col_name in weapon_types]
        df_weapon = pd.DataFrame(active_weapon, columns=col_names)

        df = pd.concat([df, df_weapon], axis=1)
        df.drop('activeWeapon',axis=1, inplace=True)

        if save_encoder == True:
            if path != None:
                if os.path.exists(path):
                    encoder_file = os.path.join(path, 'weapon_encoder.joblib')
                    if not os.path.exists(encoder_file):
                        dump(weapon_encoder, encoder_file)
                    
                else:
                    print("Not a valid path for One Hot Encoder")

    
    if 'weapon' in df.columns:
        weapon_encoder = OneHotEncoder(handle_unknown='ignore')
        weapon_encoder.fit(pd.DataFrame({"weapon": weapon_types}))

        active_weapon = weapon_encoder.transform(df[['weapon']]).toarray()
        col_names = ['active_weapon_'+col_name for col_name in weapon_types]
        df_weapon = pd.DataFrame(active_weapon, columns=col_names)

        df = pd.concat([df, df_weapon], axis=1)
        df.drop('weapon',axis=1, inplace=True)

    if 'weapon_closest_enemy' in df.columns:
        weapon_encoder = OneHotEncoder(handle_unknown='ignore')
        weapon_encoder.fit(pd.DataFrame({"weapon_closest_enemy": weapon_types}))

        active_weapon = weapon_encoder.transform(df[['weapon_closest_enemy']]).toarray()
        col_names = ['weapon_closest_enemy_'+col_name for col_name in weapon_types]
        df_weapon = pd.DataFrame(active_weapon, columns=col_names)

        df = pd.concat([df, df_weapon], axis=1)
        df.drop('weapon_closest_enemy',axis=1, inplace=True)

        if save_encoder == True:
            if path != None:
                if os.path.exists(path):
                    encoder_file = os.path.join(path, 'weapon_encoder_enemy.joblib')
                    if not os.path.exists(encoder_file):
                        dump(weapon_encoder, encoder_file)
                else:
                    print("Not a valid path for One Hot Encoder")
      
    
    return df



def encode_bool(df):
    """
    
    """
    df.replace({False: 0, True: 1}, inplace=True)
    return df



def window_timeseries(data, window_length=1, steps_forcast=1, forecast_target=[None], forecast_sequence=False, dropnan=True):
    """
    
    """    
    cols, new_col_names = list(), list()
    col_names = data.columns

    # input sequence (t-n, ... t-1, t)
    for ii in range((window_length-1), -1, -1):
        cols.append(data.shift(ii))
        new_col_names += [(f'{col} (t-{ii})') for col in col_names]
        
    # forecast sequence (t+1, ... t+n)
    if forecast_sequence == True:
        for ii in range(1, steps_forcast+1):
        
            if forecast_target != None:
                cols.append(data[forecast_target].shift(-ii))
                new_col_names += [(f'{col} (t+{ii})') for col in forecast_target]
            
            else:
                cols.append(data.shift(-ii))
                if ii == 0:
                    new_col_names += [(f'{col} (t)') for col in col_names]
                else:
                    new_col_names += [(f'{col} (t+{ii})') for col in col_names]
    # forecast step (t+n)
    else:
        if forecast_target != None:
            cols.append(data[forecast_target].shift(-steps_forcast))
            new_col_names += [(f'{col} (t+{steps_forcast})') for col in forecast_target]
        else:
            cols.append(data.shift(-steps_forcast))
            new_col_names += [(f'{col} (t+{steps_forcast})') for col in col_names]
    
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = new_col_names

    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    
    return agg