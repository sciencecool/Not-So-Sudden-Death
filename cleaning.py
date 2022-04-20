from numpy.core.numeric import full
import pandas as pd
import os
import numpy as np
import time



def get_short_rounds(df, min_length):
    """
    
    """
     
    round_num = list(df.roundNum.unique())
    short_rounds = []
    
    for r in round_num:
        round_start = df[df.roundNum == r].tick.min()
        round_end = df[df.roundNum == r].tick.max()
        round_duration = round_end - round_start
        if round_duration < 128 * min_length:
            short_rounds.append(r)
    
    #print(f"short round: {r}")
    return short_rounds
    
    

def get_interrupted_rounds(df):
    """
    
    """
    
    # find rounds with missing frames 
    round_num = list(df.roundNum.unique())
    interrupted_rounds = []
    for r in round_num:
        diffe = df[df.roundNum == r].tick.diff()
        values = diffe.value_counts()
        
        if values.index.max() >= (128 *2):
            interrupted_rounds.append(r)
            #print(f"interrupted round: {r}")()
    
    return interrupted_rounds



def get_differnt_length_per_player_rounds(df):
    """
    
    """
    
    round_num = list(df.roundNum.unique())
    differnt_length_rounds = []
    playernames = df['name'].unique()
    
    for r in round_num:
        data_length = []
        for name in playernames:
            data_length.append(len(df[(df['name']==name) & (df['roundNum']==r)].tick.unique()))
            
        #print(f"{r}   {data_length}")
        if (min(data_length) != (max(data_length))):
            differnt_length_rounds.append(r)

    return differnt_length_rounds



def align_ticks_np(df_main, df):
    
    df_main = df_main[['tick']].groupby('tick', as_index=False).aggregate('first')
    
    if ('tick' in df.columns) and not (df.empty):        
        
        ticks_main = df_main['tick'].to_numpy()
        ticks_old = df['tick'].to_numpy()
        ticks_new = np.zeros(len(ticks_old))

        for ii in range(len(ticks_old)):
            if ii > 0:
                if (ticks_old[ii] == ticks_old[ii-1]):
                    ticks_new[ii] = ticks_new[ii-1]
                else:
                    diffe = np.abs(ticks_main - ticks_old[ii])            
                    min_tick = np.amin(diffe)
                    ind = np.where(diffe == min_tick)
                    #ind = np.argmin(diffe)
                    ticks_new[ii] = ticks_main[ind[0]]
            else:
                diffe = np.abs(ticks_main - ticks_old[ii])            
                min_tick = np.amin(diffe)
                ind = np.where(diffe == min_tick)
                    #ind = np.argmin(diffe)
                ticks_new[ii] = ticks_main[ind[0]]

    if ('destroyTick ' in df.columns) and not (df.empty):        
        
        ticks_main = df_main['tick'].to_numpy()
        ticks_old = df['tick'].to_numpy()
        ticks_new = np.zeros(len(ticks_old))

        for ii in range(len(ticks_old)):
            if ii > 0:
                if (ticks_old[ii] == ticks_old[ii-1]):
                    ticks_new[ii] = ticks_new[ii-1]
                else:
                    diffe = np.abs(ticks_main - ticks_old[ii])            
                    min_tick = np.amin(diffe)
                    ind = np.where(diffe == min_tick)
                    #ind = np.argmin(diffe)
                    ticks_new[ii] = ticks_main[ind[0]]
            else:
                diffe = np.abs(ticks_main - ticks_old[ii])            
                min_tick = np.amin(diffe)
                ind = np.where(diffe == min_tick)
                    #ind = np.argmin(diffe)
                ticks_new[ii] = ticks_main[ind[0]]

    if ('throwTick' in df.columns) and not (df.empty):        
        
        ticks_main = df_main['tick'].to_numpy()
        ticks_old = df['tick'].to_numpy()
        ticks_new = np.zeros(len(ticks_old))

        for ii in range(len(ticks_old)):
            if ii > 0:
                if (ticks_old[ii] == ticks_old[ii-1]):
                    ticks_new[ii] = ticks_new[ii-1]
                else:
                    diffe = np.abs(ticks_main - ticks_old[ii])            
                    min_tick = np.amin(diffe)
                    ind = np.where(diffe == min_tick)
                    #ind = np.argmin(diffe)
                    ticks_new[ii] = ticks_main[ind[0]]
            else:
                diffe = np.abs(ticks_main - ticks_old[ii])            
                min_tick = np.amin(diffe)
                ind = np.where(diffe == min_tick)
                    #ind = np.argmin(diffe)
                ticks_new[ii] = ticks_main[ind[0]]

        df.loc[:,'tick'] = ticks_new
    
    return df



def align_ticks(df_main, df):
    """
    
    """
    
    df_main = df_main[['tick']].groupby('tick', as_index=False).aggregate('first')
    
    if ('tick' in df.columns) and not (df.empty):        
        new_tick = []
        for old_tick in df.tick:
            
            diffe = list(abs(df_main.tick - old_tick))
            ind = diffe.index(min(diffe))
            #new_t = df_main.tick[ind]
            new_tick.append(df_main.tick[ind])    
            #print(f"{old_tick}     {min(diffe)}     {ind}  {new_t}")
        
        df.tick = new_tick
    
    if ('throwTick' in df.columns) and not (df.empty):
        new_tick = []
        for old_tick in df.throwTick:
            diffe = list(abs(df_main.tick - old_tick))
            ind = diffe.index(min(diffe))
        
            new_tick.append(df_main.tick[ind])    
            #print(f"{old_tick}     {min(diffe)}     {ind}  {new_t}")
        
        df.throwTick = new_tick

    if ('destroyTick' in df.columns) and not (df.empty):
        new_tick = []
        for old_tick in df.destroyTick:
            diffe = list(abs(df_main.tick - old_tick))
            ind = diffe.index(min(diffe))
        
            new_tick.append(df_main.tick[ind])    
            #print(f"{old_tick}     {min(diffe)}     {ind}  {new_t}")
        
        df.destroyTick = new_tick

    return df



def change_col_names(df):
    """
    
    """
    col_names = df.columns.to_list()
    new_cols = []
    for col in col_names:
        new_cols.append(col[:1].lower() + col[1:])

    df.columns = new_cols
    return df



def clean_data(path_in, path_cleaned):
    """
    
    """

    dir_list = os.listdir(path_in)
    match_counter = 0
    for folder in dir_list:
        match_counter += 1
        fullpath = os.path.join(path_in, folder)
        print(f"{match_counter}/{len(dir_list)}\t{fullpath}")

        # load all dataframes
        df_kills = pd.read_parquet(os.path.join(fullpath, 'kills.parquet'))
        df_damages = pd.read_parquet(os.path.join(fullpath, 'damages.parquet'))
        df_grenades = pd.read_parquet(os.path.join(fullpath, 'grenades.parquet'))
        df_flashes = pd.read_parquet(os.path.join(fullpath, 'flashes.parquet'))
        df_weaponFires = pd.read_parquet(os.path.join(fullpath, 'weaponFires.parquet'))
        df_bombEvents = pd.read_parquet(os.path.join(fullpath, 'bombEvents.parquet'))
        df_playerFrames = pd.read_parquet(os.path.join(fullpath, 'playerFrames.parquet'))

        # change the column names        
        df_kills = change_col_names(df_kills)
        df_damages = change_col_names(df_damages)
        df_grenades = change_col_names(df_grenades)
        df_flashes = change_col_names(df_flashes)
        df_weaponFires = change_col_names(df_weaponFires)
        df_bombEvents = change_col_names(df_bombEvents)
        df_playerFrames = change_col_names(df_playerFrames)

        # create new data frame with ticks and round number from player frames containg only one row per tick
        df_ticks = df_playerFrames[['tick','roundNum']].groupby('tick', as_index=False).aggregate('first')

        # remove fault rows from dataframes
        short_rounds = get_short_rounds(df_ticks, min_length=10)
        interrupted_rounds = get_interrupted_rounds(df_ticks)
        differnt_rounds = get_differnt_length_per_player_rounds(df_playerFrames)

        print(f"Short rounds {short_rounds}")
        print(f"Interrupted rounds {interrupted_rounds}")
        print(f"Different rounds {differnt_rounds}")

        fault_rounds = short_rounds + interrupted_rounds + differnt_rounds
        fault_rounds = list(dict.fromkeys(fault_rounds))
        fault_rounds.sort()
        print(f"Remove {len(fault_rounds)} of {len(df_ticks.roundNum.unique())} ")
      

        for r in fault_rounds:
            df_playerFrames.drop(df_playerFrames[df_playerFrames.roundNum == r].index, inplace=True)
            df_bombEvents.drop(df_bombEvents[df_bombEvents.roundNum == r].index, inplace=True)
            df_damages.drop(df_damages[df_damages.roundNum == r].index, inplace=True)
            df_flashes.drop(df_flashes[df_flashes.roundNum == r].index, inplace=True)
            df_grenades.drop(df_grenades[df_grenades.roundNum == r].index, inplace=True)
            df_weaponFires.drop(df_weaponFires[df_weaponFires.roundNum == r].index, inplace=True)
            df_kills.drop(df_kills[df_kills.roundNum == r].index, inplace=True)


        # align ticks in dataframes to ticks of df_playerFrame

        if not df_playerFrames.empty:
            df_kills = align_ticks(df_playerFrames, df_kills)
            df_bombEvents = align_ticks(df_playerFrames, df_bombEvents)
            df_damages = align_ticks(df_playerFrames, df_damages)
            df_grenades = align_ticks(df_playerFrames, df_grenades)
            df_weaponFires = align_ticks(df_playerFrames, df_weaponFires)
            df_flashes = align_ticks(df_playerFrames, df_flashes)

            # save dataframes
            fullpath_cleaned = os.path.join(path_cleaned, folder)

            if not(os.path.exists(fullpath_cleaned) and os.path.isdir(fullpath_cleaned)):
                    os.mkdir(fullpath_cleaned)

                    # save dataframes to parquet 
                    df_kills.to_parquet(os.path.join(fullpath_cleaned, 'kills.parquet'))
                    df_damages.to_parquet(os.path.join(fullpath_cleaned, 'damages.parquet'))                
                    df_grenades.to_parquet(os.path.join(fullpath_cleaned, 'grenades.parquet'))
                    df_flashes.to_parquet(os.path.join(fullpath_cleaned, 'flashes.parquet'))
                    df_weaponFires.to_parquet(os.path.join(fullpath_cleaned, 'weaponFires.parquet'))
                    df_bombEvents.to_parquet(os.path.join(fullpath_cleaned, 'bombEvents.parquet'))
                    df_playerFrames.to_parquet(os.path.join(fullpath_cleaned, 'playerFrames.parquet'))


if __name__ == '__main__':

    path_raw_data = r"S:\csgodata\raw"
    path_proc_data = r"S:\csgodata\reorganized"
    path_clean_data = r"S:\csgodata\cleaned"

    clean_data(path_proc_data, path_clean_data)


 