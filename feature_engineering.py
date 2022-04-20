from matplotlib.pyplot import axis
from numpy.core.numeric import full
import pandas as pd
import numpy as np
import os
from scipy.linalg import norm

import time

from csgo.analytics.stats import weapon_type

np.seterr(invalid='ignore')

def feature_extraction(path_in, path_splited):
    
    dir_list = os.listdir(path_in)
    match_counter = 0
    for folder in dir_list:
        fullpath = os.path.join(path_in, folder)
        
        match_counter += 1
        print(f"{match_counter}/{len(dir_list)}\t{fullpath}")

        try:
            # load all dataframes
            df_kills = pd.read_parquet(os.path.join(fullpath, 'kills.parquet'))
            #df_damages = pd.read_parquet(os.path.join(fullpath, 'damages.parquet'))
            #df_grenades = pd.read_parquet(os.path.join(fullpath, 'grenades.parquet'))
            #df_flashes = pd.read_parquet(os.path.join(fullpath, 'flashes.parquet'))
            #df_weaponFires = pd.read_parquet(os.path.join(fullpath, 'weaponFires.parquet'))
            #df_bombEvents = pd.read_parquet(os.path.join(fullpath, 'bombEvents.parquet'))
            df_playerFrames = pd.read_parquet(os.path.join(fullpath, 'playerFrames.parquet'))

            player_names = df_playerFrames['name'].unique()
            
            path_splited_match = os.path.join(path_splited, folder)
            if not (os.path.exists(path_splited_match)):
                os.mkdir(path_splited_match)

            player_num = 0
            for player in player_names:
                player_num += 1
                df = features_for_player(df_playerFrames, df_kills, player)
                round_num = df['roundNum'].unique()

                for r in round_num:
                    fullpath = os.path.join(path_splited_match, str(r))
                    if not (os.path.exists(fullpath)):
                        os.mkdir(fullpath)
                    
                    filename = f"Round-{r}_Player-{player_num}.parquet"
                    fullfile = os.path.join(fullpath, filename)
                    df_round = df[df['roundNum']==r].copy()
                    
                    #reset tick
                    df_round.loc[:, 'tick'] = df_round['tick'] - df_round['tick'].min()

                    df_round.to_parquet(fullfile, index=False)
        except Exception as e:
            print(e)



def features_for_player(df_playerFrame, df_kills, playername):
    features = []

    features.append(player_info(df_playerFrame, playername))
    
    features.append(kills_from_avg(df_playerFrame, df_kills, playername))
    features.append(deaths_from_avg(df_playerFrame, df_kills, playername))
    
    features.append(total_healthpoints_enemy(df_playerFrame, playername))
    features.append(total_healthpoints_team(df_playerFrame, playername))
    
    features.append(num_enemies_alive(df_playerFrame, playername))
    features.append(num_team_members_alive(df_playerFrame, playername))
    
    features.append(num_enemies_range(df_playerFrame, playername, 200))
    features.append(num_enemies_range(df_playerFrame, playername, 500))
    features.append(num_enemies_range(df_playerFrame, playername, 1000))
    features.append(num_enemies_range(df_playerFrame, playername, 2000))

    features.append(avg_healthpoints_range_enemy(df_playerFrame, playername, 500))
    features.append(avg_healthpoints_range_enemy(df_playerFrame, playername, 1000))
    features.append(avg_healthpoints_range_enemy(df_playerFrame, playername, 2000))

    features.append(avg_equipment_value_range_enemy(df_playerFrame, playername, 500))
    features.append(avg_equipment_value_range_enemy(df_playerFrame, playername, 1000))
    features.append(avg_equipment_value_range_enemy(df_playerFrame, playername, 2000))

    features.append(num_team_members_range(df_playerFrame, playername, 200))
    features.append(num_team_members_range(df_playerFrame, playername, 500))
    features.append(num_team_members_range(df_playerFrame, playername, 1000))
    
    features.append(total_equipment_value_team(df_playerFrame, playername))
    features.append(total_equipment_value_enemy(df_playerFrame, playername))

    features.append(distance_closest_enemy(df_playerFrame, playername))
    features.append(healthpoints_closest_enemy(df_playerFrame, playername))
    features.append(weapon_closest_enemy(df_playerFrame, playername))

    
    features.append(player_alive(df_playerFrame, playername))

    df_player = pd.concat(features, axis=1)

    df_player.loc[(df_player['isAlive'] == 0), 'isDead'] = 1
    df_player.loc[(df_player['isAlive'] == 1), 'isDead'] = 0
    
    return df_player



def player_alive(df_playerFrame, playername):
    df = df_playerFrame[df_playerFrame['name'] == playername].copy()
    
    df.reset_index(drop=True, inplace=True)
    
    df.loc[(df['hp'] == 0),'isAlive'] = 0
    df.loc[(df['hp'] > 0), 'isAlive'] = 1

    #df_playerFrame.drop(df_playerFrame[df_playerFrame['name']!=playername].index)
    df.drop(df.columns.difference(['isAlive']), axis=1, inplace=True)
    
    df.reset_index(drop=True, inplace=True)
    return df 



def player_info(df_playerFrame, playername):
    
    df_player = df_playerFrame.loc[df_playerFrame.name == playername].copy()

    keep_columns = ['tick', 'roundNum', 'hp', 'activeWeapon', 'armor', 'hasHelmet', 'cash', 'equipmentValue', 'isWalking', 'isBlinded', 'isAirborne', 'isDucking', 'isStanding', 'isScoped']
    #, 'isFiring'
    df_player.drop(df_player.columns.difference(keep_columns), axis=1, inplace=True)
    # categorize_weapon
    
    df_player.loc[:,('activeWeapon')] = df_player['activeWeapon'].apply(weapon_type)
    
    df_player.reset_index(drop=True, inplace=True)
    return df_player



def total_equipment_value_team(df_playerFrame, playername):

    playernames_CT, playernames_T = get_playernames_per_team(df_playerFrame)

    if playername in playernames_CT:
        playernames_team = playernames_CT
    else:
        playernames_team = playernames_T
    
    #for player in playernames_team:
    #    df_alive = player_alive(df_playerFrame, player)
    df_playerFrame.loc[df_playerFrame['hp'] == 0, 'equipmentValue'] = 0

    df = pd.DataFrame()
    df.loc[:,'equipment_value_team'] = df_playerFrame[df_playerFrame.name.isin(playernames_team)].groupby('tick').equipmentValue.aggregate('sum')
    df.reset_index(drop=True, inplace=True)
    return df



def total_equipment_value_enemy(df_playerFrame, playername):

    playernames_CT, playernames_T = get_playernames_per_team(df_playerFrame)

    if playername in playernames_CT:
        playernames_enemy = playernames_T
    else:
        playernames_enemy = playernames_CT

    df_playerFrame.loc[df_playerFrame['hp'] == 0, 'equipmentValue'] = 0
    df = pd.DataFrame()
    df.loc[:,'equipment_value_enemy'] = df_playerFrame[df_playerFrame.name.isin(playernames_enemy)].groupby('tick').equipmentValue.aggregate('sum')
    df.reset_index(drop=True, inplace=True)
    return df



def num_enemies_range(df_playerFrame, playername, distance):
    
    playernames_CT, playernames_T = get_playernames_per_team(df_playerFrame)

    if playername in playernames_CT:
        playernames_enemy = playernames_T
    else:
        playernames_enemy = playernames_CT

    # get coordinates of player
    df_playerFrame.reset_index(drop=True, inplace=True)
    df_playerFrame.loc[df_playerFrame['hp']==0, ('x','y','z')] = 1000000
    
    pos = df_playerFrame[df_playerFrame.name == playername][['x','y','z']].to_numpy()
    
    num_players_team = len(playernames_enemy)
    round_duration_ticks = len(df_playerFrame.tick.unique())
    distance_to_enemy = np.zeros((round_duration_ticks, num_players_team))

    for ii in range(num_players_team):
        name = playernames_enemy[ii]
        position_enemies = df_playerFrame[df_playerFrame.name == name][['x','y','z']].to_numpy()
              
        distance_to_enemy[:,ii] = norm(pos - position_enemies, axis=1)
    
    df = pd.DataFrame()
    in_range = distance_to_enemy < distance
    col_name = f"enemy_in_range_{distance}"
    
    df.loc[:, col_name] = np.nansum(in_range, axis=1)
    
    # if player is not alive set to 0
    df.reset_index(drop=True, inplace=True)
    df_playerFrame.reset_index(drop=True, inplace=True)

    df_alive = player_alive(df_playerFrame, playername)
    df.loc[:,col_name] = np.multiply(df.loc[:,col_name].to_numpy(), df_alive.isAlive.to_numpy())
    
    df.reset_index(drop=True, inplace=True)
    
    return df



def num_team_members_range(df_playerFrame, playername, distance):
    
    playernames_CT, playernames_T = get_playernames_per_team(df_playerFrame)

    if playername in playernames_CT:
        playernames_team = playernames_CT
    else:
        playernames_team = playernames_T

    # get coordinates of player
    df_playerFrame.reset_index(drop=True, inplace=True)
    df_playerFrame.loc[df_playerFrame['hp']==0, ('x','y','z')] = 1000000
    pos = df_playerFrame[df_playerFrame.name == playername][['x','y','z']].to_numpy()
    

    list(playernames_team).remove(playername)

    num_players_team = len(playernames_team)
    round_duration_ticks = len(df_playerFrame.tick.unique())

    distance_to_team = np.zeros((round_duration_ticks, num_players_team))

    for ii in range(num_players_team):
        name = playernames_team[ii]
        position_teammember = df_playerFrame[df_playerFrame.name == name][['x','y','z']].to_numpy()
        distance_to_team[:,ii] = norm(pos - position_teammember, axis=1)
    
    df = pd.DataFrame()
    in_range = distance_to_team < distance
    col_name = f"team_in_range_{distance}"
    df.loc[:,col_name] = np.sum(in_range, axis=1)

    # if player is not alive set to 0
    df.reset_index(drop=True, inplace=True)
    df_alive = player_alive(df_playerFrame, playername)
    df.loc[:,col_name] = np.multiply(df.loc[:,col_name].to_numpy(), df_alive.isAlive.to_numpy())
    
    df.reset_index(drop=True, inplace=True)
    return df
    


def num_players_alive(df_playerFrame, playernames):
    '''
    
    param2: list of names
    '''

    return df_playerFrame[df_playerFrame.name.isin(playernames)].groupby('tick').isAlive.aggregate('sum')



def num_team_members_alive(df_playerFrame, playername):
    """
    
    """
    playernames_CT, playernames_T = get_playernames_per_team(df_playerFrame)

    if playername in playernames_CT:
        playernames_team = playernames_CT
    else:
        playernames_team = playernames_T

    df = pd.DataFrame()
    df.loc[:,'num_team_alive'] = num_players_alive(df_playerFrame, playernames_team)
    df.reset_index(drop=True, inplace=True)
    return df



def num_enemies_alive(df_playerFrame, playername):
    """
    
    """
    playernames_CT, playernames_T = get_playernames_per_team(df_playerFrame)

    if playername in playernames_CT:
        playernames_enemy = playernames_T
    else:
        playernames_enemy = playernames_CT

    df = pd.DataFrame()
    df.loc[:,'num_enemy_alive'] = num_players_alive(df_playerFrame, playernames_enemy)
    df.reset_index(drop=True, inplace=True)
    return df



def kills_from_avg(df_playerFrame, df_kills, playername):
    """
    
    """
        
    df_frames = df_playerFrame[['tick','roundNum']].groupby('tick', as_index=False).aggregate('first')
    df_deaths = df_kills[['tick','attackerName']]

    df_deaths = pd.get_dummies(df_deaths, prefix='', prefix_sep='', columns=['attackerName'])
    df_deaths = df_deaths.groupby('tick', as_index=False).aggregate('sum')

    df_deaths = pd.merge(df_frames, df_deaths, how='left', on= ['tick'])

    df_deaths.fillna(0, inplace=True)
    df_deaths = df_deaths.cumsum()
    df_deaths.drop(['tick','roundNum'], axis=1, inplace=True)
    df_deaths = pd.concat([df_frames, df_deaths], axis=1)

    avg_deaths = df_deaths[df_deaths.columns[2:]].mean(axis=1)
    
    # z-normalisation for kills
    std_deaths = df_deaths[df_deaths.columns[2:]].std(axis=1)

    df = pd.DataFrame()
     
    df.loc[:,'kills_from_avg'] = (df_deaths[playername] - avg_deaths) / std_deaths
    df.fillna(0, inplace=True) 
    df.reset_index(drop=True, inplace=True)

    return df



def deaths_from_avg(df_playerFrame, df_kills, playername):
    
    df_frames = df_playerFrame[['tick','roundNum']].groupby('tick', as_index=False).aggregate('first')
    df_deaths = df_kills[['tick','victimName']]


    df_deaths = pd.get_dummies(df_deaths, prefix='', prefix_sep='', columns=['victimName'])
    df_deaths = df_deaths.groupby('tick', as_index=False).aggregate('sum')

    df_deaths = pd.merge(df_frames, df_deaths, how='left', on= ['tick'])

    df_deaths.fillna(0, inplace=True)
    df_deaths = df_deaths.cumsum()
    df_deaths.drop(['tick','roundNum'], axis=1, inplace=True)
    df_deaths = pd.concat([df_frames, df_deaths], axis=1)

    avg_deaths = df_deaths[df_deaths.columns[2:]].mean(axis=1)
    
    # z-normalisation for kills
    std_deaths = df_deaths[df_deaths.columns[2:]].std(axis=1)
    df = pd.DataFrame()
    
    df.loc[:,'deaths_from_avg'] = (df_deaths[playername] - avg_deaths) / std_deaths
    df.fillna(0, inplace=True) 
    df.reset_index(drop=True, inplace=True)
    return df
    

def healthpoints_total(df_playerFrame, playernames):
    '''
    
    '''
    return df_playerFrame[df_playerFrame.name.isin(playernames)].groupby('tick').hp.aggregate('sum')



def total_healthpoints_team(df_playerFrame, playername):
    """
    
    """
    
    playernames_CT, playernames_T = get_playernames_per_team(df_playerFrame)

    if playername in playernames_CT:
        playernames_team = playernames_CT
    else:
        playernames_team = playernames_T

    df = pd.DataFrame()
    df.loc[:,'total_hp_team'] = healthpoints_total(df_playerFrame, playernames_team)
    df.reset_index(drop=True, inplace=True)
    return df



def total_healthpoints_enemy(df_playerFrames, playername):
    """
    
    """
    
    playernames_CT, playernames_T = get_playernames_per_team(df_playerFrames)

    if playername in playernames_CT:
        playernames_enemy = playernames_T
    else:
        playernames_enemy = playernames_CT

    df = pd.DataFrame()
    df.loc[:,'total_hp_enemy'] = healthpoints_total(df_playerFrames, playernames_enemy)
    df.reset_index(drop=True, inplace=True)
    return df



def avg_healthpoints_range_enemy(df_playerFrames, playername, distance):
    """
    
    """
    playernames_CT, playernames_T = get_playernames_per_team(df_playerFrames)

    if playername in playernames_CT:
        playernames_enemy = playernames_T
    else:
        playernames_enemy = playernames_CT
    

    # get coordinates of player
    df_playerFrames.reset_index(drop=True, inplace=True)

    #if player is dead set positon very high so it would not be considerd when comparing for inrange
    df_playerFrames.loc[df_playerFrames['hp']==0, ('x','y','z')] = 1000000
    
    pos = df_playerFrames[df_playerFrames.name == playername][['x','y','z']].to_numpy()
    
    num_players_team = len(playernames_enemy)
    round_duration_ticks = len(df_playerFrames.tick.unique())
    distance_to_enemy = np.zeros((round_duration_ticks, num_players_team))
    healthpoints_enemies = np.zeros((round_duration_ticks, num_players_team))

    for ii in range(num_players_team):
        name = playernames_enemy[ii]
        position_enemies = df_playerFrames[df_playerFrames.name == name][['x','y','z']].to_numpy()              
        distance_to_enemy[:,ii] = norm(pos - position_enemies, axis=1)
        hp_enemy = df_playerFrames[df_playerFrames.name == name][['hp']].to_numpy()
        healthpoints_enemies[:,ii] = hp_enemy.reshape(-1)
    
    
    df = pd.DataFrame()
    in_range = distance_to_enemy < distance
    hp_mapped = np.multiply(in_range, healthpoints_enemies)

    col_name = f"enemy_hp_in_range_{distance}"
    df.loc[:, col_name] = np.divide(np.nansum(hp_mapped, axis=1), np.nansum(in_range, axis=1))

    df.fillna(0, inplace=True) 
    df.reset_index(drop=True, inplace=True)

    return df    



def avg_equipment_value_range_enemy(df_playerFrames, playername, distance):
    """
    
    """
    playernames_CT, playernames_T = get_playernames_per_team(df_playerFrames)

    if playername in playernames_CT:
        playernames_enemy = playernames_T
    else:
        playernames_enemy = playernames_CT
    

    # get coordinates of player
    df_playerFrames.reset_index(drop=True, inplace=True)
    df_playerFrames.loc[df_playerFrames['hp']==0, ('x','y','z')] = 1000000
    
    pos = df_playerFrames[df_playerFrames.name == playername][['x','y','z']].to_numpy()
    
    num_players_team = len(playernames_enemy)
    round_duration_ticks = len(df_playerFrames.tick.unique())
    distance_to_enemy = np.zeros((round_duration_ticks, num_players_team))
    healthpoints_enemies = np.zeros((round_duration_ticks, num_players_team))

    for ii in range(num_players_team):
        name = playernames_enemy[ii]
        position_enemies = df_playerFrames[df_playerFrames.name == name][['x','y','z']].to_numpy()              
        distance_to_enemy[:,ii] = norm(pos - position_enemies, axis=1)
        healthpoints_enemies[:,ii] = df_playerFrames[df_playerFrames.name == name][['equipmentValue']].to_numpy().reshape(-1)
    
    df = pd.DataFrame()
    in_range = distance_to_enemy < distance
    hp_mapped = np.multiply(in_range, healthpoints_enemies)

    col_name = f"enemy_equipment_in_range_{distance}"
    df.loc[:, col_name] = np.nansum(hp_mapped, axis=1) / np.nansum(in_range, axis=1)

    df.fillna(0, inplace=True) 
    df.reset_index(drop=True, inplace=True)

    return df



def distance_closest_enemy(df_playerFrames, playername):
    '''

    '''
    playernames_CT, playernames_T = get_playernames_per_team(df_playerFrames)

    if playername in playernames_CT:
        playernames_enemy = playernames_T
    else:
        playernames_enemy = playernames_CT


    # get coordinates of player
    df_playerFrames.reset_index(drop=True, inplace=True)
    df_playerFrames.loc[df_playerFrames['hp']==0, ('x','y','z')] = 1000000
    
    pos = df_playerFrames[df_playerFrames.name == playername][['x','y','z']].to_numpy()
    
    num_players_team = len(playernames_enemy)
    round_duration_ticks = len(df_playerFrames.tick.unique())
    distance_to_enemy = np.zeros((round_duration_ticks, num_players_team))
    
    for ii in range(num_players_team):
        name = playernames_enemy[ii]
        position_enemies = df_playerFrames[df_playerFrames.name == name][['x','y','z']].to_numpy()              
        distance_to_enemy[:,ii] = norm(pos - position_enemies, axis=1)

    df = pd.DataFrame()
    df.loc[:,'distance_closest_enemy'] = np.min(distance_to_enemy,axis=1)
    df.reset_index(drop=True, inplace=True)

    return df



def healthpoints_closest_enemy(df_playerFrames, playername):
    """
    
    """

    closest_enemy = name_closest_enemy(df_playerFrames, playername)
    
    
    hp_enemy = []

    hps = df_playerFrames['hp'].to_numpy()
    names = df_playerFrames['name'].to_numpy()
    ticks = df_playerFrames['tick'].to_numpy()
    ticks_unique = df_playerFrames['tick'].unique()

    count = 0
    for ii in range(len(hps)):        
        if count < len(closest_enemy):
            if (ticks[ii] == ticks_unique[count]) & (names[ii] == closest_enemy[count]):
                hp_enemy.append(hps[ii])
                count += 1


    df_hps = pd.DataFrame(hp_enemy, columns=['hp_closest_enemy'])
    return df_hps    



def weapon_closest_enemy(df_playerFrames, playername):

    closest_enemy = name_closest_enemy(df_playerFrames, playername)

    active_weapon = []

    weapons = df_playerFrames['activeWeapon'].to_numpy()
    names = df_playerFrames['name'].to_numpy()
    ticks = df_playerFrames['tick'].to_numpy()
    ticks_unique = df_playerFrames['tick'].unique()
    
    count = 0
    for ii in range(len(weapons)):        
        if count < len(closest_enemy):
            if (ticks[ii] == ticks_unique[count]) & (names[ii] == closest_enemy[count]):
                active_weapon.append(weapons[ii])
                count += 1
    
    df_weapons = pd.DataFrame(active_weapon, columns=['weapon_closest_enemy'])
    df_weapons.loc[:,('weapon_closest_enemy')] = df_weapons['weapon_closest_enemy'].apply(weapon_type)

    return df_weapons    



def name_closest_enemy(df_playerFrames, playername):

    playernames_CT, playernames_T = get_playernames_per_team(df_playerFrames)

    if playername in playernames_CT:
        playernames_enemy = playernames_T
    else:
        playernames_enemy = playernames_CT

    names = []

    # get coordinates of player
    df_playerFrames.reset_index(drop=True, inplace=True)
    df_playerFrames.loc[df_playerFrames['hp']==0, ('x','y','z')] = 1000000
    
    pos = df_playerFrames[df_playerFrames.name == playername][['x','y','z']].to_numpy()
    
    num_players_team = len(playernames_enemy)
    round_duration_ticks = len(df_playerFrames.tick.unique())
    distance_to_enemy = np.zeros((round_duration_ticks, num_players_team))
    
    for ii in range(num_players_team):
        name = playernames_enemy[ii]
        position_enemies = df_playerFrames[df_playerFrames.name == name][['x','y','z']].to_numpy()              
        distance_to_enemy[:,ii] = norm(pos - position_enemies, axis=1)

    names = playernames_enemy[np.argmin(distance_to_enemy,axis=1)]

    return names


def get_playernames_per_team(df):
    round_num = df['roundNum'].unique()[0]
    
    player_names_CT = df[(df.side == 'CT') & (df.roundNum == round_num)].name.unique()
    player_names_T = df[(df.side == 'T') & (df.roundNum == round_num)].name.unique()

    return player_names_CT, player_names_T



if __name__ == '__main__':
    path_raw_data = r"S:\csgodata\raw"
    path_proc_data = r"S:\csgodata\reorganized"
    path_clean_data = r"S:\csgodata\cleaned"
    path_splited_data = r"S:\csgodata\splited"
        
    feature_extraction(path_clean_data, path_splited_data)