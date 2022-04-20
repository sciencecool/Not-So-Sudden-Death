import pandas as pd
import os
from csgo.parser import DemoParser   


def extract_from_raw_data(path_raw_data, path_proc_data):
    '''

    '''

    raw_files = os.listdir(path_raw_data)
    tecount = 0
    fnfcount = 0
    for file in raw_files:
        try:
            #print(os.path.splitext(file)[1]) 
            if os.path.splitext(file)[1] == '.dem':
                fullfile = os.path.join(path_raw_data, file)
                print(fullfile)
                parse_ok = False
                p_rate = 128

                while parse_ok==False:
                    print(f"PARSE RATE = {p_rate}")

                    demo_parser = DemoParser(demofile=fullfile, demo_id=file, parse_rate=p_rate, log=False)
                    df = demo_parser.parse(return_type="df")

                    df_kills = df['Kills']
                    df_damages = df['Damages']
                    df_grenades = df['Grenades']
                    df_flashes = df['Flashes']
                    df_weaponFires = df['WeaponFires']
                    df_bombEvents = df['BombEvents']
                    df_playerFrames = df['PlayerFrames']

                    ticks = df_playerFrames[['Tick']].groupby('Tick', as_index=False).aggregate('first')
                    tick_frames = ticks.iloc[1]['Tick'] - ticks.iloc[0]['Tick']

                    print(f"TICK FRAME {tick_frames}  PARSE RATE = {p_rate}")
                    if (tick_frames > 250) and (p_rate > 1):
                        p_rate = int(p_rate / 2)
                        print(f"PARSE AGAIN: parse rate = {p_rate}")
                    else:
                        print(f"PARSE OK")
                        parse_ok = True 


                # create new folder
                new_folder_name = os.path.splitext(df_playerFrames['MatchId'].iloc[0])[0]
                path_new_folder = os.path.join(path_proc_data, new_folder_name)
                if not(os.path.exists(path_new_folder) and os.path.isdir(path_new_folder)):
                    os.mkdir(path_new_folder)

                    # save dataframes to parquet 
                    df_kills.to_parquet(os.path.join(path_new_folder, 'kills.parquet'))
                    df_damages.to_parquet(os.path.join(path_new_folder, 'damages.parquet'))                
                    df_grenades.to_parquet(os.path.join(path_new_folder, 'grenades.parquet'))
                    df_flashes.to_parquet(os.path.join(path_new_folder, 'flashes.parquet'))
                    df_weaponFires.to_parquet(os.path.join(path_new_folder, 'weaponFires.parquet'))
                    df_bombEvents.to_parquet(os.path.join(path_new_folder, 'bombEvents.parquet'))
                    df_playerFrames.to_parquet(os.path.join(path_new_folder, 'playerFrames.parquet'))

        except (TypeError):
            print("WE GOT TypeError BUT WE PROCEED")
            tecount += 1
            print("tecount is", tecount)
            pass
        except (FileNotFoundError):
            print("WE GOT FileNotFoundError BUT WE PROCEED")
            fnfcount += 1
            print("fnfcount is", fnfcount)
            pass
    print("tecount is", tecount, "and fnfcount is", fnfcount)

if __name__ == '__main__':

    path_raw_data = r"S:\csgodata\raw"
    path_proc_data = r"S:\csgodata\reorganized"
    #path_MLready_data = r"S:\csgodata\mlready"

    extract_from_raw_data(path_raw_data, path_proc_data)


    
    