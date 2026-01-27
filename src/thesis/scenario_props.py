def scenario_prop():
    row_column_users = {
    'city_0_newyork': {
        'n_rows': 109,
        'n_per_row': 291,
        'n_ant_bs': 8,
        'n_subcarriers': 32
    },
    'city_1_losangeles': {
        'n_rows': 142,
        'n_per_row': 201,
        'n_ant_bs': 8,
        'n_subcarriers': 64
    },
    'city_2_chicago': {
        'n_rows': 139,
        'n_per_row': 200,
        'n_ant_bs': 8,
        'n_subcarriers': 128
    },
    'city_3_houston': {
        'n_rows': 154,
        'n_per_row': 202,
        'n_ant_bs': 8,
        'n_subcarriers': 256
    },
    'city_4_phoenix': {
        'n_rows': 198,
        'n_per_row': 214,
        'n_ant_bs': 8,
        'n_subcarriers': 512
    },
    'city_5_philadelphia': {
        'n_rows': 239,
        'n_per_row': 164,
        'n_ant_bs': 8,
        'n_subcarriers': 1024
    },
    'city_6_miami': {
        'n_rows': 199,
        'n_per_row': 216 ,
        'n_ant_bs': 16,
        'n_subcarriers': 32
    },
    'city_7_sandiego': {
        'n_rows': 71,
        'n_per_row': 176,
        'n_ant_bs': 16,
        'n_subcarriers': 64
    },
    'city_8_dallas': {
        'n_rows': 207,
        'n_per_row': 190,
        'n_ant_bs': 16,
        'n_subcarriers': 128
    },
    'city_9_sanfrancisco': {
        'n_rows': 196,
        'n_per_row': 206,
        'n_ant_bs': 16,
        'n_subcarriers': 256
    },
    'city_10_austin': {
        'n_rows': 255,
        'n_per_row': 137,
        'n_ant_bs': 16,
        'n_subcarriers': 512
    },
    'city_11_santaclara': {
        'n_rows': 46,
        'n_per_row': 285,
        'n_ant_bs': 32,
        'n_subcarriers': 32
    },
    'city_12_fortworth': {
        'n_rows': 85,
        'n_per_row': 179,
        'n_ant_bs': 32,
        'n_subcarriers': 64
    },
    'city_13_columbus': {
        'n_rows': 178,
        'n_per_row': 240,
        'n_ant_bs': 32,
        'n_subcarriers': 128
    },
    'city_14_charlotte': {
        'n_rows': 216,
        'n_per_row': 177,
        'n_ant_bs': 32,
        'n_subcarriers': 256
    },
    'city_15_indianapolis': {
        'n_rows': 79,
        'n_per_row': 196,
        'n_ant_bs': 64,
        'n_subcarriers': 32
    },
    'city_16_sanfrancisco': {
        'n_rows': 201,
        'n_per_row': 208,
        'n_ant_bs': 64,
        'n_subcarriers': 64
    },
    'city_17_seattle': {
        'n_rows': 185,
        'n_per_row': 205,
        'n_ant_bs': 64,
        'n_subcarriers': 128
    },
    'city_18_denver': {
        'n_rows': 84,
        'n_per_row': 204,
        'n_ant_bs': 128,
        'n_subcarriers': 32
    },
    'city_19_oklahoma': {
        'n_rows': 81,
        'n_per_row': 188,
        'n_ant_bs': 128,
        'n_subcarriers': 64
    },
    'asu_campus1_v1': {
        'n_rows': [0, 1*int(321/20)],
        'n_per_row': 411,
        'n_ant_bs': 8,
        'n_subcarriers': 32
    },
    'asu_campus1_v2': {
        'n_rows': [1*int(321/20), 2*int(321/20)],
        'n_per_row': 411,
        'n_ant_bs': 8,
        'n_subcarriers': 64
    },
    'asu_campus1_v3': {
        'n_rows': [2*int(321/20), 3*int(321/20)],
        'n_per_row': 411,
        'n_ant_bs': 8,
        'n_subcarriers': 128
    },
    'asu_campus1_v4': {
        'n_rows': [3*int(321/20), 4*int(321/20)],
        'n_per_row': 411,
        'n_ant_bs': 8,
        'n_subcarriers': 256
    },
    'asu_campus1_v5': {
        'n_rows': [4*int(321/20), 5*int(321/20)],
        'n_per_row': 411,
        'n_ant_bs': 8,
        'n_subcarriers': 512
    },
    'asu_campus1_v6': {
        'n_rows': [5*int(321/20), 6*int(321/20)],
        'n_per_row': 411,
        'n_ant_bs': 8,
        'n_subcarriers': 1024
    },
    'asu_campus1_v7': {
        'n_rows': [6*int(321/20), 7*int(321/20)],
        'n_per_row': 411,
        'n_ant_bs': 16,
        'n_subcarriers': 32
    },
    'asu_campus1_v8': {
        'n_rows': [7*int(321/20), 8*int(321/20)],
        'n_per_row': 411,
        'n_ant_bs':16,
        'n_subcarriers': 64
    },
    'asu_campus1_v9': {
        'n_rows': [8*int(321/20), 9*int(321/20)],
        'n_per_row': 411,
        'n_ant_bs': 16,
        'n_subcarriers': 128
    },
    'asu_campus1_v10': {
        'n_rows': [9*int(321/20), 10*int(321/20)],
        'n_per_row': 411,
        'n_ant_bs': 16,
        'n_subcarriers': 256
    },
    'asu_campus1_v11': {
        'n_rows': [10*int(321/20), 11*int(321/20)],
        'n_per_row': 411,
        'n_ant_bs': 16,
        'n_subcarriers': 512
    },
    'asu_campus1_v12': {
        'n_rows': [11*int(321/20), 12*int(321/20)],
        'n_per_row': 411,
        'n_ant_bs': 32,
        'n_subcarriers': 32
    },
    'asu_campus1_v13': {
        'n_rows': [12*int(321/20), 13*int(321/20)],
        'n_per_row': 411,
        'n_ant_bs': 32,
        'n_subcarriers': 64
    },
    'asu_campus1_v14': {
        'n_rows': [13*int(321/20), 14*int(321/20)],
        'n_per_row': 411,
        'n_ant_bs': 32,
        'n_subcarriers': 128
    },
    'asu_campus1_v15': {
        'n_rows': [14*int(321/20), 15*int(321/20)],
        'n_per_row': 411,
        'n_ant_bs': 32,
        'n_subcarriers': 256
    },
    'asu_campus1_v16': {
        'n_rows': [15*int(321/20), 16*int(321/20)],
        'n_per_row': 411,
        'n_ant_bs': 64,
        'n_subcarriers': 32
    },
    'asu_campus1_v17': {
        'n_rows': [16*int(321/20), 17*int(321/20)],
        'n_per_row': 411,
        'n_ant_bs': 64,
        'n_subcarriers': 64 
    },
    'asu_campus1_v18': {
        'n_rows': [17*int(321/20), 18*int(321/20)],
        'n_per_row': 411,
        'n_ant_bs': 64,
        'n_subcarriers': 128
    },
    'asu_campus1_v19': {
        'n_rows': [18*int(321/20), 19*int(321/20)],
        'n_per_row': 411,
        'n_ant_bs': 128,
        'n_subcarriers': 32
    },
    'asu_campus1_v20': {
        'n_rows': [19*int(321/20), 20*int(321/20)],
        'n_per_row': 411,
        'n_ant_bs': 128,
        'n_subcarriers': 64
    },
    'Boston5G_3p5_v1': {
        'n_rows': [812, 812 + 1*int((1622-812)/20)],
        'n_per_row': 595,
        'n_ant_bs': 8,
        'n_subcarriers': 32
    },
    'Boston5G_3p5_v2': {
        'n_rows': [812 + 1*int((1622-812)/20), 812 + 2*int((1622-812)/20)],
        'n_per_row': 595,
        'n_ant_bs': 8,
        'n_subcarriers': 64
    },
    'Boston5G_3p5_v3': {
        'n_rows': [812 + 2*int((1622-812)/20), 812 + 3*int((1622-812)/20)],
        'n_per_row': 595,
        'n_ant_bs': 8,
        'n_subcarriers': 128
    },
    'Boston5G_3p5_v4': {
        'n_rows': [812 + 3*int((1622-812)/20), 812 + 4*int((1622-812)/20)],
        'n_per_row': 595,
        'n_ant_bs': 8,
        'n_subcarriers': 256
    },
    'Boston5G_3p5_v5': {
        'n_rows': [812 + 4*int((1622-812)/20), 812 + 5*int((1622-812)/20)],
        'n_per_row': 595,
        'n_ant_bs': 8,
        'n_subcarriers': 512
    },
    'Boston5G_3p5_v6': {
        'n_rows': [812 + 5*int((1622-812)/20), 812 + 6*int((1622-812)/20)],
        'n_per_row': 595,
        'n_ant_bs': 8,
        'n_subcarriers': 1024
    },
    'Boston5G_3p5_v7': {
        'n_rows': [812 + 6*int((1622-812)/20), 812 + 7*int((1622-812)/20)],
        'n_per_row': 595,
        'n_ant_bs': 16,
        'n_subcarriers': 32
    },
    'Boston5G_3p5_v8': {
        'n_rows': [812 + 7*int((1622-812)/20), 812 + 8*int((1622-812)/20)],
        'n_per_row': 595,
        'n_ant_bs':16,
        'n_subcarriers': 64
    },
    'Boston5G_3p5_v9': {
        'n_rows': [812 + 8*int((1622-812)/20), 812 + 9*int((1622-812)/20)],
        'n_per_row': 595,
        'n_ant_bs': 16,
        'n_subcarriers': 128
    },
    'Boston5G_3p5_v10': {
        'n_rows': [812 + 9*int((1622-812)/20), 812 + 10*int((1622-812)/20)],
        'n_per_row': 595,
        'n_ant_bs': 16,
        'n_subcarriers': 256
    },
    'Boston5G_3p5_v11': {
        'n_rows': [812 + 10*int((1622-812)/20), 812 + 11*int((1622-812)/20)],
        'n_per_row': 595,
        'n_ant_bs': 16,
        'n_subcarriers': 512
    },
    'Boston5G_3p5_v12': {
        'n_rows': [812 + 11*int((1622-812)/20), 812 + 12*int((1622-812)/20)],
        'n_per_row': 595,
        'n_ant_bs': 32,
        'n_subcarriers': 32
    },
    'Boston5G_3p5_v13': {
        'n_rows': [812 + 12*int((1622-812)/20), 812 + 13*int((1622-812)/20)],
        'n_per_row': 595,
        'n_ant_bs': 32,
        'n_subcarriers': 64
    },
    'Boston5G_3p5_v14': {
        'n_rows': [812 + 13*int((1622-812)/20), 812 + 14*int((1622-812)/20)],
        'n_per_row': 595,
        'n_ant_bs': 32,
        'n_subcarriers': 128
    },
    'Boston5G_3p5_v15': {
        'n_rows': [812 + 14*int((1622-812)/20), 812 + 15*int((1622-812)/20)],
        'n_per_row': 595,
        'n_ant_bs': 32,
        'n_subcarriers': 256
    },
    'Boston5G_3p5_v16': {
        'n_rows': [812 + 15*int((1622-812)/20), 812 + 16*int((1622-812)/20)],
        'n_per_row': 595,
        'n_ant_bs': 64,
        'n_subcarriers': 32
    },
    'Boston5G_3p5_v17': {
        'n_rows': [812 + 16*int((1622-812)/20), 812 + 17*int((1622-812)/20)],
        'n_per_row': 595,
        'n_ant_bs': 64,
        'n_subcarriers': 64 
    },
    'Boston5G_3p5_v18': {
        'n_rows': [812 + 17*int((1622-812)/20), 812 + 18*int((1622-812)/20)],
        'n_per_row': 595,
        'n_ant_bs': 64,
        'n_subcarriers': 128
    },
    'Boston5G_3p5_v19': {
        'n_rows': [812 + 18*int((1622-812)/20), 812 + 19*int((1622-812)/20)],
        'n_per_row': 595,
        'n_ant_bs': 128,
        'n_subcarriers': 32
    },
    'Boston5G_3p5_v20': {
        'n_rows': [812 + 19*int((1622-812)/20), 812 + 20*int((1622-812)/20)],
        'n_per_row': 595,
        'n_ant_bs': 128,
        'n_subcarriers': 64
    },
    'O1_3p5_v1': {
        'n_rows': [0*int(3852/12), 1*int(3852/12)],
        'n_per_row': 181,
        'n_ant_bs': 8,
        'n_subcarriers': 32
    },
    'O1_3p5_v2': {
        'n_rows': [1*int(3852/12), 2*int(3852/12)],
        'n_per_row': 181,
        'n_ant_bs': 8,
        'n_subcarriers': 64
    },
    'O1_3p5_v3': {
        'n_rows': [2*int(3852/12), 3*int(3852/12)],
        'n_per_row': 181,
        'n_ant_bs': 8,
        'n_subcarriers': 128
    },
    'O1_3p5_v4': {
        'n_rows': [3*int(3852/12), 4*int(3852/12)],
        'n_per_row': 181,
        'n_ant_bs': 8,
        'n_subcarriers': 256
    },
    'O1_3p5_v5': {
        'n_rows': [4*int(3852/12), 5*int(3852/12)],
        'n_per_row': 181,
        'n_ant_bs': 8,
        'n_subcarriers': 512
    },
    'O1_3p5_v6': {
        'n_rows': [5*int(3852/12), 6*int(3852/12)],
        'n_per_row': 181,
        'n_ant_bs': 8,
        'n_subcarriers': 1024
    },
    'O1_3p5_v7': {
        'n_rows': [6*int(3852/12), 7*int(3852/12)],
        'n_per_row': 181,
        'n_ant_bs': 16,
        'n_subcarriers': 32
    },
    'O1_3p5_v8': {
        'n_rows': [7*int(3852/12), 8*int(3852/12)],
        'n_per_row': 181,
        'n_ant_bs': 16,
        'n_subcarriers': 64
    },
    'O1_3p5_v9': {
        'n_rows': [8*int(3852/12), 9*int(3852/12)],
        'n_per_row': 181,
        'n_ant_bs': 16,
        'n_subcarriers': 128
    },
    'O1_3p5_v10': {
        'n_rows': [9*int(3852/12), 10*int(3852/12)],
        'n_per_row': 181,
        'n_ant_bs': 16,
        'n_subcarriers': 256
    },
    'O1_3p5_v11': {
        'n_rows': [10*int(3852/12), 11*int(3852/12)],
        'n_per_row': 181,
        'n_ant_bs': 16,
        'n_subcarriers': 512
    },
    'O1_3p5_v12': {
        'n_rows': [11*int(3852/12), 12*int(3852/12)],
        'n_per_row': 181,
        'n_ant_bs': 32,
        'n_subcarriers': 32
    },
    'O1_3p5_v13': {
        'n_rows': [12*int(3852/12)+0*int(1351/10), 12*int(3852/12)+1*int(1351/10)],
        'n_per_row': 361,
        'n_ant_bs': 32,
        'n_subcarriers': 64
    },
    'O1_3p5_v14': {
        'n_rows': [12*int(3852/12)+1*int(1351/10), 12*int(3852/12)+2*int(1351/10)],
        'n_per_row': 181,
        'n_ant_bs': 32,
        'n_subcarriers': 128
    },
    'O1_3p5_v15': {
        'n_rows': [12*int(3852/12)+2*int(1351/10), 12*int(3852/12)+3*int(1351/10)],
        'n_per_row': 181,
        'n_ant_bs': 32,
        'n_subcarriers': 256
    },
    'O1_3p5_v16': {
        'n_rows': [12*int(3852/12)+3*int(1351/10), 12*int(3852/12)+4*int(1351/10)],
        'n_per_row': 181,
        'n_ant_bs': 64,
        'n_subcarriers': 32
    },
    'O1_3p5_v17': {
        'n_rows': [12*int(3852/12)+4*int(1351/10), 12*int(3852/12)+5*int(1351/10)],
        'n_per_row': 181,
        'n_ant_bs': 64,
        'n_subcarriers': 64
    },
    'O1_3p5_v18': {
        'n_rows': [12*int(3852/12)+5*int(1351/10), 12*int(3852/12)+6*int(1351/10)],
        'n_per_row': 181,
        'n_ant_bs': 64,
        'n_subcarriers': 128
    },
    'O1_3p5_v19': {
        'n_rows': [12*int(3852/12)+6*int(1351/10), 12*int(3852/12)+7*int(1351/10)],
        'n_per_row': 181,
        'n_ant_bs': 128,
        'n_subcarriers': 32
    },
    'O1_3p5_v20': {
        'n_rows': [12*int(3852/12)+7*int(1351/10), 12*int(3852/12)+8*int(1351/10)],
        'n_per_row': 181,
        'n_ant_bs': 128,
        'n_subcarriers': 64
    },
    'city_0_newyork_v16x64': {
        'n_rows': 109,
        'n_per_row': 291,
        'n_ant_bs': 16,
        'n_subcarriers': 64
    },
    'city_1_losangeles_v16x64': {
        'n_rows': 142,
        'n_per_row': 201,
        'n_ant_bs': 16,
        'n_subcarriers': 64
    },
    'city_2_chicago_v16x64': {
        'n_rows': 139,
        'n_per_row': 200,
        'n_ant_bs': 16,
        'n_subcarriers': 64
    },
    'city_3_houston_v16x64': {
        'n_rows': 154,
        'n_per_row': 202,
        'n_ant_bs': 16,
        'n_subcarriers': 64
    },
    'city_4_phoenix_v16x64': {
        'n_rows': 198,
        'n_per_row': 214,
        'n_ant_bs': 16,
        'n_subcarriers': 64
    },
    'city_5_philadelphia_v16x64': {
        'n_rows': 239,
        'n_per_row': 164,
        'n_ant_bs': 16,
        'n_subcarriers': 64
    },
    'city_6_miami_v16x64': {
        'n_rows': 199,
        'n_per_row': 216,
        'n_ant_bs': 16,
        'n_subcarriers': 64
    },
    'city_7_sandiego_v16x64': {
        'n_rows': 207,
        'n_per_row': 176,
        'n_ant_bs': 16,
        'n_subcarriers': 64
    },
    'city_8_dallas_v16x64': {
        'n_rows': 207,
        'n_per_row': 190,
        'n_ant_bs': 16,
        'n_subcarriers': 64
    },
    'city_9_sanfrancisco_v16x64': {
        'n_rows': 196,
        'n_per_row': 206,
        'n_ant_bs': 16,
        'n_subcarriers': 64
    }}
    return row_column_users