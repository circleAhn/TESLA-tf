DATASET = './dataset'
FD_DATASET = './finedust_dataset'
FDC_DATASET = './finedust_concentration_dataset'
FDS_DATASET = './finedust_concentration_split_dataset'
FDM_DATASET = './finedust_concentration_minrefine_dataset'
FDA_DATASET = './finedust_concentration_alldrop_dataset'
FDSIM_DATASET = './finedust_concentration_simdrop_dataset'
FDCOL_DATASET = './finedust_concentration_alldrop_firstcol_dataset'

META_DATES = './metadata/metadata_dates.csv'
META_DATES_PREP = './metadata/metadata_dates_prep.csv'

ANTWERP = 'Antwerp_'
OSLO = 'oslo_'
ZAGREB = 'Zagreb_'

FIRST_COL = '_FirstCol'
DEPLOYMENT = '_Deployment'
SECOND_COL = '_SecondCol'

DROP_SIM = {ANTWERP: 0.4, OSLO: 0.4, ZAGREB: 0.4}

COLLECT_TYPE = [FIRST_COL, DEPLOYMENT, SECOND_COL]

ANTWERP_DATA_LIST = ['402B00', '4043AE', '4043B1', '4047CD', '4047D7', # 40499F, 4043A7, 40499C dupl
                     '4047DD', '4047E0', '4047E7', '4049A6', '4065D0', 
                     '4065D3', '4065DA', '4065DD', '4065E0', '4065E3', 
                     '4065EA', '4067B0', '4067B3', '4067BA', '4067BD', 
                     '40623F', '40641B', '40642B', '402723', '406246', 
                     '406249', '406424', '408165', '408168', '408175', 
                     '408178']

OSLO_DATA_LIST = ['64A291', '64A292', '64B082', '64CB6D', '64CB70', # 64FD11, 65063E, 65325E dupl
                  '64CB78', '64E9C5', '64FD0A', '425FB3', '425FB4', # 40816F X
                  '647D5A', '648B91', '651EF5', '651EFC', '652A32', 
                  '652D3A', '652FA4', '652FAF', '6517DD', 
                  '40458D', '40817F', '40642E', '65326C',
                  '426178', '426179', '649312', '649526', '653257']

ZAGREB_DATA_LIST = ['64C52B', '64C225', '652A38', '652FA1', #427907 dupl
                    '4047D0', '40641E', '42816D', '42816D', '64876B',
                    '64876C', '427906', '428164', '648157', 
                    '648169', '649738']


ANTWERP_DROP_LIST = ['4043AE', '4047D7', '40642B', '408165', '408168']
OSLO_DROP_LIST = ['64A291', '64CB70', '64E9C5', '64FD0A', '425FB3',
                  '651EFC', '652D3A', '652FAF', '65326C', '649312',
                  '649526'] + ['40816F', '4065ED', '40641B']

ZAGREB_DROP_LIST =  ['64C225', '406414', '427906'] + ['64E03B']



ANTWERP_DROPPED_DATA_LIST = ['402B00_0', '402B00_1', '4043AE_0', '4043B1_0', '4047CD_0',
                             '4047D7_0', '4047DD_0', '4047DD_1', '4047E0_0', '4047E0_1', 
                             '4047E7_0', '4049A6_0', '4049A6_1', '4065D0_0', '4065DA_0',
                             '4065DD_0', '4065E0_0', '4065E3_0', '4065E3_1', '4065EA_0', 
                             '4067B0_0', '4067B0_1', '4067B3_0', '4067B3_1', '4067BA_0', 
                             '4067BA_1', '4067BD_0', '4067BD_1', '40623F_0', '40623F_1',
                             '40641B_0', '40642B_0', '406246_0', '406249_0', '406424_0', 
                             '408165_0', '408165_1', '408168_0', '408175_0', '408175_1', 
                             '408178_0']

OSLO_DROPPED_DATA_LIST = ['64CB6D_0', '425FB4_0', '647D5A_0', '648B91_0', '652FA4_0', 
                          '40817F_0', '426178_0', '426179_0']

ZAGREB_DROPPED_DATA_LIST = ['64C52B_0', '64C52B_1', '64C225_0', '652FA1_0', '4047D0_0', 
                            '40641E_0', '64876B_0', '64876C_0', '64876C_1', '427906_0', 
                            '428164_0', '648169_0']



ANTWERP_FIRST_COL_LIST = ['402B00', '4043AE', '4043B1', '4047CD', 
                          '4047DD', '4047E0', '4047E7', '4049A6', '4065D0', 
                          '4065D3', '4065DA', '4065DD', '4065E0', '4065E3', 
                          '4065EA', '4067B0', '4067B3', '4067BA', '4067BD', 
                          '40623F', '40641B', '40642B', '402723', '406246', 
                          '406249', '406424', '408165', '408168', '408175', 
                          '408178']
OSLO_FIRST_COL_LIST = ['64A291', '64CB6D', '64CB70', '64CB78', 
                       '64E9C5', '425FB3', '647D5A', '648B91', 
                       '651EF5', '651EFC', '652A32', '652D3A',
                       '652FA4', '652FAF', '6517DD', '40817F',
                       '40642E', '426178', '649312', '649526', '653257']

ZAGREB_FIRST_COL_LIST = ['64C52B', '64C225', '652A38', '652FA1', '4047D0', 
                         '40641E', '42816D', '64876B', '64876C', '427906', 
                         '428164', '648157', '648169', '649738']

PM_COLUMNS = ['date', 'Location.ID', 
              '5310CAT', '5325CAT', '5301CAT', '5310CST', '5325CST', '5301CST', 
              '53PT003', '53PT005', '53PT010', '53PT025', '53PT050', '53PT100',
              'OPCN3PM10', 'OPCN3PM25', 'OPCN3PM1', 
              'OPCN3Bin0', 'OPCN3Bin1', 'OPCN3Bin2', 'OPCN3Bin3', 'OPCN3Bin4', 'OPCN3Bin5', 'OPCN3Bin6', 'OPCN3Bin7', 'OPCN3Bin8', 'OPCN3Bin9',
              'OPCN3Bin10', 'OPCN3Bin11', 'OPCN3Bin12', 'OPCN3Bin13', 'OPCN3Bin14', 'OPCN3Bin15', 'OPCN3Bin16', 'OPCN3Bin17', 'OPCN3Bin18', 'OPCN3Bin19',
              'OPCN3Bin20', 'OPCN3Bin21', 'OPCN3Bin22', 'OPCN3Bin23',
              '5310CAT_flag', '5325CAT_flag', '5301CAT_flag',
              'OPCN3PM10_flag', 'OPCN3PM25_flag', 'OPCN3PM1_flag',
              'Ref.PM10', 'Ref.PM2.5', 'Ref.PM1']

PMC_COLUMNS = ['date', 'Location.ID', 
              '5310CAT', '5325CAT', '5301CAT', 
              'OPCN3PM10', 'OPCN3PM25', 'OPCN3PM1', 
              '5310CAT_flag', '5325CAT_flag', '5301CAT_flag',
              'OPCN3PM10_flag', 'OPCN3PM25_flag', 'OPCN3PM1_flag',
              'Ref.PM10', 'Ref.PM2.5', 'Ref.PM1']

PMS_COLUMNS = ['date',
              '5310CAT', '5325CAT', '5301CAT', 
              'OPCN3PM10', 'OPCN3PM25', 'OPCN3PM1', 
              '5310CAT_flag', '5325CAT_flag', '5301CAT_flag',
              'OPCN3PM10_flag', 'OPCN3PM25_flag', 'OPCN3PM1_flag']

PMF_COLUMNS = ['date', 'Location.ID', 
               '5310CAT', '5325CAT', '5301CAT', 
              'OPCN3PM10', 'OPCN3PM25', 'OPCN3PM1']

PMSIM_COLUMNS = ['date',
               '5310CAT', '5325CAT', '5301CAT', 
              'OPCN3PM10', 'OPCN3PM25', 'OPCN3PM1']

META_COLUMNS = ['ASEs', 'Deployment_Start', 'Second_Col_Start',]