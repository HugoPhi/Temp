'''
|     name     |  BS(test, train)  |  backend  | device |
| sklearn      |       None        | sklearn   |   cpu  |
| bb_numpy     | (256   , 1024   ) | numpy     |   cpu  |
| fb_numpy     | (n_test, 64     ) | numpy     |   cpu  |
| bf_numpy     | (16    , n_train) | numpy     |   cpu  |
| pf_numpy     | (n_test, 1      ) | numpy     |   cpu  |
| fp_numpy     | (1     , n_train) | numpy     |   cpu  |
| ff_numpy     | (1     , 1      ) | numpy     |   cpu  |
| bb_torch_cpu | (256   , 1024   ) | torch_cpu |   cpu  |
| fb_torch_cpu | (n_test, 64     ) | torch_cpu |   cpu  |
| bf_torch_cpu | (16    , n_train) | torch_cpu |   cpu  |
| pf_torch_cpu | (n_test, 1      ) | torch_cpu |   cpu  |
| fp_torch_cpu | (1     , n_train) | torch_cpu |   cpu  |
| ff_torch_cpu | (1     , 1      ) | torch_cpu |   cpu  |
| bb_torch     | (256   , 1024   ) | torch     |   gpu  |
| fb_torch     | (n_test, 64     ) | torch     |   gpu  |
| bf_torch     | (16    , n_train) | torch     |   gpu  |
| pf_torch     | (n_test, 1      ) | torch     |   gpu  |
| fp_torch     | (1     , n_train) | torch     |   gpu  |
| ff_torch     | (1     , 1      ) | torch     |   gpu  |
'''

train = 5000
test  = 1000