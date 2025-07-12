
# from meta.env_stock_trading.env_nasdaq100_wrds import StockEnvNAS100
import pandas as pd
#from env_nasdaq100_wrds import StockEnvNAS100

# from finrl.meta.preprocessor.preprocessors import FeatureEngineer
#from finrl.meta.env_stock_trading.env_nas100_wrds import StockEnvNAS100

from ElegantRL.elegantrl.envs.StockTradingEnv import StockTradingVecEnv

import torch
import pickle
import inspect
import gc
from ElegantRL.elegantrl.train.config import Config
from ElegantRL.elegantrl.agents.AgentPPO import AgentPPO
#from FinRLPodracer.elegantrl.agent import AgentPPO

from ElegantRL.elegantrl.train.run import train_agent


import logging

import numpy as np

# logging.basicConfig(
#     filename="script.log",
#     filemode="w",
#     level=logging.DEBUG,  # or INFO
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )
import sys
import psutil
import os

# def handle_exception(exc_type, exc_value, exc_traceback):
#     logging.exception("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

# sys.excepthook = handle_exception

# import faulthandler
# faulthandler.enable(file=open('fatal_log.txt', 'a'))

# import elegantrl
# print(elegantrl.__file__)
# exit()
# file_path = "/workspace/nasdaq_100_minute_data.csv"

# # Load the CSV file as a Dask DataFrame
# # df = dd.read_csv(file_path)
# df = pd.read_csv(file_path, nrows=10000)

# # Show the first few rows
# print('ICIIIIII ')

# print(df.head())


# df = df.rename(columns={'ticker': 'tic'})
# df = df.rename(columns={'datetime': 'date'})


# ######  BUT YOU WANT SOMETHING LIKE raw_df:

# #             date    open    high     low   close   adjcp   volume   tic   day
# # 0     2009-01-02   19.55   20.76   19.55   20.68   20.68  2845600  ADSK     1
# # 1     2009-01-05   20.60   20.92   20.05   20.88   20.88  3018600  ADSK     4
# # 2     2009-01-06   21.11   21.71   20.71   20.85   20.85  3883500  ADSK     5
# # 3     2009-01-07   20.55   20.58   19.89   20.07   20.07  3831400  ADSK     6
# # 4     2009-01-08   19.59   19.61   18.96   19.20   19.20  5436400  ADSK     7
# # ...          ...     ...     ...     ...     ...     ...      ...   ...   ...
# # 34326 2021-05-20  632.69  652.25  631.75  648.77  627.55  1086900  ASML  4522
# # 34327 2021-05-21  645.86  647.36  636.18  639.22  618.31   847500  ASML  4523

# #### IN ORDER TO THEN DO:

# # 

# # WHERE FE IS 

# tech_indicator_list = ['macd', 'boll_ub', 'boll_lb', 'rsi_30', 'dx_30']

# fe = FeatureEngineer(use_turbulence=True,
#                      user_defined_feature=False,
#                      use_technical_indicator=True,
#                      tech_indicator_list=tech_indicator_list, )


# ### WHERE FeatureEngineer is from finrl/meta/preprocessor/preprocessors.py

# processed_df = fe.preprocess_data(df)












def main():

    try:




        # # Save the new DataFrame with unique tickers
        # with open('data/df_tic_only_unique.pkl', 'wb') as f:
        #     pickle.dump(unique_tics, f)

        with open('data/df_tic_only_unique.pkl', 'rb') as f:
            df_tic_only = pickle.load(f)

        # with open('data/price_ary.pkl', 'rb') as f:
        #     price_ary = pickle.load(f)

        # price_ary = price_ary[::25]
        # with open('data/price_ary_downsampled.pkl', 'wb') as f:
        #     pickle.dump(price_ary, f)

        with open('data/price_ary_downsampled.pkl', 'rb') as f:
            price_ary = pickle.load(f)

        # with open('data/tech_ary.pkl', 'rb') as f:
        #     tech_ary = pickle.load(f)

        # tech_ary = tech_ary[::25]
        # with open('data/tech_ary_downsampled.pkl', 'wb') as f:
        #     pickle.dump(tech_ary, f)

        with open('data/tech_ary_downsampled.pkl', 'rb') as f:
            tech_ary = pickle.load(f)

        # with open('data/turbulence_ary_one_dim.pkl', 'rb') as f:
        #     turbulence_ary = pickle.load(f)


        # turbulence_ary = turbulence_ary[::25]
        # with open('data/turbulence_ary_one_dim_downsampled.pkl', 'wb') as f:
        #     pickle.dump(turbulence_ary, f)


        with open('data/turbulence_ary_one_dim_downsampled.pkl', 'rb') as f:
            turbulence_ary = pickle.load(f)


        stock_dim = price_ary.shape[1]

        print(price_ary.shape)
        print(tech_ary.shape)
        print(turbulence_ary.shape)
        gc.collect()
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()

        # Print memory in MB
        print(f"Memory used: {mem_info.rss / 1024**3:.4f} GB") # Memory used: 48375.21 MB
        #   the fuck 47 !!!GB
        # 


        # print(df_tic_only)
        # print(price_ary.shape)
        # print(tech_ary.shape)
        # print(turbulence_ary.shape)

        # # 1954080
        # exit()

        tech_id_list = [
            "macd", "boll_ub", "boll_lb", "rsi_30", "dx_30", "close_30_sma", "close_60_sma",
        ] 

        # print(len(df_tic_only)) # 156326400
        # print(len(price_ary)) # 1954080
        # print(len(tech_ary)) # 1954080

        # GROUND TRUTH PARAMETERS
        worker_num = 2
        num_envs = 4
        batch_size = 1024
        buffer_size = 4096
        horizon_len = 1024
        eval_per_step = 1024
        max_step = 1024
        repeat_times = 8

        # ###########  PARAMETER GENERAL DEFINITION
        # worker_num = 2
        # num_envs = 4
        # batch_size = 64
        # buffer_size = 256 # 
        # repeat_times = 8
        # horizon_len = 64 # <=> + ou - le batch size
        # eval_per_step = 64 # une fois que Evaluator a commencé à évaluté, ré évalue tous les eval_per_step steps
        # #                                   où "steps", c'est en fait le nbre de steps donné à evaluate_and_save, donc ??????????????????
        # max_step = 64 # le nbre max de steps à prendre pour évaluer l'environnement (voir dans get_cumulative_rewards_and_step_from_vec_env)
        
        # exit()

        args = Config(agent_class=AgentPPO, env_class=StockTradingVecEnv)

        turbulence_ary = turbulence_ary.astype(np.float16)

        args.learner_gpu_ids = tuple(i for i in range(torch.cuda.device_count()))



        # If your env needs data passed, make sure to do this:
        args.env_args = {
            #'df': df,
            'stock_dim': len(df_tic_only.tic.unique()),
            'tech_indicator_list': tech_id_list, #['macd', 'rsi', 'cci', 'adx'],  # adjust as needed
            'turbulence_thresh': 150,  # adjust if relevant
            'close_ary': price_ary.astype(np.float16),
            'tech_ary': tech_ary.astype(np.float16),
            'turbulence_ary': turbulence_ary,
            "repeat_times": repeat_times,
            "learning_rate": 2**-14,
            "net_dim": 2**8,
            "batch_size": batch_size, #2**6, #2**10,
            "eval_gap": 2**8,
            "eval_times1": 2**0,
            "eval_times2": 2**1,
            "break_step": 2**12, #int(25e6),
            "if_allow_break": False,
            "worker_num": worker_num,
            "cwd": None,
            "dims": [128, 128],
            "env_name": 'StockTradingVecEnv',
            "num_envs": num_envs, #200,
            "max_step": 640, #price_ary.shape[0] - 1,
            "state_dim": 721, #803, #1 + 2 + 3 * stock_dim + tech_ary.shape[1], # 803
            "action_dim": stock_dim,
            "if_discrete": False,
            "max_stock": len(df_tic_only.tic.unique())
            #"cwd": "/workspace/FinRL_Podracer/data"
        }


        #1 + 2 + 3 * stock_dim + tech_ary.shape[1]

        print((3 * stock_dim)) # 240
        print(tech_ary.shape[1]) # 560



        ########## THE REAL STATE SPACE ###############

        ## Balance b_t: 1
        ## Shares: how many shares per stock (80) 
        ## Closing prices: 80
        ## tech indicators: 80 x 7  (560)
        ## turbulence: 80
        ### 560+80+80+80+1 = 801



        #### TOUTES LES VARS QUI ONT A VOIR AVEC STEPS
        args.break_step = len(price_ary)
        args.horizon_len = horizon_len #2**6 # 64 # pour moi c'est batch size
        args.eval_per_step = eval_per_step
        args.max_step = max_step 



        args.net_dims = [128, 128] 
        
        args.state_dim = 721 #803 #1 + 2 + 3 * stock_dim + tech_ary.shape[1]
        args.action_dim = stock_dim
        args.env_name = 'StockTradingVecEnv'
        args.if_off_policy = False  # False when use PPO
        args.num_envs = num_envs #200
        args.num_workers = worker_num



        #  the 'args' are also the one given to the Learner, e.g. AgentPPO
        #   
        #   
        #   
        #   
        args.batch_size = batch_size #1024
        args.buffer_size = buffer_size
        args.repeat_times = repeat_times
        args.gamma = 0.99
        args.learning_rate = 2**-14
        args.buffer_init_size = buffer_size #4096
        args.ratio_clip = 0.25
        args.lambda_entropy = 0.02
        args.if_vec_env = True
        train_agent(args) 


    except Exception as e:
        logging.exception("Exception occurred during training:")

if __name__ == "__main__":
    main()








# with open('objectstorage/df_cleaned_and_tech_and_turb_and_filled.pkl', 'rb') as f:
#     processed_df = pickle.load(f)


# from elegantrl.train.config import Config
# from elegantrl.agents.AgentPPO import AgentPPO

# from elegantrl.train.run import train_agent

# args  = Config(agent_class=AgentPPO, env_class=StockTradingEnv)

# args.learner_gpu_ids = tuple(i for i in range(torch.cuda.device_count()))

# train_agent(args)