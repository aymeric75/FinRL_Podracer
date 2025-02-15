import sys
import os
import pickle

# Add the directory of the current script to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from processor_yahoofinance import YahooFinanceProcessor 

# date must be like: Month day Year

start_date='2009-01-01'
#start_date='01-01-2009'

end_date='2021-05-27'

#end_date='05-27-2021'

liste = [
'AAPL', 'ADBE', 'ADI', 'ADP', 'ADSK', 'ALGN', 'ALXN', 'AMAT', 'AMD', 'AMGN',
'AMZN', 'ASML', 'ATVI', 'BIIB', 'BKNG', 'BMRN', 'CDNS', 'CERN', 'CHKP', 'CMCSA',
'COST', 'CSCO', 'CSX', 'CTAS', 'CTSH', 'CTXS', 'DLTR', 'EA', 'EBAY', 'FAST',
'FISV', 'GILD', 'HAS', 'HSIC', 'IDXX', 'ILMN', 'INCY', 'INTC', 'INTU', 'ISRG',
'JBHT', 'KLAC', 'LRCX', 'MAR', 'MCHP', 'MDLZ', 'MNST', 'MSFT', 'MU', 'MXIM',
'NLOK', 'NTAP', 'NTES', 'NVDA', 'ORLY', 'PAYX', 'PCAR', 'PEP', 'QCOM', 'REGN',
'ROST', 'SBUX', 'SIRI', 'SNPS', 'SWKS', 'TTWO', 'TXN', 'VRSN', 'VRTX', 'WBA',
'WDC', 'WLTW', 'XEL', 'XLNX'
]

yfp = YahooFinanceProcessor()

for ele in liste:

    df = yfp.scrap_data([ele], start_date, end_date)

    if df is None:
        print("{} could not be scrapped".format(str(ele)))
        continue
    str_tickers = "_".join(tuple([ele]))
    with open("PriceData_"+str_tickers+"_"+str(start_date)+"_"+str(end_date)+".pkl", 'wb') as file:
        pickle.dump(df, file)