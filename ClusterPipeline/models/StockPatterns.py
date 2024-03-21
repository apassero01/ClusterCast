import yfinance
import mplfinance as mpf
import pandas as pd
import numpy as np
import pandas_ta as ta
import ta as technical_analysis


class IndicatorFactory:
    '''
    Abstract class for a factory that creates features for specific indicator(s)
    '''
    def __init__(self):
        pass

class OHLCVFactory(IndicatorFactory):
    '''
    Factory class to create features for a stock from its OHLCV data
    '''
    def __init__(self, df):
        self.df = df
        self.ohlcPctChg_df = pd.DataFrame(index=self.df.index)
        self.ohlcIntraDay_df = pd.DataFrame(index=self.df.index)
    
    def createPctChg(self):
        '''
        Method to create percent change features for a stock's OHLCV data
        '''
        for col in self.df.columns:
            new_col = 'pctChg' + col
            self.ohlcPctChg_df[new_col] = self.df[col].pct_change() * 100 
            self.ohlcPctChg_df[new_col] = self.ohlcPctChg_df[new_col].replace([np.inf, -np.inf], np.nan)
            self.ohlcPctChg_df[new_col] = self.ohlcPctChg_df[new_col].fillna(method='bfill')        
        return self.ohlcPctChg_df

    def createIntraDay(self):
        '''
        Method to create intra-day features for a stock's OHLCV data
        '''

        self.ohlcIntraDay_df['opHi'] = (self.df.high - self.df.open) / self.df.open * 100.0
        
        # % drop from open to low
        self.ohlcIntraDay_df['opLo'] = (self.df.low - self.df.open) / self.df.open * 100.0

        # % drop from high to close
        self.ohlcIntraDay_df['hiCl'] = (self.df.close - self.df.high) / self.df.high * 100.0

        # % raise from low to close
        self.ohlcIntraDay_df['loCl'] = (self.df.close - self.df.low) / self.df.low * 100.0

        # % spread from low to high
        self.ohlcIntraDay_df['hiLo'] = (self.df.high - self.df.low) / self.df.low * 100.0

        # % spread from open to close
        self.ohlcIntraDay_df['opCl'] = (self.df.close - self.df.open) / self.df.open * 100.0

        # # Calculations for the percentage changes
        self.ohlcIntraDay_df["pctChgClOp"] = np.insert(np.divide(self.df.open.values[1:], self.df.close.values[0:-1]) * 100.0 - 100.0, 0, np.nan)

        self.ohlcIntraDay_df["pctChgClLo"] = np.insert(np.divide(self.df.low.values[1:], self.df.close.values[0:-1]) * 100.0 - 100.0, 0, np.nan)

        self.ohlcIntraDay_df["pctChgClHi"] = np.insert(np.divide(self.df.high.values[1:], self.df.close.values[0:-1]) * 100.0 - 100.0, 0, np.nan)

        for col in self.ohlcIntraDay_df.columns:
            self.ohlcIntraDay_df[col] = self.ohlcIntraDay_df[col].fillna(method='bfill')

        return self.ohlcIntraDay_df

class MovingAverageFactory(IndicatorFactory):
    '''
    Factory class to create moving average features for a stock. 
    '''
    def __init__(self, df):
        self.df = df 
        self.sma_df = pd.DataFrame(index=self.df.index)
        self.ema_df = pd.DataFrame(index=self.df.index)
        self.smaVol_df = pd.DataFrame(index=self.df.index)
        self.emaPctDiff_df = pd.DataFrame(index=self.df.index)
        self.smaPctDiff_df = pd.DataFrame(index=self.df.index)
        self.smaPctDiffVol_df = pd.DataFrame(index=self.df.index)
        self.emaDerivative_df = pd.DataFrame(index=self.df.index)
        self.smaDerivative_df = pd.DataFrame(index=self.df.index)
        self.smaDerivativeVol_df = pd.DataFrame(index=self.df.index)

    def createAllFeatures(self):
        '''
        Method to create moving average features for a stock. 
        
        '''
        pass

    def createSMA(self, periods = [5, 10, 20, 30, 50, 100,200]):
        '''
        Method to create simple moving average feature for a stock. 
        
        '''
        for window in periods:
            self.sma_df["sma" + str(window)] = ta.sma(self.df.close, length=window)
            self.sma_df["sma" + str(window)] = self.sma_df["sma" + str(window)].fillna(method='bfill')
        
        return self.sma_df
    
    def createEMA(self, periods = [5, 10, 20, 30, 50, 100,200]):
        '''
        Method to create exponential moving average feature for a stock. 
        
        '''
        for window in periods:
            self.ema_df["ema" + str(window)] = ta.ema(self.df.close, length=window)
            self.ema_df["ema" + str(window)] = self.ema_df["ema" + str(window)].fillna(method='bfill')
        return self.ema_df
    
    def createSMAVolume(self, periods = [5,10,20,50] ):
        '''
        Method to create exponential moving average feature for a stocks volume. 
        '''
        for window in periods:
            self.smaVol_df["smaVol" + str(window)] = ta.sma(self.df.volume, length=window)
            self.smaVol_df["smaVol" + str(window)] = self.smaVol_df["smaVol" + str(window)].fillna(method='bfill')
        
        return self.smaVol_df
    
    def createSMAPctDiff(self, closeOnly = False):
        '''
        Method to create percent difference between close price and simple moving averages
        @param closeOnly boolean to determine if only close price should be used or the difference between all smas
        '''
        for col in self.sma_df.columns: 
            col_name = 'pctDiff+' + col + '_close'
            self.smaPctDiff_df[col_name] = (self.df.close - self.sma_df[col])/self.sma_df[col] * 100
            self.smaPctDiff_df[col_name] = self.smaPctDiff_df[col_name].fillna(method='bfill')
            if not closeOnly:
                for i,col in enumerate(self.sma_df.columns):
                    for j in range(i+1, len(self.sma_df.columns)):
                        col2 = self.sma_df.columns[j]
                        col_name = 'pctDiff+' + col + '_' + col2 
                        self.smaPctDiff_df[col_name] = (self.sma_df[col] - self.sma_df[col2])/self.sma_df[col2] * 100
                        self.smaPctDiff_df[col_name] = self.smaPctDiff_df[col_name].fillna(method='bfill')

        return self.smaPctDiff_df
    
    def createEMAPctDiff(self, closeOnly = False):
        '''
        Method to create percent difference between close price and exponential moving averages
        @param closeOnly boolean to determine if only close price should be used or the difference between all emas
        '''
        for col in list(self.ema_df.columns): 
            col_name = 'pctDiff+' + col + '_close'
            self.emaPctDiff_df[col_name] = (self.df.close - self.ema_df[col])/self.ema_df[col] * 100
            self.emaPctDiff_df[col_name] = self.emaPctDiff_df[col_name].fillna(method='bfill')
        if not closeOnly:
            for i,col in enumerate(self.ema_df.columns):
                for j in range(i+1, len(self.ema_df.columns)):
                    col2 = self.ema_df.columns[j]
                    col_name = 'pctDiff+' + col + '_' + col2 
                    self.emaPctDiff_df[col_name] = (self.ema_df[col] - self.ema_df[col2])/self.ema_df[col2] * 100
                    self.emaPctDiff_df[col_name] = self.emaPctDiff_df[col_name].fillna(method='bfill')

        return self.emaPctDiff_df
    
    def createSMAPctDiffVol(self, VolOnly = False):
        '''
        Method to create percent difference between volume and exponential moving averages
        @param VolOnly boolean to determine if only volume should be used or the difference between all emas
        '''
        for col in self.smaVol_df.columns: 
            col_name = 'pctDiff+' + col + '_volume'
            self.smaPctDiffVol_df[col_name] = (self.df.volume - self.smaVol_df[col])/self.smaVol_df[col] * 100
            self.smaPctDiffVol_df[col_name] = self.smaPctDiffVol_df[col_name].fillna(method='bfill')
        if not VolOnly:
            for i, col in enumerate(self.smaVol_df.columns):
                for j in range(i+1, len(self.smaVol_df.columns)):
                    col2 = self.smaVol_df.columns[j]
                    col_name = 'pctDiff+' + col + '_' + col2 
                    self.smaPctDiffVol_df[col_name] = (self.smaVol_df[col] - self.smaVol_df[col2])/self.smaVol_df[col2] * 100
                    self.smaPctDiffVol_df[col_name] = self.smaPctDiffVol_df[col_name].fillna(method='bfill')

        return self.smaPctDiffVol_df
    def createSMADerivative(self):
        '''
        Method to create derivative of simple moving averages
        '''
        for col in self.sma_df.columns:
            period = int(col.replace('sma',''))
            self.smaDerivative_df['deriv+'+col] = (self.sma_df[col] - self.sma_df[col].shift(period)) / period
            self.smaDerivative_df['deriv+'+col] = self.smaDerivative_df['deriv+'+col].fillna(method='bfill')

        return self.smaDerivative_df
    
    def createEMADerivative(self):
        '''
        Method to create derivative of exponential moving averages
        '''
        for col in self.ema_df.columns:
            period = int(col.replace('ema',''))
            self.emaDerivative_df['deriv+'+col] = (self.ema_df[col] - self.ema_df[col].shift(period)) / period
            self.emaDerivative_df['deriv+'+col] = self.emaDerivative_df['deriv+'+col].fillna(method='bfill')

        return self.emaDerivative_df

    def createSMADerivativeVol(self):
        '''
        Method to create derivative of exponential moving averages for volume
        '''
        for col in self.smaVol_df.columns:
            period = int(col.replace('smaVol',''))
    
            self.smaDerivativeVol_df['deriv+'+col] = (self.smaVol_df[col] - self.smaVol_df[col].shift(period)) / period
            self.smaDerivativeVol_df['deriv+'+col] = self.smaDerivativeVol_df['deriv+'+col].fillna(method='bfill')

        return self.smaDerivativeVol_df
    

class BandFactory(IndicatorFactory):
    '''
    Factory class to create band related features for a stock
    '''
    def __init__(self,df):
        self.df = df 
        self.bollinger_df = pd.DataFrame(index=self.df.index)
        self.bollPctDiff_df = pd.DataFrame(index=self.df.index)
    
    def createBB(self, periods = [10,20,40,60]): 
        '''
        Method to create bollinger bands for a stock
        '''
        self.bb_periods = periods

        for window in periods: 
            bollinger_obj = technical_analysis.volatility.BollingerBands(close=self.df.close, window=window)
            high_col = "bb_high" + str(window)
            low_col = "bb_low" + str(window)

            self.bollinger_df[high_col] = bollinger_obj.bollinger_hband()
            self.bollinger_df[low_col] = bollinger_obj.bollinger_lband()
            self.bollinger_df[high_col] = self.bollinger_df[high_col].fillna(method='bfill')
            self.bollinger_df[low_col] = self.bollinger_df[low_col].fillna(method='bfill')

        return self.bollinger_df

    def createBBPctDiff(self): 
        '''
        Method to create percent difference between close price and bollinger bands
        '''
        for window in self.bb_periods: 
            high_col = "bb_high" + str(window)
            low_col = "bb_low" + str(window)
            pct_diff_high_low_col = "pctDiff+bb_high_low" + str(window)
            pct_diff_high_close_col = "pctDiff+bb_high_close" + str(window)
            pct_diff_low_close_col = "pctDiff+bb_low_close" + str(window)
            bb_indicator_col = "bb_indicator" + str(window)
            self.bollPctDiff_df[pct_diff_high_low_col] = (self.bollinger_df[high_col] - self.bollinger_df[low_col])/self.bollinger_df[low_col] * 100
            self.bollPctDiff_df[pct_diff_high_close_col] = (self.bollinger_df[high_col] - self.df.close)/self.df.close * 100
            self.bollPctDiff_df[pct_diff_low_close_col] = (self.bollinger_df[low_col] - self.df.close)/self.df.close * 100
            self.bollPctDiff_df[bb_indicator_col] = (self.df.close - self.bollinger_df[low_col])/(self.bollinger_df[high_col] - self.bollinger_df[low_col]) * 100

    
        return self.bollPctDiff_df

class MomentumFactory(IndicatorFactory):
    '''
    Factory class to create memory related features for a stock
    '''
    def __init__(self, df):
        self.df = df
        self.rsi_df = pd.DataFrame(index=self.df.index)
        self.macd_df = pd.DataFrame(index=self.df.index)
        self.stoch_df = pd.DataFrame(index=self.df.index)
    
    def createRSI(self, periods = [5,10,20,50,100]):
        '''
        Method to create relative strength index for a stock
        '''
        for period in periods:
            self.rsi_df["rsi" + str(period)] = ta.rsi(self.df.close, length=period)
            self.rsi_df["rsi" + str(period)] = self.rsi_df["rsi" + str(period)].fillna(method='bfill')
        
        return self.rsi_df

    def createMACD(self):
        new_macd_df = self.df.ta.macd(fast=12, slow=26, append = False)
        self.macd_df = pd.concat([self.macd_df, new_macd_df], axis=1)
        self.macd_df.columns = ["macd", "macd_signal", "macd_diff"]
        self.macd_df = self.macd_df.fillna(method='bfill')
        return self.macd_df

    def createStoch(self):
        print(self.stoch_df.head(1))
        new_stoch_df = self.df.ta.stoch(high="high", low="low", close="close", append=False)
        self.stoch_df = pd.concat([self.stoch_df, new_stoch_df], axis=1)
        self.stoch_df.columns = ["stoch_k", "stoch_d"]
        self.stoch_df = self.stoch_df.fillna(method='bfill')
        return self.stoch_df



    

class StockPatterns:
    '''
    Class FindPatters iterates over every date creating instances of support 
    and levels and checks for predefined patterns. 
    '''

    #SwingChange is the percent range away from the current price that a previous high or low is considered a relitive high or low
    SWINGCHANGE = .07




    def __init__(self,df):
        '''
        Initializes FindPatterns object 
        @param Stock object of type Stock
        '''
        self.df = df
        self.currentLevels = []
        self.relativeHighs = []
        self.relativeLows = []
    

    def analyzePriceData(self):
        '''
        Method to loop over a stocks panda dataset containing relavent price data checking for patterns. 
        '''


        relMax = Price(0)
        relmin = Price(0)

        for index, period in self.df.iterrows(): 
            
            periodHigh = period["High"]
            periodLow = period["Low"]
            date = period.name
            

            periodHigh = Price(periodHigh,date)
            periodLow = Price(periodLow,date)
            # self.currentLevels += [Patterns.PriceLevels("resistance",periodHigh,date)]
            closestlevels = self.getClosestlevels(periodHigh)

            if periodHigh > relMax: 
                relMax = periodHigh
                relMax.setDate(date)
            
            #Considered far enough away from previus high to consider price a relative high 
            if periodLow.price < relMax.price-relMax.price*self.SWINGCHANGE: 
                self.addRelativeHigh(relMax,date)
                relMax = periodHigh
            if periodHigh == closestlevels.price:
                ##TODO check for both support and resistance 
                closestlevels.addResisTouch(date)

            if periodLow < relmin:
                relmin = periodLow
                relmin.setDate(date)

            if periodHigh.price > relmin.price+relmin.price*self.SWINGCHANGE:
                self.addRelativeLow(relmin,date)
                relmin = periodLow

            if periodLow == closestlevels.price:
                closestlevels.addSupportTouch(date)
    


    def addRelativeHigh(self,relMax,date):
        '''
        Method to add relative high to list of other relative highs. 
        If it already exists in the list relative high is removed and a levels
        object is istantiated at that price. 
        '''
        
        if (relMax in self.relativeHighs):

            repeatMax = self.relativeHighs[self.relativeHighs.index(relMax)]
            newLevel = PriceLevels("resistance", relMax,relMax.date)
            newLevel.addDate("resistance",repeatMax.date)
            self.relativeHighs.remove(relMax)
            if (newLevel not in self.currentLevels): 
                self.currentLevels += [newLevel]
        else: 
            self.relativeHighs += [relMax]
        
    def addRelativeLow(self,relmin,date):
        '''
        Method to add relative low to list of other relative lows. 
        If it already exists in the list relative low is removed and a levels
        object is istantiated at that price. 
        '''
        if (relmin in self.relativeLows):
            repeatMin = self.relativeLows[self.relativeLows.index(relmin)]
            newLevel = PriceLevels("support", relmin,relmin.date)
            newLevel.addDate("support",repeatMin.date)
            self.relativeLows.remove(relmin)
            if (newLevel not in self.currentLevels): 
                self.currentLevels += [newLevel]
        else: 
            self.relativeLows += [relmin]
    

    def getClosestlevels(self,price):
        ##TODO return support and resistance values
        difference = price; 
        closestIndex = 0; 

        if len(self.currentLevels) == 0:
            
            return PriceLevels("support", Price(0),'01-01-2001')
        
        for index,curlevels in enumerate(self.currentLevels): 
            if abs(price-curlevels.price) < difference: 
                difference = abs(price-curlevels.price)
                closestIndex = index 
        
        return self.currentLevels[closestIndex]

    def mergeLevels(self, pct_diff):
        '''
        Method to merge levels that are close to each other. 
        '''
        sortedLevels = sorted(self.currentLevels,key = lambda x: x.price)

        newLevels = []
        
        for index,level in enumerate(sortedLevels):
            if index == 0:
                continue
            lastLevelVal = sortedLevels[index-1].price.price
            curLevelVal = level.price.price
            difference = abs(lastLevelVal - curLevelVal)/lastLevelVal

            if difference < pct_diff:
                newLevels.append(sortedLevels[index-1].mergeLevels(level))
        
        self.currentLevels = newLevels
                
        
    
    def returnStock(self):
        self.curStock.levels = self.currentLevels
        return self.curStock


class Price: 
    '''
    Price class holds relevant information to a certain price for a stock.
    '''
    EPSILON = .01

    
    def __init__(self,price,date = None): 
        '''
        @param price A numeric price value
        '''
        self.price = price 
        self.date = date
    

    def setDate(self,date):
        '''
        Add date value to list for multiple occurances of self price
        '''
        self.date = date
    
    def __eq__(self,otherPrice):
        '''
        Overrides equality to allow prices to be considered equal if they are within percent range EPSILON
        '''
        rangeOfEquality = self.price* self.EPSILON
        if otherPrice.price >= self.price - rangeOfEquality and otherPrice.price <= self.price+rangeOfEquality:
            return True
        else:
            return False
    
    def __sub__(self,other):
        return Price(self.price - other.price)
    
    def __add__(self,other):
        return Price(self.price + other.price)
    def __truediv__(self,other):   
        return Price(self.price/other.price)


    def __gt__(self,other):
        return (self.price > other.price)
    
    def __lt__(self,other):
        return (self.price < other.price)

    def __repr__(self):
        return str(self.price)
    
    def __abs__(self):
        if self.price > 0: 
            return Price(self.price)
        else:
            return Price(self.price*-1)
        


class PriceLevels: 
    
    def __init__(self,levelType, price, date):
        self.price = price 
        if levelType == "support":
            self.supportDates = [date]
            self.resistanceDates = []
            self.supportTouches = 2; 
            self.resistanceTouches = 0; 
        else: 
            self.supportDates = []
            self.resistanceDates = [date]
            self.supportTouches = 0
            self.resistanceTouches = 2 
    
    def addResisTouch(self,date):
        self.resistanceDates+=[date]
        self.resistanceTouches += 1
    
    def addSupportTouch(self,date):
        self.supportDates+=[date]
        self.supportTouches += 1
    
    def addDate(self,levelType, date,):
        if levelType == "support":
            self.supportDates += [date]
        else:
            self.resistanceDates += [date]

    
    def __eq__(self,other):
        return self.price == other.price

    def __repr__(self):
        ##TODO update repr for support and resistance 
        resisString = str(self.price) + " "
        resisString += str(self.resistanceTouches)
        # for date in self.dates:
        #     resisString += date.strftime("%Y-%m-%d") + " "
        return resisString
    
    def getTotalTouches(self):
        return self.resistanceTouches + self.supportTouches

    def mergeLevels(self,other):
        self.resistanceTouches += other.resistanceTouches
        self.supportTouches += other.supportTouches
        self.supportDates += other.supportDates
        self.resistanceDates += other.resistanceDates

        self.price = (self.price + other.price)/Price(2)

        return self 



