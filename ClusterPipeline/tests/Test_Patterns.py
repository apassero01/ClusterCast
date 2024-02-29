from django.test import TestCase
import pandas as pd
import numpy as np
from ClusterPipeline.models.StockPatterns import MovingAverageFactory, BandFactory, MomentumFactory, OHLCVFactory
import yfinance as yf   


class TestOHLCVFactory(TestCase):
    def setUp(self):
        self.df = yf.download('AAPL', start='2020-01-01')
        self.df = self.df.rename(columns={'Close': 'close'})
        self.df = self.df.rename(columns={'Volume': 'volume'})
        self.df = self.df.rename(columns={'High': 'high'})
        self.df = self.df.rename(columns={'Low': 'low'})
        self.df = self.df.rename(columns={'Open': 'open'})
        self.factory = OHLCVFactory(self.df)
    
    def test_createPctChg(self):
        pctChg_df = self.factory.createPctChg()

        for col in self.df.columns:
            self.assertTrue(f'pctChg{col}' in pctChg_df.columns)
        
        self.assertTrue(len(pctChg_df.columns) == len(self.df.columns))

        self.assertFalse(pctChg_df.isnull().values.any())  # Check for NaNs

        self.assertAlmostEqual(pctChg_df['pctChgclose'].iloc[20], (self.df['close'].iloc[20] - self.df['close'].iloc[19]) / self.df['close'].iloc[19]*100)  # Check computed value

    def test_createIntraDay(self):
        intraday_df = self.factory.createIntraDay()

        new_cols = ['opHi', 'opLo', 'hiCl', 'loCl', 'hiLo', 'opCl', 'pctChgClOp', 'pctChgClLo', 'pctChgClHi']

        self.assertTrue(all([col in intraday_df.columns for col in new_cols]))
        self.assertTrue(len(intraday_df.columns) == len(new_cols))

        self.assertFalse(intraday_df.isnull().values.any())  # Check for NaNs

        self.assertAlmostEqual(intraday_df['opHi'].iloc[20], (self.df['high'].iloc[20] - self.df['open'].iloc[20]) / self.df['open'].iloc[20] * 100)  # Check computed value

class TestMovingAverageFactory(TestCase):
    def setUp(self):
        self.df = yf.download('AAPL', start='2020-01-01')
        self.df = self.df.rename(columns={'Close': 'close'})
        self.df = self.df.rename(columns={'Volume': 'volume'})
        self.factory = MovingAverageFactory(self.df)

    def test_createSMA(self):
        periods = [5, 10, 20, 50, 100, 200]
        sma_df = self.factory.createSMA(periods)
        for period in periods:
            self.assertTrue(f'sma{period}' in sma_df.columns)
        self.assertTrue(len(sma_df.columns) == len(periods))
        self.assertFalse(sma_df.isnull().values.any())  # Check for NaNs
        self.assertAlmostEqual(sma_df['sma5'].iloc[20], self.df['close'].iloc[16:21].mean())  # Check computed value

    def test_createEMA(self):
        periods = [5, 10, 20, 50, 100, 200]
        ema_df = self.factory.createEMA(periods=periods)
        for period in periods:
            self.assertTrue(f'ema{period}' in ema_df.columns)
        self.assertTrue(len(ema_df.columns) == len(periods))

        self.assertFalse(ema_df.isnull().values.any())  # Check for NaNs
        # Add check for computed value...
        self.assertAlmostEqual(ema_df['ema5'].iloc[20], self.df['close'].iloc[16:21].ewm(span=5, adjust=True).mean().iloc[-1], places=1)  # Check computed value

    def test_createSMAVolume(self):
        periods = [5, 10, 20, 50, 100, 200]
        smaVol_df = self.factory.createSMAVolume(periods)
        for period in periods:
            self.assertTrue(f'smaVol{period}' in smaVol_df.columns)

        self.assertTrue(len(smaVol_df.columns) == len(periods))

        self.assertFalse(smaVol_df.isnull().values.any())  # Check for NaNs
        # Add check for computed value...
        self.assertAlmostEqual(smaVol_df['smaVol5'].iloc[20], self.df['volume'].iloc[16:21].mean())  # Check computed value

    def test_createSMAPctDiff(self):
        periods = [5, 10, 20, 50, 100, 200]
        sma_df = self.factory.createSMA(periods)
        smaPctDiff_df = self.factory.createSMAPctDiff()
        for period in periods:
            self.assertTrue(f'pctDiff+sma{period}_close' in smaPctDiff_df.columns)
        
        self.assertTrue(len(smaPctDiff_df.columns) == len(periods) + len(periods)*(len(periods)-1)//2)
        self.assertFalse(smaPctDiff_df.isnull().values.any())  # Check for NaNs
        # Add check for computed value...
        self.assertAlmostEqual(smaPctDiff_df['pctDiff+sma5_close'].iloc[20], (self.df['close'].iloc[20] - sma_df['sma5'].iloc[20]) / sma_df['sma5'].iloc[20]*100)  # Check computed value

    def test_createEMAPctDiff(self):    
        periods = [5, 10, 20, 50, 100, 200]
        ema_df = self.factory.createEMA(periods)
        for period in periods:
            self.assertTrue(f'pctDiff+ema{period}_close' in self.factory.createEMAPctDiff().columns)

        self.assertFalse(self.factory.createEMAPctDiff().isnull().values.any())  # Check for NaNs
        self.assertTrue(len(self.factory.createEMAPctDiff().columns) == len(periods) + len(periods)*(len(periods)-1)//2)
        emaPctDiff_df = self.factory.createEMAPctDiff()
        self.assertTrue('pctDiff+ema5_close' in emaPctDiff_df.columns)
        self.assertFalse(emaPctDiff_df.isnull().values.any())  # Check for NaNs
        # Add check for computed value...
        self.assertAlmostEqual(emaPctDiff_df['pctDiff+ema5_close'].iloc[20], (self.df['close'].iloc[20] - ema_df['ema5'].iloc[20]) / ema_df['ema5'].iloc[20]*100)  # Check computed value

    def test_createSMAPctDiffVol(self):
        periods = [5, 10, 20, 50, 100, 200]
        smaVol_df = self.factory.createSMAVolume(periods)
        
        smaPctDiffVol_df = self.factory.createSMAPctDiffVol()
        for period in periods:
            self.assertTrue(f'pctDiff+smaVol{period}_volume' in smaPctDiffVol_df.columns)
        self.assertTrue(len(smaPctDiffVol_df.columns) == len(periods) + len(periods)*(len(periods)-1)//2)
        self.assertFalse(smaPctDiffVol_df.isnull().values.any())  # Check for NaNs
        # Add check for computed value...
        self.assertAlmostEqual(smaPctDiffVol_df['pctDiff+smaVol5_volume'].iloc[20], (self.df['volume'].iloc[20] - smaVol_df['smaVol5'].iloc[20]) / smaVol_df['smaVol5'].iloc[20]*100)  # Check computed value

    def test_createSMADerivative(self):
        periods = [5, 10, 20, 50, 100, 200]
        sma_df = self.factory.createSMA(periods)
        smaDerivative_df = self.factory.createSMADerivative()
        for period in periods:
            self.assertTrue(f'deriv+sma{period}' in smaDerivative_df.columns)
        self.assertTrue(len(smaDerivative_df.columns) == len(periods))

        self.assertFalse(smaDerivative_df.isnull().values.any())  # Check for NaNs
        # Add check for computed value...
        self.assertAlmostEqual(smaDerivative_df['deriv+sma5'].iloc[40], (sma_df['sma5'].iloc[40] - sma_df['sma5'].iloc[35]) / 5)

    def test_createEMADerivative(self):
        periods = [5, 10, 20, 50, 100, 200]
        ema_df = self.factory.createEMA(periods)
        emaDerivative_df = self.factory.createEMADerivative()
        for period in periods:
            self.assertTrue(f'deriv+ema{period}' in emaDerivative_df.columns)
        self.assertTrue(len(emaDerivative_df.columns) == len(periods))
        self.assertFalse(emaDerivative_df.isnull().values.any())  # Check for NaNs
        # Add check for computed value...
        self.assertAlmostEqual(emaDerivative_df['deriv+ema5'].iloc[20], (ema_df['ema5'].iloc[20] - ema_df['ema5'].iloc[15]) / 5)

    def test_createSMADerivativeVol(self):
        periods = [5, 10, 20, 50, 100, 200]
        smaVol_df = self.factory.createSMAVolume(periods)
        smaDerivativeVol_df = self.factory.createSMADerivativeVol()
        for period in periods:
            self.assertTrue(f'deriv+smaVol{period}' in smaDerivativeVol_df.columns)
        self.assertTrue(len(smaDerivativeVol_df.columns) == len(periods))
        self.assertFalse(smaDerivativeVol_df.isnull().values.any())  # Check for NaNs
        # Add check for computed value...
        self.assertAlmostEqual(smaDerivativeVol_df['deriv+smaVol5'].iloc[20], (smaVol_df['smaVol5'].iloc[20] - smaVol_df['smaVol5'].iloc[15]) / 5)


class TestBandFactory(TestCase):
    def setUp(self):
        self.df = yf.download('AAPL', start='2020-01-01')
        self.df = self.df.rename(columns={'Close': 'close'})
        self.df = self.df.rename(columns={'Volume': 'volume'})
        self.factory = BandFactory(self.df)

    def test_createBB(self):
        periods = [5, 10, 20, 50, 100, 200]
        bb_df = self.factory.createBB(periods)
        for period in periods:
            self.assertTrue(f'bb_high{period}' in bb_df.columns)
            self.assertTrue(f'bb_low{period}' in bb_df.columns)
        self.assertTrue(len(bb_df.columns) == len(periods)*2)

        self.assertFalse(bb_df.isnull().values.any())  # Check for NaNs


    def test_createBBPctDiff(self):
        periods = [5, 10, 20, 50, 100, 200]
        bb_df = self.factory.createBB(periods)
        bbPctDiff_df = self.factory.createBBPctDiff()
        for period in periods:
            self.assertTrue(f'pctDiff+bb_high_low{period}' in bbPctDiff_df.columns)
            self.assertTrue(f'pctDiff+bb_high_close{period}' in bbPctDiff_df.columns)
            self.assertTrue(f'pctDiff+bb_low_close{period}' in bbPctDiff_df.columns)
            self.assertTrue(f'bb_indicator{period}' in bbPctDiff_df.columns)
        self.assertTrue(len(bbPctDiff_df.columns) == len(periods)*4)

        self.assertFalse(bbPctDiff_df.isnull().values.any())  # Check for NaNs
        # Add check for computed value...
        self.assertAlmostEqual(bbPctDiff_df['pctDiff+bb_high_low5'].iloc[20], (bb_df['bb_high5'].iloc[20] - bb_df['bb_low5'].iloc[20]) / bb_df['bb_low5'].iloc[20]*100)


class TestMomentumFactory(TestCase):
    def setUp(self):
        self.df = yf.download('AAPL', start='2020-01-01')
        self.df = self.df.rename(columns={'Close': 'close'})
        self.df = self.df.rename(columns={'Volume': 'volume'})
        self.factory = MomentumFactory(self.df)

    def test_createRSI(self):
        periods = [5, 10, 20, 50, 100, 200]
        rsi_df = self.factory.createRSI(periods)
        for period in periods:
            self.assertTrue(f'rsi{period}' in rsi_df.columns)
        self.assertTrue(len(rsi_df.columns) == len(periods))

        self.assertFalse(rsi_df.isnull().values.any())
    
    def test_createStoch(self):

        stoch_df = self.factory.createStoch()

        columns = ['stoch_k', 'stoch_d']

        self.assertTrue(all([col in stoch_df.columns for col in columns]))

        self.assertFalse(stoch_df.isnull().values.any())
    
    def test_createMACD(self):

        macd_df = self.factory.createMACD()

        columns = ["macd", "macd_signal", "macd_diff"]

        self.assertTrue(all([col in macd_df.columns for col in columns]))

        self.assertFalse(macd_df.isnull().values.any())

        