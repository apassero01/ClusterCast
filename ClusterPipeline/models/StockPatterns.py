
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

        for index, period in self.priceData.iterrows(): 
            
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