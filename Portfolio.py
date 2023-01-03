"""
This class defines an object (Portfolio) for keeping track of our current investments and profits/losses for each currency pair.
The class provides five methods for each currency pair either buy currency units, sell currency units, calculate the
current profit/loss, calculate the balance of the pair and close the trade.

Assumption for HWK 3: there is always enough money of the needed currency to do the desired transaction
"""


class Portfolio(object):
    def __init__(self, from_, to, client):
        self.client = client
        self.from_ = from_
        self.to = to

        # Keep track of the average price we bought or sold at to calculate the exact profits
        self.average_buy_sell_rate = 0
        self.nb_curr_units = 0
        self.dollars_spend = 0

    """
    This defines a function to buy the base currency of a certain currency pair, can be used for LONG positions.

    Arguments:
        - amount: Amount of currency units that the trader wants to buy of the base currency.

    Returns:
        - None
    """

    def buy_curr(self, amount):

        # A ticker is a unique number set in the trading platform for every operation
        ticker = "C:" + self.from_ + self.to
        last_quote = self.client.get_last_quote(ticker)

        # Use the ask price as buy price
        exchange_rate = last_quote.ask_price

        # Keep track of the average buy and sell rate to calculate the exact profits
        self.average_buy_sell_rate = (self.average_buy_sell_rate * self.nb_curr_units + exchange_rate * amount) / (
                self.nb_curr_units + amount)

        self.nb_curr_units += amount

        # Keep track of the amount of dollars spend for the balance
        if self.from_ == "USD":
            self.dollars_spend = self.dollars_spend + amount

        else:
            self.dollars_spend = self.dollars_spend + amount * exchange_rate

        print(
            f"[BUY, {self.from_}_{self.to}] Bought {amount} currency units of the target currency {self.from_} at rate: {exchange_rate}.")

    """
    This defines a function to sell the base currency of a certain currency pair, can be used for SHORT positions.

    Arguments:
        - amount: Amount of currency units that the trader wants to sell of the base currency.

    Returns:
        - None
    """

    def sell_curr(self, amount):

        # A ticker is a unique number set in the trading platform for every operation
        ticker = "C:" + self.from_ + self.to
        last_quote = self.client.get_last_quote(ticker)

        # Use the bid price as sell price
        exchange_rate = last_quote.bid_price

        # Keep track of the average buy and sell rate to calculate the exact profits
        self.average_buy_sell_rate = (self.average_buy_sell_rate * self.nb_curr_units + exchange_rate * amount) / (
                self.nb_curr_units + amount)

        self.nb_curr_units += amount

        if self.from_ == "USD":
            self.dollars_spend = self.dollars_spend - amount

        else:
            self.dollars_spend = self.dollars_spend - amount * exchange_rate

        print(
            f"[SELL, {self.from_}_{self.to}] Sold {amount} currency units of the target currency {self.from_} at rate: {exchange_rate}.")

    """
    This function calculates the exact profits/losses for a trade, expressed it in the quote currency and converts it to USD. 
    (To realise the profits, the trade has to be closed)

    Note that for short positions a negative profit means we make money

    I used this source (https://www.investopedia.com/articles/forex/12/calculating-profits-and-losses-of-forex-trades.asp)
    for calculating the profits/losses.

        Arguments:
        - position_type: String, that is either "SHORT" or "LONG" depending on the type of the position

    Returns:
        - Profit, float number representing the profit/loss
    """

    def calculate_profit(self, position_type):
        ticker = "C:" + self.from_ + self.to
        last_quote = self.client.get_last_quote(ticker)

        # Calculate exact profits
        if position_type == "LONG":

            current_rate = sell_price = last_quote.bid_price
            diff_rates = (sell_price - self.average_buy_sell_rate)
            profit = self.nb_curr_units * diff_rates

        elif position_type == "SHORT":

            current_rate = buy_price = last_quote.ask_price
            diff_rates = (buy_price - self.average_buy_sell_rate)
            profit = self.nb_curr_units * diff_rates

        else:
            raise ValueError("Not a valid position type")

        # Convert profits to USD if the quote currency is not USD
        if self.to == "USD":
            return profit

        else:
            profit_USD = profit / current_rate
            return profit_USD

    """
    This function converts the balance of a currency pair to US dollars.

    Arguments:
        - position_type: String, that is either "SHORT" or "LONG" depending on the type of the position

    Returns:
        - Balance, float number representing the balance of a currency pair
    """

    def convert_balance(self, position_type):

        # Assumption: each currency pair has either USD as quote currency or as base currency. Code can be extended in the
        # future to also support cross pair currencies.

        # Gives profit in the USD currency
        profit = self.calculate_profit(position_type)

        balance = self.dollars_spend + profit

        return balance

    """
    This function closes the trade by selling/buying back the base currency. If the trader wishes to close his/her trade in
    this manner, this function can simply be called after the trade. Returns the realised profit/loss

    Arguments:
        - position_type: String, that is either "short" or "long" depending on the type of the position

    Returns:
        - Profit, float that represents the finalized profit
    """

    def close_trade(self, position_type):

        if position_type == "LONG":

            realized_profit = self.calculate_profit("LONG")
            self.sell_curr(self.nb_curr_units)

            self.nb_curr_units = 0

            return realized_profit


        elif position_type == "SHORT":

            realized_profit = self.calculate_profit("SHORT")
            self.buy_curr(self.nb_curr_units)

            self.nb_curr_units = 0

            return realized_profit

        else:
            raise ValueError("Not a valid position type")

    """
        Still do this
    """

    def exact_return(self, position_type):

        # Gives profit in the USD currency
        profit = self.calculate_profit(position_type)

        exact_return = profit / self.dollars_spend

        return exact_return
