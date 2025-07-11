//+------------------------------------------------------------------+
//|                                                 TestEASample.mq4 |
//|                        Copyright 2024, QNTI Trading Systems Inc |
//|                                             https://www.qnti.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, QNTI Trading Systems Inc"
#property link      "https://www.qnti.com"
#property version   "1.00"
#property strict

//--- input parameters
extern double LotSize = 0.01;
extern int StopLoss = 50;
extern int TakeProfit = 100;
extern int MagicNumber = 123456;
extern string TradeComment = "QNTI Test EA";

//--- global variables
int ticket;
double point;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    point = Point;
    if(Digits == 5 || Digits == 3) point *= 10;
    
    Print("QNTI Test EA initialized successfully");
    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    Print("QNTI Test EA deinitialized");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    // Simple moving average crossover strategy
    double ma_fast = iMA(NULL, 0, 10, 0, MODE_SMA, PRICE_CLOSE, 1);
    double ma_slow = iMA(NULL, 0, 20, 0, MODE_SMA, PRICE_CLOSE, 1);
    double ma_fast_prev = iMA(NULL, 0, 10, 0, MODE_SMA, PRICE_CLOSE, 2);
    double ma_slow_prev = iMA(NULL, 0, 20, 0, MODE_SMA, PRICE_CLOSE, 2);
    
    // Check for buy signal
    if(ma_fast > ma_slow && ma_fast_prev <= ma_slow_prev && OrdersTotal() == 0)
    {
        ticket = OrderSend(Symbol(), OP_BUY, LotSize, Ask, 3, 
                          Ask - StopLoss * point, Ask + TakeProfit * point, 
                          TradeComment, MagicNumber, 0, clrBlue);
        
        if(ticket > 0)
            Print("Buy order opened: ", ticket);
        else
            Print("Failed to open buy order: ", GetLastError());
    }
    
    // Check for sell signal
    if(ma_fast < ma_slow && ma_fast_prev >= ma_slow_prev && OrdersTotal() == 0)
    {
        ticket = OrderSend(Symbol(), OP_SELL, LotSize, Bid, 3, 
                          Bid + StopLoss * point, Bid - TakeProfit * point, 
                          TradeComment, MagicNumber, 0, clrRed);
        
        if(ticket > 0)
            Print("Sell order opened: ", ticket);
        else
            Print("Failed to open sell order: ", GetLastError());
    }
}

//+------------------------------------------------------------------+
//| Custom function to check trading conditions                      |
//+------------------------------------------------------------------+
bool IsTradeAllowed()
{
    return(IsTradeAllowed() && !IsTradeContextBusy());
}