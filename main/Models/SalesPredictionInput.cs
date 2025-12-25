using Microsoft.ML.Data;

namespace SalesForecasting.Core.Models;

public class SalesPredictionInput
{
    [LoadColumn(0)]
    public float WeekNumber { get; set; }
    
    [LoadColumn(1)]
    public float Month { get; set; }
    
    [LoadColumn(2)]
    public float Quarter { get; set; }
    
    [LoadColumn(3)]
    public float IsBlackFridayWeek { get; set; }
    
    [LoadColumn(4)]
    public float IsHolidaySeason { get; set; }
    
    [LoadColumn(5)]
    public float Lag1 { get; set; }
    
    [LoadColumn(6)]
    public float Lag2 { get; set; }
    
    [LoadColumn(7)]
    public float Lag3 { get; set; }
    
    [LoadColumn(8)]
    public float Lag4 { get; set; }
    
    [LoadColumn(9)]
    public float RollingAvg4Weeks { get; set; }
    
    [LoadColumn(10)]
    public float Trend { get; set; }
    
    [LoadColumn(11)]
    public float CategoryHistoricalAvg { get; set; }

    [LoadColumn(12)]
    public string ProductCategory { get; set; } = string.Empty;

    [LoadColumn(13)]
    [ColumnName("Label")]
    public float ActualSales { get; set; }


}