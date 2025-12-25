using Microsoft.ML.Data;

namespace SalesForecasting.Core.Models;

public class SalesPredictionOutput
{
    [ColumnName("Score")]
    public float PredictedSales { get; set; }
    
    // Alias dla kompatybilnoœci
    public float Score => PredictedSales;
}