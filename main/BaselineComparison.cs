using SalesForecasting.Core.Models;
using Serilog;

namespace SalesForecasting.ML;

/// <summary>
/// Porównanie modelu ML z prostym baseline (naive forecast)
/// </summary>
public class BaselineComparison
{
    private readonly ILogger _logger;

    public BaselineComparison(ILogger logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Oblicza metryki dla prostego baseline: œrednia z ostatnich 4 tygodni
    /// </summary>
    public ModelMetrics EvaluateNaiveBaseline(List<SalesPredictionInput> testData)
    {
        var predictions = testData.Select(x => new
        {
            Actual = (double)x.ActualSales,
            // Naive: predykcja = œrednia z ostatnich 4 tygodni
            Predicted = (double)x.RollingAvg4Weeks
        }).ToList();

        var mae = predictions.Average(p => Math.Abs(p.Actual - p.Predicted));
        var rmse = Math.Sqrt(predictions.Average(p => Math.Pow(p.Actual - p.Predicted, 2)));

        var meanActual = predictions.Average(p => p.Actual);
        var ssTot = predictions.Sum(p => Math.Pow(p.Actual - meanActual, 2));
        var ssRes = predictions.Sum(p => Math.Pow(p.Actual - p.Predicted, 2));
        var rSquared = 1 - (ssRes / ssTot);

        var mape = predictions.Where(p => p.Actual >= 3)
            .Average(p => Math.Abs((p.Actual - p.Predicted) / p.Actual)) * 100;

        return new ModelMetrics
        {
            Category = "Naive Baseline",
            RSquared = rSquared,
            MAE = mae,
            RMSE = rmse,
            MAPE = mape,
            TestSamples = testData.Count,
            TrainedAt = DateTime.UtcNow
        };
    }
}