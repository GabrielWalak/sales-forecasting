using Microsoft.ML;
using SalesForecasting.Core.Models;
using Serilog;

namespace SalesForecasting.ML;

/// <summary>
/// Analiza prognoz dla różnych horyzontów czasowych (1 tydzień, 2 tygodnie, miesiąc)
/// </summary>
public class ForecastHorizonAnalyzer
{
    private readonly MLContext _mlContext;
    private readonly ILogger _logger;

    public ForecastHorizonAnalyzer(ILogger logger)
    {
        _mlContext = new MLContext(seed: 42);
        _logger = logger;
    }

    public List<HorizonAnalysisResult> AnalyzeHorizons(
        ITransformer model,
        List<SalesPredictionInput> testData,
        int[] horizonsInWeeks = null!)
    {
        horizonsInWeeks ??= [1, 2, 4, 8];
        
        var results = new List<HorizonAnalysisResult>();

        Console.WriteLine("\n" + new string('═', 80));
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("📅 ANALIZA DOKŁADNOŚCI DLA RÓŻNYCH HORYZONTÓW CZASOWYCH");
        Console.ResetColor();
        Console.WriteLine(new string('═', 80));

        foreach (var horizon in horizonsInWeeks)
        {
            var horizonResult = EvaluateHorizon(model, testData, horizon);
            results.Add(horizonResult);
            PrintHorizonResult(horizonResult);
        }

        PrintHorizonSummary(results);
        return results;
    }

    private HorizonAnalysisResult EvaluateHorizon(
        ITransformer model,
        List<SalesPredictionInput> testData,
        int horizonWeeks)
    {
        _logger.Information("Evaluating forecast horizon: {Horizon} weeks", horizonWeeks);

        var predictions = new List<(double Actual, double Predicted)>();
        var orderedData = testData.OrderBy(x => x.WeekNumber).ToList();
        
        // Utwórz prediction engine raz (wydajność)
        var predEngine = _mlContext.Model.CreatePredictionEngine<SalesPredictionInput, SalesPredictionOutput>(model);
        
        for (int i = horizonWeeks - 1; i < orderedData.Count; i++)
        {
            var currentSample = orderedData[i];
            var modifiedSample = CreateHorizonAdjustedSample(currentSample, horizonWeeks, orderedData, i);
            
            var pred = predEngine.Predict(modifiedSample);
            
            // ✅ POPRAWKA: Używamy PredictedSales zamiast Score
            var predictedValue = Math.Exp(pred.PredictedSales) - 1;
            var actualValue = (double)currentSample.ActualSales;
            
            predictions.Add((actualValue, Math.Max(0, predictedValue)));
        }

        if (!predictions.Any())
        {
            return new HorizonAnalysisResult
            {
                HorizonWeeks = horizonWeeks,
                HorizonName = GetHorizonName(horizonWeeks),
                SampleCount = 0
            };
        }

        // Oblicz metryki
        var mae = predictions.Average(p => Math.Abs(p.Actual - p.Predicted));
        var rmse = Math.Sqrt(predictions.Average(p => Math.Pow(p.Actual - p.Predicted, 2)));
        
        var meanActual = predictions.Average(p => p.Actual);
        var ssTot = predictions.Sum(p => Math.Pow(p.Actual - meanActual, 2));
        var ssRes = predictions.Sum(p => Math.Pow(p.Actual - p.Predicted, 2));
        var rSquared = ssTot > 0 ? 1 - (ssRes / ssTot) : 0;
        
        var mape = predictions
            .Where(p => p.Actual > 0)
            .DefaultIfEmpty((1, 1))
            .Average(p => Math.Abs((p.Actual - p.Predicted) / p.Actual)) * 100;

        return new HorizonAnalysisResult
        {
            HorizonWeeks = horizonWeeks,
            HorizonName = GetHorizonName(horizonWeeks),
            SampleCount = predictions.Count,
            RSquared = rSquared,
            MAE = mae,
            RMSE = rmse,
            MAPE = mape,
            AccuracyDegradation = 0
        };
    }

    private SalesPredictionInput CreateHorizonAdjustedSample(
        SalesPredictionInput original,
        int horizonWeeks,
        List<SalesPredictionInput> allData,
        int currentIndex)
    {
        var adjusted = new SalesPredictionInput
        {
            WeekNumber = original.WeekNumber,
            Month = original.Month,
            Quarter = original.Quarter,
            IsBlackFridayWeek = original.IsBlackFridayWeek,
            IsHolidaySeason = original.IsHolidaySeason,
            ProductCategory = original.ProductCategory,
            CategoryHistoricalAvg = original.CategoryHistoricalAvg,
            Trend = original.Trend,
            ActualSales = original.ActualSales
        };

        switch (horizonWeeks)
        {
            case 1:
                adjusted.Lag1 = original.Lag1;
                adjusted.Lag2 = original.Lag2;
                adjusted.Lag3 = original.Lag3;
                adjusted.Lag4 = original.Lag4;
                adjusted.RollingAvg4Weeks = original.RollingAvg4Weeks;
                break;
            case 2:
                adjusted.Lag1 = original.Lag2;
                adjusted.Lag2 = original.Lag3;
                adjusted.Lag3 = original.Lag4;
                adjusted.Lag4 = currentIndex >= 5 ? allData[currentIndex - 5].ActualSales : original.Lag4;
                adjusted.RollingAvg4Weeks = (adjusted.Lag1 + adjusted.Lag2 + adjusted.Lag3 + adjusted.Lag4) / 4;
                break;
            case 4:
                adjusted.Lag1 = original.Lag4;
                adjusted.Lag2 = currentIndex >= 5 ? allData[currentIndex - 5].ActualSales : original.Lag4;
                adjusted.Lag3 = currentIndex >= 6 ? allData[currentIndex - 6].ActualSales : original.Lag4;
                adjusted.Lag4 = currentIndex >= 7 ? allData[currentIndex - 7].ActualSales : original.Lag4;
                adjusted.RollingAvg4Weeks = (adjusted.Lag1 + adjusted.Lag2 + adjusted.Lag3 + adjusted.Lag4) / 4;
                break;
            default:
                var baseIndex = Math.Max(0, currentIndex - horizonWeeks);
                adjusted.Lag1 = allData[baseIndex].ActualSales;
                adjusted.Lag2 = baseIndex > 0 ? allData[baseIndex - 1].ActualSales : adjusted.Lag1;
                adjusted.Lag3 = baseIndex > 1 ? allData[baseIndex - 2].ActualSales : adjusted.Lag2;
                adjusted.Lag4 = baseIndex > 2 ? allData[baseIndex - 3].ActualSales : adjusted.Lag3;
                adjusted.RollingAvg4Weeks = (adjusted.Lag1 + adjusted.Lag2 + adjusted.Lag3 + adjusted.Lag4) / 4;
                break;
        }

        return adjusted;
    }

    private string GetHorizonName(int weeks) => weeks switch
    {
        1 => "1 tydzień",
        2 => "2 tygodnie",
        4 => "1 miesiąc",
        8 => "2 miesiące",
        12 => "Kwartał",
        _ => $"{weeks} tygodni"
    };

    private void PrintHorizonResult(HorizonAnalysisResult result)
    {
        Console.WriteLine($"\n📅 Horyzont: {result.HorizonName} ({result.HorizonWeeks} tyg.)");
        Console.WriteLine($"   Próbek: {result.SampleCount}");
        
        Console.ForegroundColor = result.RSquared > 0.7 ? ConsoleColor.Green : 
                                  result.RSquared > 0.5 ? ConsoleColor.Yellow : ConsoleColor.Red;
        Console.WriteLine($"   R²:   {result.RSquared:F4}");
        Console.ResetColor();
        
        Console.WriteLine($"   MAE:  {result.MAE:F2}");
        Console.WriteLine($"   RMSE: {result.RMSE:F2}");
        Console.WriteLine($"   MAPE: {result.MAPE:F2}%");
    }

    private void PrintHorizonSummary(List<HorizonAnalysisResult> results)
    {
        if (!results.Any()) return;
        
        Console.WriteLine("\n" + new string('═', 80));
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("📊 PODSUMOWANIE HORYZONTÓW CZASOWYCH:");
        Console.ResetColor();
        
        var baseline = results.First().RSquared;
        
        foreach (var r in results)
        {
            r.AccuracyDegradation = baseline > 0 ? ((baseline - r.RSquared) / baseline) * 100 : 0;
        }

        Console.WriteLine($"\n{"Horyzont",-15} {"R²",-10} {"MAE",-12} {"MAPE",-10} {"Degradacja",-12}");
        Console.WriteLine(new string('-', 60));
        
        foreach (var r in results)
        {
            Console.ForegroundColor = r.AccuracyDegradation < 10 ? ConsoleColor.Green :
                                      r.AccuracyDegradation < 30 ? ConsoleColor.Yellow : ConsoleColor.Red;
            Console.WriteLine($"{r.HorizonName,-15} {r.RSquared,-10:F4} {r.MAE,-12:F2} {r.MAPE,-10:F2}% {r.AccuracyDegradation,-12:F1}%");
            Console.ResetColor();
        }

        Console.WriteLine("\n💡 WNIOSKI:");
        var bestHorizon = results.OrderByDescending(r => r.RSquared).First();
        var maxReliableHorizon = results.Where(r => r.RSquared > 0.5).OrderByDescending(r => r.HorizonWeeks).FirstOrDefault();
        
        Console.WriteLine($"   • Najlepsza dokładność: {bestHorizon.HorizonName} (R²={bestHorizon.RSquared:F4})");
        if (maxReliableHorizon != null)
            Console.WriteLine($"   • Maksymalny wiarygodny horyzont (R²>0.5): {maxReliableHorizon.HorizonName}");
    }
}

public class HorizonAnalysisResult
{
    public int HorizonWeeks { get; set; }
    public string HorizonName { get; set; } = string.Empty;
    public int SampleCount { get; set; }
    public double RSquared { get; set; }
    public double MAE { get; set; }
    public double RMSE { get; set; }
    public double MAPE { get; set; }
    public double AccuracyDegradation { get; set; }
}