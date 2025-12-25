using Microsoft.ML;
using SalesForecasting.Core.Models;
using Serilog;
using System.Text.Json;

namespace SalesForecasting.ML;

/// <summary>
/// Generator raportów wizualizacyjnych dla modelu ML
/// </summary>
public class VisualizationReportGenerator
{
    private readonly ILogger _logger;
    private readonly string _modelsDirectory;

    public VisualizationReportGenerator(ILogger logger, string modelsDirectory)
    {
        _logger = logger;
        _modelsDirectory = modelsDirectory;
    }

    /// <summary>
    /// Generuje wszystkie wykresy i raporty wizualizacyjne
    /// </summary>
    public void GenerateAllVisualizations(
        string modelPath,
        string preprocessedDataPath,
        DateTime trainEndDate,
        DateTime testStartDate)
    {
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("??? GENEROWANIE WYKRESÓW ???");
        Console.ResetColor();

        _logger.Information("Starting visualization generation");

        try
        {
            if (!File.Exists(modelPath))
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("? B£¥D: Brak modelu. Najpierw wykonaj opcjê 2 (Trenuj model).");
                Console.ResetColor();
                return;
            }

            Console.WriteLine("? £adowanie modelu i danych...");
            var mlContext = new MLContext(seed: 42);
            var model = mlContext.Model.Load(modelPath, out _);

            if (!File.Exists(preprocessedDataPath))
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("? B£¥D: Brak danych. Najpierw wykonaj opcjê 1.");
                Console.ResetColor();
                return;
            }

            var json = File.ReadAllText(preprocessedDataPath);
            var weeklyData = JsonSerializer.Deserialize<List<WeeklySalesData>>(json);

            if (weeklyData == null || !weeklyData.Any())
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("? B£¥D: Nie mo¿na wczytaæ danych");
                Console.ResetColor();
                return;
            }

            var processor = new SalesForecasting.Data.OlistDataProcessor(_logger);
            var (train, test) = processor.SplitGlobalByDate(weeklyData, trainEndDate, testStartDate);

            Console.WriteLine($"? Wczytano: {test.Count} próbek testowych\n");

            // 1. PREDICTED VS. ACTUAL SCATTER PLOT
            Console.WriteLine("? [1/5] Generowanie wykresu Predicted vs. Actual...");
            var (actual, predicted) = GetPredictions(mlContext, model, test);
            
            var visualizer = new VisualizationGenerator(_logger);
            var plotPath1 = Path.Combine(_modelsDirectory, "01_predicted_vs_actual.png");
            visualizer.GeneratePredictedVsActualPlot(actual, predicted, plotPath1);
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"   ? Zapisano: {plotPath1}");
            Console.ResetColor();

            // 2. FEATURE IMPORTANCE BAR CHART
            var featureImportancePath = Path.Combine(_modelsDirectory, "feature_importance.json");
            if (File.Exists(featureImportancePath))
            {
                Console.WriteLine("? [2/5] Generowanie wykresu Feature Importance...");
                var fiJson = File.ReadAllText(featureImportancePath);
                var fiResults = JsonSerializer.Deserialize<List<FeatureImportanceResult>>(fiJson);
                
                if (fiResults != null && fiResults.Any())
                {
                    var plotPath2 = Path.Combine(_modelsDirectory, "02_feature_importance.png");
                    visualizer.GenerateFeatureImportancePlot(fiResults, plotPath2);
                    Console.ForegroundColor = ConsoleColor.Green;
                    Console.WriteLine($"   ? Zapisano: {plotPath2}");
                    Console.ResetColor();
                }
                else
                {
                    Console.ForegroundColor = ConsoleColor.Yellow;
                    Console.WriteLine("   ? Pominiêto: brak danych feature importance (uruchom opcjê 7)");
                    Console.ResetColor();
                }
            }
            else
            {
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine("? [2/5] Feature Importance: pominiêto (uruchom opcjê 7 najpierw)");
                Console.ResetColor();
            }

            // 3. TIME SERIES PLOT (dla przyk³adowej kategorii)
            Console.WriteLine("? [3/5] Generowanie wykresu Time Series (TOP kategoria)...");
            var topCategory = weeklyData
                .GroupBy(x => x.Category)
                .OrderByDescending(g => g.Sum(x => x.Quantity))
                .First()
                .Key;
            
            var plotPath3 = Path.Combine(_modelsDirectory, "03_time_series.png");
            visualizer.GenerateTimeSeriesPlot(weeklyData, topCategory, trainEndDate, testStartDate, plotPath3);
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"   ? Zapisano: {plotPath3} (kategoria: {topCategory})");
            Console.ResetColor();

            // 4. RESIDUALS HISTOGRAM
            Console.WriteLine("? [4/5] Generowanie histogramu Residuals...");
            var residuals = actual.Zip(predicted, (a, p) => a - p).ToList();
            var plotPath4 = Path.Combine(_modelsDirectory, "04_residuals_histogram.png");
            visualizer.GenerateResidualsHistogram(residuals, plotPath4);
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"   ? Zapisano: {plotPath4}");
            Console.ResetColor();

            // 5. METRICS COMPARISON (Baseline vs. FastTree)
            var baselineComparisonPath = Path.Combine(_modelsDirectory, "baseline_comparison.json");
            if (File.Exists(baselineComparisonPath))
            {
                Console.WriteLine("? [5/5] Generowanie wykresu Metrics Comparison...");
                var compJson = File.ReadAllText(baselineComparisonPath);
                var comparison = JsonSerializer.Deserialize<BaselineComparisonReport>(compJson);
                
                if (comparison != null)
                {
                    var plotPath5 = Path.Combine(_modelsDirectory, "05_metrics_comparison.png");
                    visualizer.GenerateMetricsComparisonPlot(comparison, plotPath5);
                    Console.ForegroundColor = ConsoleColor.Green;
                    Console.WriteLine($"   ? Zapisano: {plotPath5}");
                    Console.ResetColor();
                }
            }
            else
            {
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine("? [5/5] Metrics Comparison: pominiêto (uruchom opcjê 8 najpierw)");
                Console.ResetColor();
            }

            // RESIDUAL ANALYSIS REPORT
            Console.WriteLine("\n? Analiza Residuals...");
            var residualAnalyzer = new ResidualAnalyzer();
            var residualReport = residualAnalyzer.AnalyzeResiduals(actual, predicted);

            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("\n??? ANALIZA RESIDUALS (B£ÊDÓW) ???");
            Console.ResetColor();
            Console.WriteLine($"Mean residual:         {residualReport.MeanResidual:F2}");
            Console.WriteLine($"Std dev residual:      {residualReport.StdDevResidual:F2}");
            Console.WriteLine($"Median residual:       {residualReport.MedianResidual:F2}");
            Console.WriteLine($"Max overprediction:    {residualReport.MaxOverprediction:F2}");
            Console.WriteLine($"Max underprediction:   {residualReport.MaxUnderprediction:F2}");

            // Zapisz raport residuals
            var residualPath = Path.Combine(_modelsDirectory, "residual_analysis.json");
            var residualJson = JsonSerializer.Serialize(residualReport, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(residualPath, residualJson);

            // PODSUMOWANIE
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"\n\n? Wszystkie wykresy zapisane w katalogu: {_modelsDirectory}");
            Console.WriteLine($"   • 01_predicted_vs_actual.png");
            Console.WriteLine($"   • 02_feature_importance.png (jeœli dostêpne)");
            Console.WriteLine($"   • 03_time_series.png");
            Console.WriteLine($"   • 04_residuals_histogram.png");
            Console.WriteLine($"   • 05_metrics_comparison.png (jeœli dostêpne)");
            Console.WriteLine($"   • residual_analysis.json");
            Console.ResetColor();

            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("\n?? TIP: U¿yj tych wykresów w pracy dyplomowej (rozdzia³ Wyniki)!");
            Console.ResetColor();
        }
        catch (Exception ex)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"? B£¥D: {ex.Message}");
            Console.WriteLine($"   Stack trace: {ex.StackTrace}");
            Console.ResetColor();
            _logger.Error(ex, "Visualization generation failed");
        }
    }

    /// <summary>
    /// Oblicza predykcje na zbiorze testowym
    /// </summary>
    private (List<double> actual, List<double> predicted) GetPredictions(
        MLContext mlContext,
        ITransformer model,
        List<SalesPredictionInput> testData)
    {
        // Transform test data (log1p)
        var transformedTestData = testData.Select(x => new SalesPredictionInput
        {
            WeekNumber = x.WeekNumber,
            Month = x.Month,
            Quarter = x.Quarter,
            IsBlackFridayWeek = x.IsBlackFridayWeek,
            IsHolidaySeason = x.IsHolidaySeason,
            Lag1 = x.Lag1,
            Lag2 = x.Lag2,
            Lag3 = x.Lag3,
            Lag4 = x.Lag4,
            RollingAvg4Weeks = x.RollingAvg4Weeks,
            Trend = x.Trend,
            CategoryHistoricalAvg = x.CategoryHistoricalAvg,
            ProductCategory = x.ProductCategory,
            ActualSales = (float)Math.Log(x.ActualSales + 1)
        }).ToList();

        var testDataView = mlContext.Data.LoadFromEnumerable(transformedTestData);
        var predictions = model.Transform(testDataView);

        // Extract actual vs predicted (odwróæ log-transform)
        var predictionResults = mlContext.Data
            .CreateEnumerable<SalesPredictionResult>(predictions, false)
            .Select(p => new
            {
                Actual = Math.Exp(p.Label) - 1,
                Predicted = Math.Exp(p.Score) - 1
            })
            .ToList();

        return (
            predictionResults.Select(p => p.Actual).ToList(),
            predictionResults.Select(p => p.Predicted).ToList()
        );
    }

    /// <summary>
    /// Klasa pomocnicza do odczytu predykcji z ML.NET
    /// </summary>
    private class SalesPredictionResult
    {
        public float Label { get; set; }
        public float Score { get; set; }
    }
}

/// <summary>
/// Raport porównania z baseline (dla deserializacji JSON)
/// </summary>
public class BaselineComparisonReport
{
    public ModelMetrics NaiveBaseline { get; set; } = new();
    public ModelMetrics FastTreeML { get; set; } = new();
    public ImprovementMetrics Improvements { get; set; } = new();
}

/// <summary>
/// Metryki poprawy w porównaniu z baseline
/// </summary>
public class ImprovementMetrics
{
    public double RSquared { get; set; }
    public double MAE { get; set; }
    public double RMSE { get; set; }
    public double MAPE { get; set; }
}