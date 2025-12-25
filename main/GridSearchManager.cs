using Microsoft.ML;
using Microsoft.ML.Trainers.FastTree;
using SalesForecasting.Core.Models;
using Serilog;
using System.Diagnostics;

namespace SalesForecasting.ML;

/// <summary>
/// Manager do Grid Search hiperparametrów FastTree
/// </summary>
public class GridSearchManager
{
    private readonly MLContext _mlContext;
    private readonly ILogger _logger;

    public GridSearchManager(ILogger logger)
    {
        _mlContext = new MLContext(seed: 42);
        _logger = logger;
    }

    /// <summary>
    /// Przeprowadza Grid Search dla modelu globalnego
    /// </summary>
    public List<GridSearchResult> PerformGridSearch(
        List<SalesPredictionInput> trainData,
        List<SalesPredictionInput> testData,
        List<GridSearchConfiguration> configurations)
    {
        _logger.Information("Starting Grid Search with {Count} configurations", configurations.Count);

        var results = new List<GridSearchResult>();
        var configNumber = 0;

        foreach (var config in configurations)
        {
            configNumber++;
            Console.WriteLine($"\n🔍 Testing configuration {configNumber}/{configurations.Count}:");
            Console.WriteLine($"   {config}");

            var stopwatch = Stopwatch.StartNew();

            try
            {
                // Trenuj model z daną konfiguracją
                var model = TrainModelWithConfig(trainData, config);

                // Ewaluuj model
                var metrics = EvaluateModel(model, testData);

                stopwatch.Stop();

                var result = new GridSearchResult
                {
                    Configuration = config,
                    Metrics = metrics,
                    TrainingTime = stopwatch.Elapsed
                };

                results.Add(result);

                // Pokaż wyniki
                Console.ForegroundColor = ConsoleColor.Green;
                Console.WriteLine($"   ✓ R²={metrics.RSquared:F4}, MAE={metrics.MAE:F2}, RMSE={metrics.RMSE:F2}, MAPE={metrics.MAPE:F2}%");
                Console.WriteLine($"   ⏱ Training time: {stopwatch.Elapsed.TotalSeconds:F1}s");
                Console.ResetColor();

                _logger.Information(
                    "Config: {Config} → R²={R2:F4}, MAE={MAE:F2}, Time={Time:F1}s",
                    config.ToString(), metrics.RSquared, metrics.MAE, stopwatch.Elapsed.TotalSeconds);
            }
            catch (Exception ex)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"   ❌ ERROR: {ex.Message}");
                Console.ResetColor();
                _logger.Error(ex, "Grid Search failed for config: {Config}", config.ToString());
            }
        }

        return results;
    }

    /// <summary>
    /// Trenuje model z konkretną konfiguracją hiperparametrów
    /// </summary>
    private ITransformer TrainModelWithConfig(
        List<SalesPredictionInput> trainData,
        GridSearchConfiguration config)
    {
        // Log-transform (jak w GlobalModelTrainer)
        var transformedData = trainData.Select(x => new SalesPredictionInput
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

        var dataView = _mlContext.Data.LoadFromEnumerable(transformedData);

        // Pipeline z konfigurowalnymi hiperparametrami
        var pipeline = _mlContext.Transforms
            .Categorical.OneHotEncoding("ProductCategoryEncoded", nameof(SalesPredictionInput.ProductCategory))
            .Append(_mlContext.Transforms.Concatenate("Features",
                "ProductCategoryEncoded",
                nameof(SalesPredictionInput.WeekNumber),
                nameof(SalesPredictionInput.Month),
                nameof(SalesPredictionInput.Quarter),
                nameof(SalesPredictionInput.IsBlackFridayWeek),
                nameof(SalesPredictionInput.IsHolidaySeason),
                nameof(SalesPredictionInput.Lag1),
                nameof(SalesPredictionInput.Lag2),
                nameof(SalesPredictionInput.Lag3),
                nameof(SalesPredictionInput.Lag4),
                nameof(SalesPredictionInput.RollingAvg4Weeks),
                nameof(SalesPredictionInput.Trend),
                nameof(SalesPredictionInput.CategoryHistoricalAvg)))
            .Append(_mlContext.Transforms.NormalizeMinMax("Features"))
            .Append(_mlContext.Regression.Trainers.FastTree(new FastTreeRegressionTrainer.Options
            {
                NumberOfTrees = config.NumberOfTrees,
                NumberOfLeaves = config.NumberOfLeaves,
                MinimumExampleCountPerLeaf = config.MinimumExampleCountPerLeaf,
                LearningRate = config.LearningRate,
                LabelColumnName = "Label",
                FeatureColumnName = "Features"
            }));

        return pipeline.Fit(dataView);
    }

    /// <summary>
    /// Ewaluuje model (identyczny jak w GlobalModelTrainer)
    /// </summary>
    private ModelMetrics EvaluateModel(ITransformer model, List<SalesPredictionInput> testData)
    {
        // Transformuj test set (log1p)
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

        var testDataView = _mlContext.Data.LoadFromEnumerable(transformedTestData);
        var predictions = model.Transform(testDataView);

        // Odwróć transformację (expm1)
        var predictionResults = _mlContext.Data
            .CreateEnumerable<SalesPredictionResult>(predictions, false)
            .Select(p => new
            {
                Actual = Math.Exp(p.Label) - 1,
                Predicted = Math.Exp(p.Score) - 1
            })
            .ToList();

        // Oblicz metryki
        var mae = predictionResults.Average(p => Math.Abs(p.Actual - p.Predicted));
        var rmse = Math.Sqrt(predictionResults.Average(p => Math.Pow(p.Actual - p.Predicted, 2)));
        
        var meanActual = predictionResults.Average(p => p.Actual);
        var ssTot = predictionResults.Sum(p => Math.Pow(p.Actual - meanActual, 2));
        var ssRes = predictionResults.Sum(p => Math.Pow(p.Actual - p.Predicted, 2));
        var rSquared = 1 - (ssRes / ssTot);

        // MAPE (filtr ≥3)
        var mape = CalculateMAPE(predictionResults
            .Where(p => p.Actual >= 3)
            .Select(p => (p.Actual, p.Predicted)));

        return new ModelMetrics
        {
            Category = "GridSearch",
            RSquared = rSquared,
            MAE = mae,
            RMSE = rmse,
            MAPE = mape,
            TestSamples = testData.Count,
            TrainedAt = DateTime.UtcNow
        };
    }

    private double CalculateMAPE(IEnumerable<(double Actual, double Predicted)> predictions)
    {
        var predictionList = predictions.ToList();
        
        if (!predictionList.Any())
            return 0;

        return predictionList.Average(p => Math.Abs((p.Actual - p.Predicted) / p.Actual)) * 100;
    }

    /// <summary>
    /// Generuje standardową siatkę hiperparametrów do przeszukiwania
    /// </summary>
    public static List<GridSearchConfiguration> GenerateStandardGrid()
    {
        var configurations = new List<GridSearchConfiguration>();

        var treesOptions = new[] { 50, 100, 200 };
        var leavesOptions = new[] { 10, 20, 30 };
        var minLeafOptions = new[] { 5, 10 };
        var learningRateOptions = new[] { 0.05, 0.1, 0.2 };

        foreach (var trees in treesOptions)
        foreach (var leaves in leavesOptions)
        foreach (var minLeaf in minLeafOptions)
        foreach (var lr in learningRateOptions)
        {
            configurations.Add(new GridSearchConfiguration
            {
                NumberOfTrees = trees,
                NumberOfLeaves = leaves,
                MinimumExampleCountPerLeaf = minLeaf,
                LearningRate = lr
            });
        }

        return configurations;
    }

    /// <summary>
    /// Generuje ograniczoną siatkę (szybka wersja do testów)
    /// </summary>
    public static List<GridSearchConfiguration> GenerateQuickGrid()
    {
        return new List<GridSearchConfiguration>
        {
            new() { NumberOfTrees = 50, NumberOfLeaves = 20, MinimumExampleCountPerLeaf = 5, LearningRate = 0.1 },
            new() { NumberOfTrees = 100, NumberOfLeaves = 20, MinimumExampleCountPerLeaf = 5, LearningRate = 0.1 },
            new() { NumberOfTrees = 100, NumberOfLeaves = 30, MinimumExampleCountPerLeaf = 5, LearningRate = 0.1 },
            new() { NumberOfTrees = 100, NumberOfLeaves = 20, MinimumExampleCountPerLeaf = 10, LearningRate = 0.1 },
            new() { NumberOfTrees = 100, NumberOfLeaves = 20, MinimumExampleCountPerLeaf = 5, LearningRate = 0.05 },
            new() { NumberOfTrees = 100, NumberOfLeaves = 20, MinimumExampleCountPerLeaf = 5, LearningRate = 0.2 },
        };
    }

    private class SalesPredictionResult
    {
        public float Label { get; set; }
        public float Score { get; set; }
    }
}