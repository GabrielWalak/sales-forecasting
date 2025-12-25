using Microsoft.ML;
using SalesForecasting.Core.Models;
using Serilog;

namespace SalesForecasting.ML;

/// <summary>
/// Trainer for the GLOBAL weekly model (all categories together)
/// </summary>
public class GlobalModelTrainer
{
    private readonly MLContext _mlContext;
    private readonly ILogger _logger;

    public GlobalModelTrainer(ILogger logger)
    {
        _mlContext = new MLContext(seed: 42);
        _logger = logger;
    }

    /// <summary>
    /// Trains a GLOBAL FastTree model with all categories.
    /// ✅ ProductCategory is one-hot encoded automatically.
    /// </summary>
    public ITransformer TrainGlobalModel(List<SalesPredictionInput> trainData)
    {
        _logger.Information("Training GLOBAL FastTree model with {Count} samples across all categories", 
            trainData.Count);

        // ✅ Transform labels before training (log1p)
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
            ActualSales = (float)Math.Log(x.ActualSales + 1)  // ✅ log1p transformation
        }).ToList();

        var trainDataView = _mlContext.Data.LoadFromEnumerable(transformedData);

        // Pipeline unchanged
        var pipeline = _mlContext.Transforms
            .Categorical.OneHotEncoding("ProductCategoryEncoded", nameof(SalesPredictionInput.ProductCategory))
            .Append(_mlContext.Transforms.Concatenate(
                "Features",
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
            .Append(_mlContext.Regression.Trainers.FastTree(new Microsoft.ML.Trainers.FastTree.FastTreeRegressionTrainer.Options
            {
                NumberOfTrees = 300,
                NumberOfLeaves = 60,
                MinimumExampleCountPerLeaf = 8,
                LearningRate = 0.04,
                LabelColumnName = "Label",
                FeatureColumnName = "Features"
            }));

        _logger.Information("Training GLOBAL model with log-transformed labels...");
        var model = pipeline.Fit(trainDataView);

        _logger.Information("GLOBAL model trained successfully");
        return model;
    }

    /// <summary>
    /// Evaluates the GLOBAL model
    /// </summary>
    public ModelMetrics EvaluateGlobalModel(ITransformer model, List<SalesPredictionInput> testData)
    {
        _logger.Information("Evaluating GLOBAL model with {Count} test samples", testData.Count);

        // ✅ Transform test set (log1p)
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
            ActualSales = (float)Math.Log(x.ActualSales + 1)  // ✅ log1p
        }).ToList();

        var testDataView = _mlContext.Data.LoadFromEnumerable(transformedTestData);
        var predictions = model.Transform(testDataView);

        // ✅ Invert transform (expm1) before computing metrics
        var predictionResults = _mlContext.Data
            .CreateEnumerable<SalesPredictionResult>(predictions, false)
            .Select(p => new
            {
                Actual = Math.Exp(p.Label) - 1,      // ✅ expm1 (odwrotność log1p)
                Predicted = Math.Exp(p.Score) - 1    // ✅ expm1
            })
            .ToList();

        // Compute metrics on the original scale
        var mae = predictionResults.Average(p => Math.Abs(p.Actual - p.Predicted));
        var rmse = Math.Sqrt(predictionResults.Average(p => Math.Pow(p.Actual - p.Predicted, 2)));
        
        // R² on the original scale
        var meanActual = predictionResults.Average(p => p.Actual);
        var ssTot = predictionResults.Sum(p => Math.Pow(p.Actual - meanActual, 2));
        var ssRes = predictionResults.Sum(p => Math.Pow(p.Actual - p.Predicted, 2));
        var rSquared = 1 - (ssRes / ssTot);

        // MAPE on the original scale
        var mape = CalculateMAPE(predictionResults
            .Select(p => new SalesPredictionResult 
            { 
                Label = (float)p.Actual, 
                Score = (float)p.Predicted 
            })
            .ToList());

        var modelMetrics = new ModelMetrics
        {
            Category = "GLOBAL_MODEL",
            RSquared = rSquared,
            MAE = mae,
            RMSE = rmse,
            MAPE = mape,
            TrainSamples = 0,
            TestSamples = testData.Count,
            TrainedAt = DateTime.UtcNow
        };

        _logger.Information(
            "GLOBAL Model: R²={R2:F4}, MAE={MAE:F2}, RMSE={RMSE:F2}, MAPE={MAPE:F2}%",
            modelMetrics.RSquared, modelMetrics.MAE, modelMetrics.RMSE, modelMetrics.MAPE);

        return modelMetrics;
    }

    /// <summary>
    /// Prediction with automatic inverse log transform
    /// </summary>
    public float PredictSales(ITransformer model, SalesPredictionInput input)
    {
        var predictionEngine = _mlContext.Model
            .CreatePredictionEngine<SalesPredictionInput, SalesPredictionOutput>(model);

        // Model predicts in log scale
        var logPrediction = predictionEngine.Predict(input);
        
        // ✅ Invert transform (expm1)
        var actualPrediction = (float)(Math.Exp(logPrediction.PredictedSales) - 1);
        
        return Math.Max(0, actualPrediction);  // Cannot be negative
    }

    private double CalculateMAPE(List<SalesPredictionResult> predictions)
    {
        // ✅ Ignore values < 3 units (too small for MAPE)
        var validPredictions = predictions.Where(p => p.Label >= 3).ToList();
        
        if (!validPredictions.Any())
            return 0;

        var mape = validPredictions.Average(p => Math.Abs((p.Label - p.Score) / p.Label)) * 100;
        
        _logger.Information("MAPE calculated on {Count}/{Total} samples (filtered < 3 units)", 
            validPredictions.Count, predictions.Count);
        
        return mape;
    }

    private class SalesPredictionResult
    {
        public float Label { get; set; }
        public float Score { get; set; }
    }
}