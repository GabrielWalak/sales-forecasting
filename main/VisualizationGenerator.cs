using SalesForecasting.Core.Models;
using Serilog;
using ScottPlot;

namespace SalesForecasting.ML;

/// <summary>
/// Generator wykresów dla analizy wyników
/// </summary>
public class VisualizationGenerator
{
    private readonly ILogger _logger;

    public VisualizationGenerator(ILogger logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Tworzy wykres Predicted vs. Actual (scatter plot)
    /// </summary>
    public void GeneratePredictedVsActualPlot(
        List<double> actual,
        List<double> predicted,
        string outputPath)
    {
        var plt = new Plot();

        // Scatter plot
        var scatter = plt.Add.Scatter(actual.ToArray(), predicted.ToArray());
        scatter.MarkerSize = 5;
        scatter.Color = ScottPlot.Color.FromHex("#2196F3");

        // Linia y=x (idealna predykcja)
        var max = Math.Max(actual.Max(), predicted.Max());
        var line = plt.Add.Line(0, 0, max, max);
        line.Color = ScottPlot.Color.FromHex("#FF0000");
        line.LinePattern = LinePattern.Dashed;
        line.LineWidth = 2;

        plt.Axes.SetLimits(0, max, 0, max);
        plt.Title("Predicted vs. Actual Sales");
        plt.XLabel("Actual Sales");
        plt.YLabel("Predicted Sales");
        plt.Legend.IsVisible = false;

        plt.SavePng(outputPath, 1920, 1080);
        _logger.Information("Saved predicted vs actual plot to {Path}", outputPath);
    }

    /// <summary>
    /// Wykres Feature Importance (bar chart)
    /// </summary>
    public void GenerateFeatureImportancePlot(
        List<FeatureImportanceResult> results,
        string outputPath)
    {
        var plt = new Plot();

        var featureNames = results.Select(r => r.FeatureName).ToArray();
        var importance = results.Select(r => r.ImportanceScore).ToArray();

        var positions = ScottPlot.Generate.Consecutive(featureNames.Length);
        var bars = plt.Add.Bars(positions, importance);
        
        // Koloruj według ważności
        for (int i = 0; i < bars.Bars.Count; i++)
        {
            if (importance[i] > 75)
                bars.Bars[i].FillColor = ScottPlot.Color.FromHex("#4CAF50"); // Zielony
            else if (importance[i] > 40)
                bars.Bars[i].FillColor = ScottPlot.Color.FromHex("#FFC107"); // Żółty
            else
                bars.Bars[i].FillColor = ScottPlot.Color.FromHex("#9E9E9E"); // Szary
        }

        plt.Axes.Bottom.TickGenerator = new ScottPlot.TickGenerators.NumericManual(positions, featureNames);
        plt.Axes.Bottom.TickLabelStyle.Rotation = 45;
        plt.Axes.Bottom.TickLabelStyle.Alignment = Alignment.MiddleRight;

        plt.Title("Feature Importance (Permutation)");
        plt.YLabel("Importance Score (%)");

        plt.SavePng(outputPath, 1920, 1080);
        _logger.Information("Saved feature importance plot to {Path}", outputPath);
    }

    /// <summary>
    /// Wykres Time Series (actual vs predicted w czasie)
    /// </summary>
    public void GenerateTimeSeriesPlot(
        List<WeeklySalesData> weeklyData,
        string category,
        DateTime trainEndDate,
        DateTime testStartDate,
        string outputPath)
    {
        var categoryData = weeklyData
            .Where(x => x.Category == category)
            .OrderBy(x => x.WeekStartDate)
            .ToList();

        var dates = categoryData.Select(x => x.WeekStartDate.ToOADate()).ToArray();
        var sales = categoryData.Select(x => (double)x.Quantity).ToArray();

        var plt = new Plot();
        var scatter = plt.Add.Scatter(dates, sales);
        scatter.LegendText = category;
        scatter.LineWidth = 2;
        scatter.MarkerSize = 0;

        // Dodaj linię podziału train/test
        var trainLine = plt.Add.VerticalLine(trainEndDate.ToOADate());
        trainLine.Color = ScottPlot.Color.FromHex("#FF0000");
        trainLine.LineWidth = 2;
        trainLine.LinePattern = LinePattern.Dashed;
        trainLine.LegendText = "Train/Test Split";

        plt.Title($"Time Series: {category}");
        plt.XLabel("Date");
        plt.YLabel("Sales Quantity");
        plt.Axes.DateTimeTicksBottom();
        plt.ShowLegend();

        plt.SavePng(outputPath, 1920, 1080);
        _logger.Information("Saved time series plot to {Path}", outputPath);
    }

    /// <summary>
    /// Histogram residuals (rozkład błędów)
    /// </summary>
    public void GenerateResidualsHistogram(List<double> residuals, string outputPath)
    {
        var plt = new Plot();
        
        // ✅ POPRAWKA: Najpierw utwórz histogram statistics, potem dodaj do wykresu
        var histStats = ScottPlot.Statistics.Histogram.WithBinCount(50, residuals.ToArray());
        var hist = plt.Add.Bars(histStats.Bins, histStats.Counts);
        
        // Koloruj słupki
        foreach (var bar in hist.Bars)
        {
            bar.FillColor = ScottPlot.Color.FromHex("#4CAF50");
            bar.LineColor = ScottPlot.Color.FromHex("#388E3C");
        }

        plt.Title("Residuals Distribution");
        plt.XLabel("Residual (Actual - Predicted)");
        plt.YLabel("Frequency");

        plt.SavePng(outputPath, 1920, 1080);
        _logger.Information("Saved residuals histogram to {Path}", outputPath);
    }

    /// <summary>
    /// Wykres porównania metryk (Baseline vs FastTree)
    /// </summary>
    public void GenerateMetricsComparisonPlot(
        BaselineComparisonReport comparison,
        string outputPath)
    {
        var plt = new Plot();

        var metricNames = new[] { "MAE", "RMSE", "MAPE (%)" };
        var naiveValues = new double[] 
        { 
            comparison.NaiveBaseline.MAE, 
            comparison.NaiveBaseline.RMSE, 
            comparison.NaiveBaseline.MAPE 
        };
        var fastTreeValues = new double[] 
        { 
            comparison.FastTreeML.MAE, 
            comparison.FastTreeML.RMSE, 
            comparison.FastTreeML.MAPE 
        };

        var positions = ScottPlot.Generate.Consecutive(metricNames.Length);
        
        // ✅ POPRAWKA: Użyj Add.Bars (liczba mnoga) zamiast Add.Bar
        var bar1 = plt.Add.Bars(positions, naiveValues);
        bar1.LegendText = "Naive Baseline"; // ✅ POPRAWKA: zamiast Label
        
        // Koloruj słupki Naive
        foreach (var bar in bar1.Bars)
        {
            bar.FillColor = ScottPlot.Color.FromHex("#F44336");
        }

        var bar2 = plt.Add.Bars(positions.Select(x => x + 0.4).ToArray(), fastTreeValues);
        bar2.LegendText = "FastTree ML"; // ✅ POPRAWKA: zamiast Label
        
        // Koloruj słupki FastTree
        foreach (var bar in bar2.Bars)
        {
            bar.FillColor = ScottPlot.Color.FromHex("#4CAF50");
        }

        plt.Axes.Bottom.TickGenerator = new ScottPlot.TickGenerators.NumericManual(positions, metricNames);
        plt.Title("Metrics Comparison: Baseline vs. FastTree");
        plt.YLabel("Error Value");
        plt.ShowLegend();

        plt.SavePng(outputPath, 1920, 1080);
        _logger.Information("Saved metrics comparison plot to {Path}", outputPath);
    }
}