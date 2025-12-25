using Serilog;

namespace SalesForecasting.ML;

/// <summary>
/// Profit and loss analysis based on forecasts
/// </summary>
public class ProfitLossAnalyzer
{
    private readonly ILogger _logger;

    public decimal CostPerUnitOverstock { get; set; } = 5.00m;
    public decimal CostPerUnitStockout { get; set; } = 15.00m;
    public decimal AverageMarginPerUnit { get; set; } = 20.00m;

    public ProfitLossAnalyzer(ILogger logger)
    {
        _logger = logger;
    }

    public ProfitLossReport AnalyzeProfitLoss(
        List<double> actual,
        List<double> predicted,
        string scenario = "default")
    {
        var report = new ProfitLossReport { Scenario = scenario, TotalPredictions = actual.Count };

        decimal totalOverstockLoss = 0, totalStockoutLoss = 0;
        int overstockCount = 0, stockoutCount = 0, accurateCount = 0;

        for (int i = 0; i < actual.Count; i++)
        {
            var actualSales = (decimal)actual[i];
            var predictedSales = (decimal)Math.Max(0, predicted[i]);
            var error = predictedSales - actualSales;

            if (error > 0.5m) { overstockCount++; totalOverstockLoss += error * CostPerUnitOverstock; }
            else if (error < -0.5m) { stockoutCount++; totalStockoutLoss += Math.Abs(error) * CostPerUnitStockout; }
            else { accurateCount++; }
        }

        report.OverstockInstances = overstockCount;
        report.StockoutInstances = stockoutCount;
        report.AccurateInstances = accurateCount;
        report.TotalOverstockLoss = totalOverstockLoss;
        report.TotalStockoutLoss = totalStockoutLoss;
        report.TotalLoss = totalOverstockLoss + totalStockoutLoss;

        return report;
    }

    public void CompareScenarios(List<double> actual, List<double> mlPredicted, List<double> naivePredicted)
    {
        Console.WriteLine("\n" + new string('═', 100));
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("💰 PROFIT & LOSS ANALYSIS - SCENARIO COMPARISON");
        Console.ResetColor();
        Console.WriteLine(new string('═', 100));

        Console.WriteLine($"\n📋 PARAMETERS: Overstock=R${CostPerUnitOverstock:N2}, Stockout=R${CostPerUnitStockout:N2}, Margin=R${AverageMarginPerUnit:N2}");

        var mlReport = AnalyzeProfitLoss(actual, mlPredicted, "FastTree ML");
        var naiveReport = AnalyzeProfitLoss(actual, naivePredicted, "Naive Baseline");

        Console.WriteLine($"\n{"Metric",-30} {"Naive",-18} {"FastTree ML",-18} {"Difference",-18}");
        Console.WriteLine(new string('-', 90));
        Console.WriteLine($"{"Overstock instances",-30} {naiveReport.OverstockInstances,-18} {mlReport.OverstockInstances,-18} {mlReport.OverstockInstances - naiveReport.OverstockInstances,-18}");
        Console.WriteLine($"{"Stockout instances",-30} {naiveReport.StockoutInstances,-18} {mlReport.StockoutInstances,-18} {mlReport.StockoutInstances - naiveReport.StockoutInstances,-18}");
        Console.WriteLine($"{"Overstock losses",-30} R${naiveReport.TotalOverstockLoss,-17:N2} R${mlReport.TotalOverstockLoss,-17:N2} R${mlReport.TotalOverstockLoss - naiveReport.TotalOverstockLoss,-17:N2}");
        Console.WriteLine($"{"Stockout losses",-30} R${naiveReport.TotalStockoutLoss,-17:N2} R${mlReport.TotalStockoutLoss,-17:N2} R${mlReport.TotalStockoutLoss - naiveReport.TotalStockoutLoss,-17:N2}");
        
        var savings = naiveReport.TotalLoss - mlReport.TotalLoss;
        Console.ForegroundColor = savings > 0 ? ConsoleColor.Green : ConsoleColor.Red;
        Console.WriteLine($"{"TOTAL LOSSES",-30} R${naiveReport.TotalLoss,-17:N2} R${mlReport.TotalLoss,-17:N2} R${-savings,-17:N2}");
        Console.WriteLine($"\n💵 SAVINGS WITH ML: R${savings:N2}");
        Console.ResetColor();
    }
}

public class ProfitLossReport
{
    public string Scenario { get; set; } = string.Empty;
    public int TotalPredictions { get; set; }
    public int OverstockInstances { get; set; }
    public int StockoutInstances { get; set; }
    public int AccurateInstances { get; set; }
    public decimal TotalOverstockLoss { get; set; }
    public decimal TotalStockoutLoss { get; set; }
    public decimal TotalLoss { get; set; }
}