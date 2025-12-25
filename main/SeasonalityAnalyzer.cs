using SalesForecasting.Core.Models;
using Serilog;

namespace SalesForecasting.ML;

/// <summary>
/// Analiza sezonowości i trendów w danych Olist
/// </summary>
public class SeasonalityAnalyzer
{
    private readonly ILogger _logger;

    public SeasonalityAnalyzer(ILogger logger)
    {
        _logger = logger;
    }

    public SeasonalityReport AnalyzeSeasonality(List<WeeklySalesData> weeklyData)
    {
        _logger.Information("Starting seasonality analysis");

        return new SeasonalityReport
        {
            MonthlyPatterns = AnalyzeMonthlyPatterns(weeklyData),
            QuarterlyPatterns = AnalyzeQuarterlyPatterns(weeklyData),
            SalesPeaks = DetectSalesPeaks(weeklyData),
            LongTermTrend = AnalyzeLongTermTrend(weeklyData)
        };
    }

    private List<MonthlyPattern> AnalyzeMonthlyPatterns(List<WeeklySalesData> data)
    {
        // ✅ POPRAWKA: Quantity zamiast SalesVolume
        var monthlyStats = data
            .GroupBy(x => x.WeekStartDate.Month)
            .Select(g => new MonthlyPattern
            {
                Month = g.Key,
                MonthName = new DateTime(2000, g.Key, 1).ToString("MMMM"),
                AverageSales = g.Average(x => x.Quantity),
                TotalSales = g.Sum(x => x.Quantity),
                SampleCount = g.Count()
            })
            .OrderBy(m => m.Month)
            .ToList();

        var grandMean = monthlyStats.Average(m => m.AverageSales);
        foreach (var month in monthlyStats)
            month.SeasonalIndex = grandMean > 0 ? month.AverageSales / grandMean : 1;

        return monthlyStats;
    }

    private List<QuarterlyPattern> AnalyzeQuarterlyPatterns(List<WeeklySalesData> data)
    {
        // ✅ POPRAWKA: Quantity zamiast SalesVolume
        return data
            .GroupBy(x => (x.WeekStartDate.Month - 1) / 3 + 1)
            .Select(g => new QuarterlyPattern
            {
                Quarter = g.Key,
                QuarterName = $"Q{g.Key}",
                AverageSales = g.Average(x => x.Quantity),
                TotalSales = g.Sum(x => x.Quantity)
            })
            .OrderBy(q => q.Quarter)
            .ToList();
    }

    private List<SalesPeak> DetectSalesPeaks(List<WeeklySalesData> data)
    {
        var orderedData = data.OrderBy(x => x.WeekStartDate).ToList();
        if (!orderedData.Any()) return new List<SalesPeak>();

        // ✅ POPRAWKA: Quantity zamiast SalesVolume
        var mean = orderedData.Average(x => x.Quantity);
        var stdDev = Math.Sqrt(orderedData.Average(x => Math.Pow(x.Quantity - mean, 2)));
        var threshold = mean + 2 * stdDev;

        return orderedData
            .Where(x => x.Quantity > threshold)
            .Select(x => new SalesPeak
            {
                WeekStartDate = x.WeekStartDate,
                Sales = x.Quantity,
                PercentAboveMean = mean > 0 ? ((x.Quantity - mean) / mean) * 100 : 0,
                PossibleReason = IdentifyPeakReason(x.WeekStartDate)
            })
            .ToList();
    }

    private string IdentifyPeakReason(DateTime date)
    {
        // Black Friday - trzeci/czwarty tydzień listopada (w USA ostatni piątek, ale promocje trwają cały tydzień)
        if (date.Month == 11 && date.Day >= 15 && date.Day <= 30)
            return "🛒 Black Friday / Cyber Week";
        
        // Boże Narodzenie (Natal) - cały grudzień do 25
        if (date.Month == 12 && date.Day >= 1 && date.Day <= 25)
            return "🎄 Sezon świąteczny (Natal)";
        
        // Dia das Mães - 2. niedziela maja (około 8-14 maja)
        if (date.Month == 5 && date.Day >= 1 && date.Day <= 14)
            return "💐 Dia das Mães";
        
        // Dia dos Pais - 2. niedziela sierpnia (około 8-14 sierpnia)
        if (date.Month == 8 && date.Day >= 1 && date.Day <= 14)
            return "👔 Dia dos Pais";
        
        // Dia das Crianças - 12 października
        if (date.Month == 10 && date.Day >= 5 && date.Day <= 15)
            return "🧸 Dia das Crianças";
        
        // Carnaval - luty/marzec (ruchoma data)
        if ((date.Month == 2 || date.Month == 3) && date.Day >= 1 && date.Day <= 15)
            return "🎭 Carnaval (możliwy)";
        
        // Walentynki w Brazylii - 12 czerwca (Dia dos Namorados)
        if (date.Month == 6 && date.Day >= 5 && date.Day <= 15)
            return "❤️ Dia dos Namorados";
        
        // Początek roku szkolnego w Brazylii - luty
        if (date.Month == 2 && date.Day >= 15 && date.Day <= 28)
            return "📚 Powrót do szkoły";

        return "❓ Nieznana przyczyna";
    }

    private TrendAnalysis AnalyzeLongTermTrend(List<WeeklySalesData> data)
    {
        var orderedData = data.OrderBy(x => x.WeekStartDate).ToList();
        var n = orderedData.Count;
        if (n < 2) return new TrendAnalysis { TrendDirection = "Brak danych" };

        double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
        for (int i = 0; i < n; i++)
        {
            sumX += i;
            // ✅ POPRAWKA: Quantity zamiast SalesVolume
            sumY += orderedData[i].Quantity;
            sumXY += i * orderedData[i].Quantity;
            sumX2 += i * i;
        }

        var denominator = n * sumX2 - sumX * sumX;
        var slope = denominator != 0 ? (n * sumXY - sumX * sumY) / denominator : 0;

        return new TrendAnalysis
        {
            Slope = slope,
            TrendDirection = slope > 0.1 ? "Rosnący" : slope < -0.1 ? "Malejący" : "Stabilny",
            WeeklyGrowthRate = slope,
            MonthlyGrowthRate = slope * 4
        };
    }

    public void PrintSeasonalityReport(SeasonalityReport report)
    {
        Console.WriteLine("\n" + new string('═', 100));
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("📅 ANALIZA SEZONOWOŚCI I TRENDÓW - DANE OLIST");
        Console.ResetColor();
        Console.WriteLine(new string('═', 100));

        Console.WriteLine("\n📊 WZORCE MIESIĘCZNE:");
        Console.WriteLine($"{"Miesiąc",-15} {"Śr. sprzedaż",-15} {"Indeks sez.",-15}");
        Console.WriteLine(new string('-', 50));
        foreach (var m in report.MonthlyPatterns)
        {
            Console.ForegroundColor = m.SeasonalIndex > 1.1 ? ConsoleColor.Green : 
                                      m.SeasonalIndex < 0.9 ? ConsoleColor.Red : ConsoleColor.Gray;
            Console.WriteLine($"{m.MonthName,-15} {m.AverageSales,-15:F0} {m.SeasonalIndex,-15:F2}");
            Console.ResetColor();
        }

        Console.WriteLine($"\n📈 TREND: {report.LongTermTrend.TrendDirection} (slope: {report.LongTermTrend.Slope:F2})");

        if (report.SalesPeaks.Any())
        {
            Console.WriteLine($"\n🔝 SZCZYTY SPRZEDAŻY (>2σ):");
            foreach (var peak in report.SalesPeaks.OrderByDescending(p => p.Sales).Take(5))
                Console.WriteLine($"   {peak.WeekStartDate:yyyy-MM-dd}: {peak.Sales:F0} (+{peak.PercentAboveMean:F0}%) - {peak.PossibleReason}");
        }
    }
}

#region Models
public class SeasonalityReport
{
    public List<MonthlyPattern> MonthlyPatterns { get; set; } = new();
    public List<QuarterlyPattern> QuarterlyPatterns { get; set; } = new();
    public List<SalesPeak> SalesPeaks { get; set; } = new();
    public TrendAnalysis LongTermTrend { get; set; } = new();
}

public class MonthlyPattern
{
    public int Month { get; set; }
    public string MonthName { get; set; } = string.Empty;
    public double AverageSales { get; set; }
    public double TotalSales { get; set; }
    public int SampleCount { get; set; }
    public double SeasonalIndex { get; set; }
}

public class QuarterlyPattern
{
    public int Quarter { get; set; }
    public string QuarterName { get; set; } = string.Empty;
    public double AverageSales { get; set; }
    public double TotalSales { get; set; }
}

public class SalesPeak
{
    public DateTime WeekStartDate { get; set; }
    public double Sales { get; set; }
    public double PercentAboveMean { get; set; }
    public string PossibleReason { get; set; } = string.Empty;
}

public class TrendAnalysis
{
    public double Slope { get; set; }
    public string TrendDirection { get; set; } = string.Empty;
    public double WeeklyGrowthRate { get; set; }
    public double MonthlyGrowthRate { get; set; }
}
#endregion