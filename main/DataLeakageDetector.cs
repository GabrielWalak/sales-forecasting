using SalesForecasting.Core.Models;
using Serilog;
using System.Text;

namespace SalesForecasting.ML;

/// <summary>
/// Detektor data leakage w modelach ML
/// Sprawdza korelacje między features a labelą, waliduje chronologię danych
/// </summary>
public class DataLeakageDetector
{
    private readonly ILogger _logger;

    public DataLeakageDetector(ILogger logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Główna metoda wykrywająca data leakage
    /// </summary>
    public LeakageReport DetectLeakage(
        List<SalesPredictionInput> trainData,
        List<SalesPredictionInput> testData,
        List<WeeklySalesData> originalData)
    {
        _logger.Information("Starting data leakage detection...");

        var report = new LeakageReport
        {
            TrainSamples = trainData.Count,
            TestSamples = testData.Count,
            AnalysisDate = DateTime.UtcNow
        };

        // Test 1: Korelacja między features a labelą
        report.FeatureCorrelations = CalculateFeatureCorrelations(trainData);
        
        // Test 2: Sprawdź podejrzanie wysokie korelacje (>0.95 = potencjalny leakage)
        report.SuspiciousFeatures = DetectSuspiciousCorrelations(report.FeatureCorrelations);

        // Test 3: Sprawdź czy lagi są poprawne (correlation < correlation z labelą)
        report.LagValidation = ValidateLagFeatures(trainData);

        // Test 4: Sprawdź chronologię danych
        report.TemporalValidation = ValidateTemporalOrder(originalData, trainData, testData);

        // Test 5: Sprawdź normalizację (czy min/max z train pokrywają test)
        report.NormalizationCheck = CheckNormalizationLeakage(trainData, testData);

        // Test 6: Sprawdź overlap między train/test
        report.DataOverlap = CheckTrainTestOverlap(trainData, testData);

        _logger.Information("Data leakage detection completed");
        return report;
    }

    /// <summary>
    /// Oblicza korelację Pearsona między każdą feature a labelą
    /// </summary>
    private Dictionary<string, double> CalculateFeatureCorrelations(List<SalesPredictionInput> data)
    {
        _logger.Information("Calculating feature correlations...");

        var correlations = new Dictionary<string, double>();
        var labels = data.Select(x => (double)x.ActualSales).ToList();

        // Korelacja dla każdej numerycznej feature
        correlations["WeekNumber"] = CalculatePearsonCorrelation(
            data.Select(x => (double)x.WeekNumber).ToList(), labels);
        
        correlations["Month"] = CalculatePearsonCorrelation(
            data.Select(x => (double)x.Month).ToList(), labels);
        
        correlations["Quarter"] = CalculatePearsonCorrelation(
            data.Select(x => (double)x.Quarter).ToList(), labels);
        
        correlations["IsBlackFridayWeek"] = CalculatePearsonCorrelation(
            data.Select(x => (double)x.IsBlackFridayWeek).ToList(), labels);
        
        correlations["IsHolidaySeason"] = CalculatePearsonCorrelation(
            data.Select(x => (double)x.IsHolidaySeason).ToList(), labels);
        
        correlations["Lag1"] = CalculatePearsonCorrelation(
            data.Select(x => (double)x.Lag1).ToList(), labels);
        
        correlations["Lag2"] = CalculatePearsonCorrelation(
            data.Select(x => (double)x.Lag2).ToList(), labels);
        
        correlations["Lag3"] = CalculatePearsonCorrelation(
            data.Select(x => (double)x.Lag3).ToList(), labels);
        
        correlations["Lag4"] = CalculatePearsonCorrelation(
            data.Select(x => (double)x.Lag4).ToList(), labels);
        
        correlations["RollingAvg4Weeks"] = CalculatePearsonCorrelation(
            data.Select(x => (double)x.RollingAvg4Weeks).ToList(), labels);
        
        correlations["Trend"] = CalculatePearsonCorrelation(
            data.Select(x => (double)x.Trend).ToList(), labels);
        
        correlations["CategoryHistoricalAvg"] = CalculatePearsonCorrelation(
            data.Select(x => (double)x.CategoryHistoricalAvg).ToList(), labels);

        return correlations;
    }

    /// <summary>
    /// Korelacja Pearsona między dwoma listami
    /// </summary>
    private double CalculatePearsonCorrelation(List<double> x, List<double> y)
    {
        if (x.Count != y.Count || x.Count == 0)
            return 0;

        var avgX = x.Average();
        var avgY = y.Average();

        var numerator = x.Zip(y, (xi, yi) => (xi - avgX) * (yi - avgY)).Sum();
        var denomX = Math.Sqrt(x.Sum(xi => Math.Pow(xi - avgX, 2)));
        var denomY = Math.Sqrt(y.Sum(yi => Math.Pow(yi - avgY, 2)));

        if (denomX == 0 || denomY == 0)
            return 0;

        return numerator / (denomX * denomY);
    }

    /// <summary>
    /// Wykrywa podejrzanie wysokie korelacje (potencjalny leakage)
    /// </summary>
    private List<string> DetectSuspiciousCorrelations(Dictionary<string, double> correlations)
    {
        const double SUSPICIOUS_THRESHOLD = 0.95; // Korelacja >95% = podejrzana

        var suspicious = correlations
            .Where(kvp => Math.Abs(kvp.Value) > SUSPICIOUS_THRESHOLD)
            .Select(kvp => $"{kvp.Key} (r={kvp.Value:F4})")
            .ToList();

        if (suspicious.Any())
        {
            _logger.Warning("Detected {Count} suspicious features with correlation > 0.95", suspicious.Count);
        }

        return suspicious;
    }

    /// <summary>
    /// Waliduje czy lagi są poprawne (Lag1 powinien mieć wyższą korelację niż Lag2, etc.)
    /// </summary>
    private LagValidationResult ValidateLagFeatures(List<SalesPredictionInput> data)
    {
        _logger.Information("Validating lag features...");

        var result = new LagValidationResult();
        var labels = data.Select(x => (double)x.ActualSales).ToList();

        var lag1Corr = CalculatePearsonCorrelation(data.Select(x => (double)x.Lag1).ToList(), labels);
        var lag2Corr = CalculatePearsonCorrelation(data.Select(x => (double)x.Lag2).ToList(), labels);
        var lag3Corr = CalculatePearsonCorrelation(data.Select(x => (double)x.Lag3).ToList(), labels);
        var lag4Corr = CalculatePearsonCorrelation(data.Select(x => (double)x.Lag4).ToList(), labels);

        result.Lag1Correlation = lag1Corr;
        result.Lag2Correlation = lag2Corr;
        result.Lag3Correlation = lag3Corr;
        result.Lag4Correlation = lag4Corr;

        // Sprawdź czy korelacje maleją z odległością (expected behavior)
        result.IsValid = Math.Abs(lag1Corr) >= Math.Abs(lag2Corr) &&
                         Math.Abs(lag2Corr) >= Math.Abs(lag3Corr) &&
                         Math.Abs(lag3Corr) >= Math.Abs(lag4Corr);

        if (!result.IsValid)
        {
            _logger.Warning("Lag features validation FAILED - correlations don't decrease with distance");
            result.Issues.Add("Lag correlations don't follow expected pattern (Lag1 > Lag2 > Lag3 > Lag4)");
        }

        return result;
    }

    /// <summary>
    /// Sprawdza chronologię danych (train przed test)
    /// </summary>
    private TemporalValidationResult ValidateTemporalOrder(
        List<WeeklySalesData> originalData,
        List<SalesPredictionInput> trainData,
        List<SalesPredictionInput> testData)
    {
        _logger.Information("Validating temporal order...");

        var result = new TemporalValidationResult();

        var trainTrends = trainData.Select(x => (int)x.Trend).ToList();
        var testTrends = testData.Select(x => (int)x.Trend).ToList();

        var maxTrainTrend = trainTrends.Any() ? trainTrends.Max() : 0;
        var minTestTrend = testTrends.Any() ? testTrends.Min() : 0;

        result.MaxTrainTrend = maxTrainTrend;
        result.MinTestTrend = minTestTrend;
        result.IsChronological = maxTrainTrend < minTestTrend;

        if (!result.IsChronological)
        {
            _logger.Warning("Temporal order violation detected: Max train trend ({MaxTrain}) >= Min test trend ({MinTest})",
                maxTrainTrend, minTestTrend);
            result.Issues.Add($"Train max trend ({maxTrainTrend}) >= Test min trend ({minTestTrend})");
            
            // Dodatkowa diagnostyka
            var trainCategories = trainData.Select(x => x.ProductCategory).Distinct().Count();
            var testCategories = testData.Select(x => x.ProductCategory).Distinct().Count();
            result.Issues.Add($"Train categories: {trainCategories}, Test categories: {testCategories}");
            result.Issues.Add("PROBLEM: Trend jest per-kategoria zamiast globalny!");
        }

        return result;
    }

    /// <summary>
    /// Sprawdza czy normalizacja nie powoduje leakage
    /// (min/max z test nie powinny wykraczać poza train)
    /// </summary>
    private NormalizationCheckResult CheckNormalizationLeakage(
        List<SalesPredictionInput> trainData,
        List<SalesPredictionInput> testData)
    {
        _logger.Information("Checking normalization leakage...");

        var result = new NormalizationCheckResult();

        // Sprawdź zakresy dla kluczowych features
        result.Checks.Add(CheckFeatureRange("Lag1", 
            trainData.Select(x => (double)x.Lag1),
            testData.Select(x => (double)x.Lag1)));

        result.Checks.Add(CheckFeatureRange("RollingAvg4Weeks",
            trainData.Select(x => (double)x.RollingAvg4Weeks),
            testData.Select(x => (double)x.RollingAvg4Weeks)));

        result.Checks.Add(CheckFeatureRange("CategoryHistoricalAvg",
            trainData.Select(x => (double)x.CategoryHistoricalAvg),
            testData.Select(x => (double)x.CategoryHistoricalAvg)));

        result.IsSafe = result.Checks.All(c => c.IsWithinRange);

        return result;
    }

    private FeatureRangeCheck CheckFeatureRange(
        string featureName,
        IEnumerable<double> trainValues,
        IEnumerable<double> testValues)
    {
        var trainList = trainValues.ToList();
        var testList = testValues.ToList();

        var trainMin = trainList.Min();
        var trainMax = trainList.Max();
        var testMin = testList.Min();
        var testMax = testList.Max();

        var check = new FeatureRangeCheck
        {
            FeatureName = featureName,
            TrainMin = trainMin,
            TrainMax = trainMax,
            TestMin = testMin,
            TestMax = testMax,
            IsWithinRange = testMin >= trainMin * 0.95 && testMax <= trainMax * 1.05 // 5% tolerance
        };

        if (!check.IsWithinRange)
        {
            _logger.Warning("Feature {Feature}: Test range [{TestMin:F2}, {TestMax:F2}] outside Train range [{TrainMin:F2}, {TrainMax:F2}]",
                featureName, testMin, testMax, trainMin, trainMax);
        }

        return check;
    }

    /// <summary>
    /// Sprawdza czy train i test nie mają wspólnych rekordów
    /// </summary>
    private DataOverlapResult CheckTrainTestOverlap(
        List<SalesPredictionInput> trainData,
        List<SalesPredictionInput> testData)
    {
        _logger.Information("Checking train/test overlap...");

        var result = new DataOverlapResult();

        // Porównaj na podstawie kombinacji (ProductCategory, Trend, ActualSales)
        var trainKeys = trainData.Select(x => $"{x.ProductCategory}_{x.Trend}_{x.ActualSales}").ToHashSet();
        var testKeys = testData.Select(x => $"{x.ProductCategory}_{x.Trend}_{x.ActualSales}").ToHashSet();

        var overlap = trainKeys.Intersect(testKeys).ToList();
        result.OverlappingRecords = overlap.Count;
        result.HasOverlap = overlap.Any();

        if (result.HasOverlap)
        {
            _logger.Warning("Found {Count} overlapping records between train and test", overlap.Count);
        }

        return result;
    }

    /// <summary>
    /// Generuje raport tekstowy
    /// </summary>
    public string GenerateReport(LeakageReport report)
    {
        var sb = new StringBuilder();
        sb.AppendLine("═══════════════════════════════════════════════════════");
        sb.AppendLine("            DATA LEAKAGE DETECTION REPORT");
        sb.AppendLine("═══════════════════════════════════════════════════════");
        sb.AppendLine($"Analysis Date: {report.AnalysisDate:yyyy-MM-dd HH:mm:ss}");
        sb.AppendLine($"Train Samples: {report.TrainSamples:N0}");
        sb.AppendLine($"Test Samples:  {report.TestSamples:N0}");
        sb.AppendLine();

        // Feature Correlations
        sb.AppendLine("─── FEATURE CORRELATIONS WITH LABEL ───");
        foreach (var kvp in report.FeatureCorrelations.OrderByDescending(x => Math.Abs(x.Value)))
        {
            var color = Math.Abs(kvp.Value) > 0.95 ? "🔴" :
                        Math.Abs(kvp.Value) > 0.7 ? "🟡" : "🟢";
            sb.AppendLine($"  {color} {kvp.Key,-25}: {kvp.Value,7:F4}");
        }
        sb.AppendLine();

        // Suspicious Features
        if (report.SuspiciousFeatures.Any())
        {
            sb.AppendLine("─── ⚠️  SUSPICIOUS FEATURES (>0.95 correlation) ───");
            foreach (var feature in report.SuspiciousFeatures)
            {
                sb.AppendLine($"  🔴 {feature}");
            }
            sb.AppendLine();
        }

        // Lag Validation
        sb.AppendLine("─── LAG FEATURES VALIDATION ───");
        sb.AppendLine($"  Lag1 correlation: {report.LagValidation.Lag1Correlation:F4}");
        sb.AppendLine($"  Lag2 correlation: {report.LagValidation.Lag2Correlation:F4}");
        sb.AppendLine($"  Lag3 correlation: {report.LagValidation.Lag3Correlation:F4}");
        sb.AppendLine($"  Lag4 correlation: {report.LagValidation.Lag4Correlation:F4}");
        sb.AppendLine($"  Status: {(report.LagValidation.IsValid ? "✅ VALID" : "❌ INVALID")}");
        foreach (var issue in report.LagValidation.Issues)
        {
            sb.AppendLine($"    - {issue}");
        }
        sb.AppendLine();

        // Temporal Validation
        sb.AppendLine("─── TEMPORAL ORDER VALIDATION ───");
        sb.AppendLine($"  Max Train Trend: {report.TemporalValidation.MaxTrainTrend}");
        sb.AppendLine($"  Min Test Trend:  {report.TemporalValidation.MinTestTrend}");
        sb.AppendLine($"  Status: {(report.TemporalValidation.IsChronological ? "✅ CHRONOLOGICAL" : "❌ OVERLAP")}");
        foreach (var issue in report.TemporalValidation.Issues)
        {
            sb.AppendLine($"    - {issue}");
        }
        sb.AppendLine();

        // Normalization Check
        sb.AppendLine("─── NORMALIZATION CHECK ───");
        foreach (var check in report.NormalizationCheck.Checks)
        {
            var status = check.IsWithinRange ? "✅" : "⚠️";
            sb.AppendLine($"  {status} {check.FeatureName}:");
            sb.AppendLine($"      Train: [{check.TrainMin:F2}, {check.TrainMax:F2}]");
            sb.AppendLine($"      Test:  [{check.TestMin:F2}, {check.TestMax:F2}]");
        }
        sb.AppendLine($"  Status: {(report.NormalizationCheck.IsSafe ? "✅ SAFE" : "⚠️  POTENTIAL LEAKAGE")}");
        sb.AppendLine();

        // Data Overlap
        sb.AppendLine("─── TRAIN/TEST OVERLAP CHECK ───");
        sb.AppendLine($"  Overlapping records: {report.DataOverlap.OverlappingRecords}");
        sb.AppendLine($"  Status: {(report.DataOverlap.HasOverlap ? "⚠️  OVERLAP DETECTED" : "✅ NO OVERLAP")}");
        sb.AppendLine();

        // Final Verdict
        sb.AppendLine("═══════════════════════════════════════════════════════");
        var overallStatus = !report.SuspiciousFeatures.Any() &&
                           report.LagValidation.IsValid &&
                           report.TemporalValidation.IsChronological &&
                           report.NormalizationCheck.IsSafe &&
                           !report.DataOverlap.HasOverlap;

        if (overallStatus)
        {
            sb.AppendLine("  ✅ OVERALL: NO DATA LEAKAGE DETECTED");
        }
        else
        {
            sb.AppendLine("  ⚠️  OVERALL: POTENTIAL DATA LEAKAGE DETECTED");
            sb.AppendLine("  Please review the warnings above.");
        }
        sb.AppendLine("═══════════════════════════════════════════════════════");

        return sb.ToString();
    }
}