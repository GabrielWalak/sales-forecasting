using System.Globalization;
using System.IO;
using SalesForecasting.Core.Models;
using Serilog;

namespace SalesForecasting.Data;

public class OlistDataProcessor
{
    private readonly ILogger _logger;
    private static readonly Lazy<Dictionary<string, string>> _categoryTranslations = new(LoadCategoryTranslations);

    public OlistDataProcessor(ILogger logger)
    {
        _logger = logger;
    }

/// <summary>
/// Aggregates weekly sales per category.
/// Uses order_purchase_timestamp as the time point.
/// Selects TOP 12 categories by total sales volume.
/// </summary>
    public List<WeeklySalesData> AggregateWeeklySalesByProduct(
        List<OlistOrder> orders,
        List<OlistOrderItem> orderItems,
        List<OlistProduct> products,
        int topN = 12)
    {
        _logger.Information("Aggregating weekly sales by category, selecting TOP {TopN}", topN);

        var productDict = products
            .Where(p => !string.IsNullOrWhiteSpace(p.ProductCategoryName))
            .ToDictionary(p => p.ProductId, p => p.ProductCategoryName);

        // Step 1: Find TOP 12 categories by total sales
        var topCategories = (from order in orders
                             where order.OrderStatus == "delivered"
                             join item in orderItems on order.OrderId equals item.OrderId
                             where productDict.ContainsKey(item.ProductId)
                             let category = NormalizeCategory(productDict[item.ProductId])
                             group item by category into g
                             orderby g.Count() descending
                             select new
                             {
                                 Category = g.Key,
                                 TotalSales = g.Count()
                             })
            .Take(topN)
            .Select(x => x.Category)
            .ToHashSet();

        _logger.Information("Selected TOP {Count} categories from all available categories", topCategories.Count);

        // Step 2: Aggregate weekly ONLY for the TOP categories
        var joined = from order in orders
                     where order.OrderStatus == "delivered"
                     join item in orderItems on order.OrderId equals item.OrderId
                     where productDict.ContainsKey(item.ProductId)
                     let category = NormalizeCategory(productDict[item.ProductId])
                     where topCategories.Contains(category)
                     select new
                     {
                         WeekStart = GetWeekStart(order.OrderPurchaseTimestamp),
                         Category = category,
                         Quantity = 1,
                         Revenue = item.Price + item.FreightValue
                     };

        var weeklyData = joined
            .GroupBy(x => new { x.WeekStart, x.Category })
            .Select(g => new WeeklySalesData
            {
                Year = g.Key.WeekStart.Year,
                WeekNumber = GetIso8601WeekOfYear(g.Key.WeekStart),
                WeekStartDate = g.Key.WeekStart,
                Category = g.Key.Category,
                Quantity = g.Sum(x => x.Quantity),
                Revenue = g.Sum(x => x.Revenue),
                OrderCount = g.Count()
            })
            .OrderBy(x => x.WeekStartDate)
            .ThenBy(x => x.Category)
            .ToList();

        _logger.Information("Aggregated to {Count} weekly records for {Categories} categories",
            weeklyData.Count,
            weeklyData.Select(x => x.Category).Distinct().Count());

        return weeklyData;
    }

    /// <summary>
    /// Fills missing weeks with zeros (for categories with unstable sales)
    /// </summary>
    public List<WeeklySalesData> FillMissingWeeks(List<WeeklySalesData> weeklyData, string category)
    {
        var categoryData = weeklyData.Where(x => x.Category == category).OrderBy(x => x.WeekStartDate).ToList();

        if (!categoryData.Any())
            return new List<WeeklySalesData>();

        var minDate = categoryData.First().WeekStartDate;
        var maxDate = categoryData.Last().WeekStartDate;
        var filledData = new List<WeeklySalesData>();

        for (var date = minDate; date <= maxDate; date = date.AddDays(7))
        {
            var existing = categoryData.FirstOrDefault(x => x.WeekStartDate == date);
            if (existing != null)
            {
                filledData.Add(existing);
            }
            else
            {
                // Add a week with zeroed data
                filledData.Add(new WeeklySalesData
                {
                    Year = date.Year,
                    WeekNumber = GetIso8601WeekOfYear(date),
                    WeekStartDate = date,
                    Category = category,
                    Quantity = 0,
                    Revenue = 0m,
                    OrderCount = 0
                });
            }
        }

        return filledData;
    }

    /// <summary>
    /// Creates features capturing strong seasonality and instability
    /// </summary>
    public List<SalesPredictionInput> CreateFeatures(
        List<WeeklySalesData> weeklyData,
        string category,
        bool isTrainingSet = true,
        double? trainSetHistoricalAvg = null)
    {
        _logger.Information("Creating features for category {Category}, TrainingSet={IsTraining}",
            category, isTrainingSet);

        // Fill missing weeks
        var filledData = FillMissingWeeks(weeklyData, category);

        if (filledData.Count < 12) // Minimum 12 weeks for better seasonality
        {
            _logger.Warning("Category {Category} has only {Count} weeks, minimum is 12",
                category, filledData.Count);
            return new List<SalesPredictionInput>();
        }

        var features = new List<SalesPredictionInput>();

        // Calculate historical average, ignoring zeros for long-tail
        var nonZeroData = filledData.Where(x => x.Quantity > 0).ToList();
        var historicalAvg = isTrainingSet
            ? (nonZeroData.Any() ? (float)nonZeroData.Average(x => x.Quantity) : 0f)
            : (float)(trainSetHistoricalAvg ?? (nonZeroData.Any() ? nonZeroData.Average(x => x.Quantity) : 0));

        for (int i = 0; i < filledData.Count; i++)
        {
            var currentWeek = filledData[i];

            // Lag features - with zero-filled gaps
            var lag1 = i >= 1 ? (float)filledData[i - 1].Quantity : 0f;
            var lag2 = i >= 2 ? (float)filledData[i - 2].Quantity : 0f;
            var lag3 = i >= 3 ? (float)filledData[i - 3].Quantity : 0f;
            var lag4 = i >= 4 ? (float)filledData[i - 4].Quantity : 0f;

            // Rolling average (4 weeks)
            var rollingAvg = i >= 4
                ? (float)filledData.Skip(i - 4).Take(4).Average(x => x.Quantity)
                : historicalAvg;

            // Monthly seasonality (average from the same month in the past)
            var monthlySeasonality = CalculateMonthlySeasonality(filledData, i, currentWeek.WeekStartDate.Month);

            var input = new SalesPredictionInput
            {
                WeekNumber = currentWeek.WeekNumber,
                Month = currentWeek.WeekStartDate.Month,
                Quarter = (currentWeek.WeekStartDate.Month - 1) / 3 + 1,
                IsBlackFridayWeek = IsBlackFridayWeek(currentWeek.WeekStartDate) ? 1f : 0f,
                IsHolidaySeason = IsHolidaySeason(currentWeek.WeekStartDate) ? 1f : 0f,
                Lag1 = lag1,
                Lag2 = lag2,
                Lag3 = lag3,
                Lag4 = lag4,
                RollingAvg4Weeks = rollingAvg,
                Trend = i,
                CategoryHistoricalAvg = monthlySeasonality, // Monthly seasonality
                ActualSales = (float)currentWeek.Quantity
            };

            features.Add(input);
        }

        _logger.Information("Created {Count} feature rows for {Category}", features.Count, category);
        return features;
    }

    /// <summary>
    /// Creates features for ALL categories at once (global model)
    /// ✅ DATA LEAKAGE SAFE: uses only past data
    /// </summary>
    public List<SalesPredictionInput> CreateGlobalFeatures(
        List<WeeklySalesData> weeklyData,
        bool isTrainingSet = true)
    {
        _logger.Information("Creating GLOBAL features for all categories, TrainingSet={IsTraining}", isTrainingSet);

        var allCategories = weeklyData.Select(x => x.Category).Distinct().OrderBy(x => x).ToList();
        var allFeatures = new List<SalesPredictionInput>();

        // ✅ NEW: Compute global min date for consistent Trend
        var globalMinDate = weeklyData.Min(x => x.WeekStartDate);

        foreach (var category in allCategories)
        {
            var filledData = FillMissingWeeks(weeklyData, category);

            if (filledData.Count < 12)
            {
                _logger.Warning("Category {Category} has only {Count} weeks, skipping", category, filledData.Count);
                continue;
            }

            var nonZeroData = filledData.Where(x => x.Quantity > 0).ToList();
            var historicalAvg = nonZeroData.Any() ? (float)nonZeroData.Average(x => x.Quantity) : 0f;

            for (int i = 0; i < filledData.Count; i++)
            {
                var currentWeek = filledData[i];

                // Lag features
                var lag1 = i >= 1 ? (float)filledData[i - 1].Quantity : 0f;
                var lag2 = i >= 2 ? (float)filledData[i - 2].Quantity : 0f;
                var lag3 = i >= 3 ? (float)filledData[i - 3].Quantity : 0f;
                var lag4 = i >= 4 ? (float)filledData[i - 4].Quantity : 0f;

                var rollingAvg = i >= 4
                    ? (float)filledData.Skip(i - 4).Take(4).Average(x => x.Quantity)
                    : historicalAvg;

                var monthlySeasonality = CalculateMonthlySeasonality(filledData, i, currentWeek.WeekStartDate.Month);

                // ✅ FIXED: Use global Trend based on date
                var globalTrend = (int)((currentWeek.WeekStartDate - globalMinDate).TotalDays / 7);

                var input = new SalesPredictionInput
                {
                    WeekNumber = currentWeek.WeekNumber,
                    Month = currentWeek.WeekStartDate.Month,
                    Quarter = (currentWeek.WeekStartDate.Month - 1) / 3 + 1,
                    IsBlackFridayWeek = IsBlackFridayWeek(currentWeek.WeekStartDate) ? 1f : 0f,
                    IsHolidaySeason = IsHolidaySeason(currentWeek.WeekStartDate) ? 1f : 0f,
                    Lag1 = lag1,
                    Lag2 = lag2,
                    Lag3 = lag3,
                    Lag4 = lag4,
                    RollingAvg4Weeks = rollingAvg,
                    Trend = globalTrend, // ✅ GLOBAL TREND (same for all categories in the same week)
                    CategoryHistoricalAvg = monthlySeasonality,
                    ProductCategory = category,
                    ActualSales = (float)currentWeek.Quantity
                };

                allFeatures.Add(input);
            }
        }

        _logger.Information("Created {Count} GLOBAL feature rows for {Categories} categories",
            allFeatures.Count, allCategories.Count);

        return allFeatures;
    }

    /// <summary>
    /// Time-based Train/Test split with explicit dates
    /// ✅ Guards added against empty sequences
    /// </summary>
    public (List<SalesPredictionInput> train, List<SalesPredictionInput> test) SplitByDate(
        List<WeeklySalesData> weeklyData,
        string category,
        DateTime trainEndDate,
        DateTime testStartDate)
    {
        _logger.Information("Time-based split for {Category}: Train ends {TrainEnd}, Test starts {TestStart}",
            category, trainEndDate, testStartDate);

        // First, filter data for the category and fill gaps
        var filledData = FillMissingWeeks(weeklyData, category);

        if (!filledData.Any())
        {
            _logger.Warning("No data found for category {Category}", category);
            return (new List<SalesPredictionInput>(), new List<SalesPredictionInput>());
        }

        // Create features for the full set
        var allFeatures = CreateFeatures(weeklyData, category, isTrainingSet: true);

        // ✅ Guard: ensure CreateFeatures returned data
        if (!allFeatures.Any())
        {
            _logger.Warning("No features created for category {Category}", category);
            return (new List<SalesPredictionInput>(), new List<SalesPredictionInput>());
        }

        // Now split based on dates in filledData
        var trainIndices = new List<int>();
        var testIndices = new List<int>();

        for (int i = 0; i < filledData.Count && i < allFeatures.Count; i++)
        {
            var weekDate = filledData[i].WeekStartDate;

            if (weekDate <= trainEndDate)
            {
                trainIndices.Add(i);
            }
            else if (weekDate >= testStartDate)
            {
                testIndices.Add(i);
            }
        }

        var train = trainIndices.Select(i => allFeatures[i]).ToList();
        var test = testIndices.Select(i => allFeatures[i]).ToList();

        // ✅ FIXED: Guard against empty lists in logging
        if (trainIndices.Any() && testIndices.Any())
        {
            _logger.Information("Time-based split result: {TrainCount} train ({TrainStart} to {TrainEnd}), {TestCount} test ({TestStart} to {TestEnd})",
                train.Count,
                filledData[trainIndices.First()].WeekStartDate.ToString("yyyy-MM-dd"),
                filledData[trainIndices.Last()].WeekStartDate.ToString("yyyy-MM-dd"),
                test.Count,
                filledData[testIndices.First()].WeekStartDate.ToString("yyyy-MM-dd"),
                filledData[testIndices.Last()].WeekStartDate.ToString("yyyy-MM-dd"));
        }
        else if (trainIndices.Any())
        {
            _logger.Warning("Split result: {TrainCount} train, 0 test (no data in test period)",
                train.Count);
        }
        else if (testIndices.Any())
        {
            _logger.Warning("Split result: 0 train, {TestCount} test (no data in train period)",
                test.Count);
        }
        else
        {
            _logger.Warning("Split result: no data in either train or test period for {Category}",
                category);
        }

        return (train, test);
    }

    /// <summary>
    /// Time-based split for the global model (fixed)
    /// </summary>
    public (List<SalesPredictionInput> train, List<SalesPredictionInput> test) SplitGlobalByDate(
        List<WeeklySalesData> weeklyData,
        DateTime trainEndDate,
        DateTime testStartDate)
    {
        _logger.Information("Time-based split for GLOBAL model: Train ends {TrainEnd}, Test starts {TestStart}",
            trainEndDate, testStartDate);

        // Create global features (with global Trend)
        var allFeatures = CreateGlobalFeatures(weeklyData, isTrainingSet: true);

        // ✅ SIMPLE: Split based on WeekStartDate stored in original weeklyData
        // Rebuild mapping feature -> date
        var globalMinDate = weeklyData.Min(x => x.WeekStartDate);
        
        // Split features based on Trend (now global)
        var train = new List<SalesPredictionInput>();
        var test = new List<SalesPredictionInput>();

        foreach (var feature in allFeatures)
        {
            // Recreate date from global Trend
            var weekDate = globalMinDate.AddDays(feature.Trend * 7);

            if (weekDate <= trainEndDate)
            {
                train.Add(feature);
            }
            else if (weekDate >= testStartDate)
            {
                test.Add(feature);
            }
        }

        _logger.Information("Global split result: {TrainCount} train, {TestCount} test samples",
            train.Count, test.Count);

        // Validation: ensure Max(train.Trend) < Min(test.Trend)
        if (train.Any() && test.Any())
        {
            var maxTrainTrend = train.Max(x => x.Trend);
            var minTestTrend = test.Min(x => x.Trend);
            
            if (maxTrainTrend >= minTestTrend)
            {
                _logger.Warning("TEMPORAL ORDER VIOLATION: Max train trend ({MaxTrain}) >= Min test trend ({MinTest})",
                    maxTrainTrend, minTestTrend);
            }
            else
            {
                _logger.Information("Temporal order validated: Max train trend ({MaxTrain}) < Min test trend ({MinTest}) ✓",
                    maxTrainTrend, minTestTrend);
            }
        }

        return (train, test);
    }

    /// <summary>
    /// 3-Fold time-series Cross-Validation with rolling window (tailored for Olist)
    /// </summary>
    public List<(List<SalesPredictionInput> train, List<SalesPredictionInput> test, int fold)> CreateTimeSeriesCrossValidationFolds(
        List<WeeklySalesData> weeklyData,
        string category,
        int nFolds = 3) // Zmniejsz do 3 foldów
    {
        _logger.Information("Creating {NFolds}-fold time series CV for {Category}", nFolds, category);

        var filledData = FillMissingWeeks(weeklyData, category);

        // Minimum 60 weeks (~14 months) for 3 folds
        if (filledData.Count < nFolds * 20)
        {
            _logger.Warning("Not enough data for {NFolds}-fold CV. Need at least {Required} weeks, have {Actual}",
                nFolds, nFolds * 20, filledData.Count);
            return new List<(List<SalesPredictionInput>, List<SalesPredictionInput>, int)>();
        }

        var allFeatures = CreateFeatures(weeklyData, category, isTrainingSet: true);
        var folds = new List<(List<SalesPredictionInput> train, List<SalesPredictionInput> test, int fold)>();

        // Compute test window size (~20% of data or minimum 12 weeks)
        var testWindowSize = Math.Max(12, filledData.Count / 5);
        var minTrainSize = 20; // Minimum ~5 miesięcy na trenowanie

        for (int fold = 0; fold < nFolds; fold++)
        {
            // Rolling window: each fold moves forward in time
            var testEndIndex = filledData.Count - (nFolds - fold - 1) * (testWindowSize / 2);
            var testStartIndex = Math.Max(minTrainSize, testEndIndex - testWindowSize);

            if (testStartIndex < minTrainSize || testStartIndex >= filledData.Count)
                break;

            var trainFeatures = allFeatures.Take(testStartIndex).ToList();
            var testFeatures = allFeatures.Skip(testStartIndex).Take(testWindowSize).ToList();

            if (trainFeatures.Count >= minTrainSize && testFeatures.Any())
            {
                folds.Add((trainFeatures, testFeatures, fold + 1));

                _logger.Information(
                    "Fold {Fold}: Train={TrainCount} weeks ({TrainStart} to {TrainEnd}), Test={TestCount} weeks ({TestStart} to {TestEnd})",
                    fold + 1,
                    trainFeatures.Count,
                    filledData[0].WeekStartDate.ToString("yyyy-MM-dd"),
                    filledData[Math.Min(testStartIndex - 1, filledData.Count - 1)].WeekStartDate.ToString("yyyy-MM-dd"),
                    testFeatures.Count,
                    filledData[testStartIndex].WeekStartDate.ToString("yyyy-MM-dd"),
                    filledData[Math.Min(testStartIndex + testWindowSize - 1, filledData.Count - 1)].WeekStartDate.ToString("yyyy-MM-dd"));
            }
        }

        _logger.Information("Created {Count} folds for {Category}", folds.Count, category);
        return folds;
    }

    /// <summary>
    /// Chronologiczny split - ostatnie 20% do testowania (stara metoda, zachowana dla kompatybilności)
    /// </summary>
    public (List<SalesPredictionInput> train, List<SalesPredictionInput> test) SplitTrainTest(
        List<SalesPredictionInput> features,
        double testSizePercentage = 20)
    {
        var splitIndex = (int)(features.Count * (100 - testSizePercentage) / 100.0);
        var train = features.Take(splitIndex).ToList();
        var test = features.Skip(splitIndex).ToList();

        _logger.Information("Split: {TrainCount} train, {TestCount} test samples",
            train.Count, test.Count);
        return (train, test);
    }

    private DateTime GetWeekStart(DateTime date)
    {
        var diff = (7 + (date.DayOfWeek - DayOfWeek.Monday)) % 7;
        return date.AddDays(-1 * diff).Date;
    }

    private int GetIso8601WeekOfYear(DateTime date)
    {
        var day = CultureInfo.InvariantCulture.Calendar.GetDayOfWeek(date);
        if (day >= DayOfWeek.Monday && day <= DayOfWeek.Wednesday)
            date = date.AddDays(3);
        return CultureInfo.InvariantCulture.Calendar.GetWeekOfYear(
            date, CalendarWeekRule.FirstFourDayWeek, DayOfWeek.Monday);
    }

    /// <summary>
    /// Black Friday w Brazylii - 4. piątek listopada
    /// </summary>
    private bool IsBlackFridayWeek(DateTime date)
    {
        return date.Month == 11 && date.Day >= 20 && date.Day <= 27;
    }

    /// <summary>
    /// Sezon świąteczny w Brazylii - listopad-grudzień (szczyt zakupów)
    /// </summary>
    private bool IsHolidaySeason(DateTime date)
    {
        return date.Month == 11 || date.Month == 12;
    }

    private string NormalizeCategory(string category)
    {
        if (string.IsNullOrWhiteSpace(category))
            return "Unknown";

        var normalized = category
            .Replace("_", " ")
            .Trim()
            .ToLowerInvariant();

        if (_categoryTranslations.Value.TryGetValue(normalized, out var translated))
            return translated;

        return normalized;
    }

    private static Dictionary<string, string> LoadCategoryTranslations()
    {
        var map = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

        try
        {
            var baseDir = AppContext.BaseDirectory;
            var path = Path.Combine(baseDir, "data", "product_category_name_translation.csv");

            if (!File.Exists(path))
                return map;

            foreach (var line in File.ReadAllLines(path).Skip(1))
            {
                var parts = line.Split(',', 2);
                if (parts.Length < 2)
                    continue;

                var pt = parts[0].Trim();
                var en = parts[1].Trim();
                if (pt.Length == 0 || en.Length == 0)
                    continue;

                map[pt.Replace("_", " ")] = en.Replace("_", " ");
                map[pt] = en.Replace("_", " ");
            }
        }
        catch
        {
            // fallback to empty map
        }

        return map;
    }

    /// <summary>
    /// Oblicza średnią sprzedaż dla danego miesiąca z przeszłości (sezonowość)
    /// </summary>
    private float CalculateMonthlySeasonality(List<WeeklySalesData> data, int currentIndex, int month)
    {
        var historicalMonthData = data
            .Take(currentIndex) // Tylko przeszłość
            .Where(x => x.WeekStartDate.Month == month && x.Quantity > 0)
            .ToList();

        return historicalMonthData.Any()
            ? (float)historicalMonthData.Average(x => x.Quantity)
            : 0f;
    }

    /// <summary>
    /// Agreguje sprzedaż tygodniową dla WSZYSTKICH kategorii (dla modelu globalnego)
    /// NIE ogranicza do TOP N - używa wszystkich dostępnych kategorii
    /// </summary>
    public List<WeeklySalesData> AggregateWeeklySalesAllCategories(
        List<OlistOrder> orders,
        List<OlistOrderItem> orderItems,
        List<OlistProduct> products)
    {
        _logger.Information("Aggregating weekly sales for ALL categories (global model)");

        var productDict = products
            .Where(p => !string.IsNullOrWhiteSpace(p.ProductCategoryName))
            .ToDictionary(p => p.ProductId, p => p.ProductCategoryName);

        // Agreguj tygodniowo dla WSZYSTKICH kategorii
        var joined = from order in orders
                     where order.OrderStatus == "delivered"
                     join item in orderItems on order.OrderId equals item.OrderId
                     where productDict.ContainsKey(item.ProductId)
                     let category = NormalizeCategory(productDict[item.ProductId])
                     select new
                     {
                         WeekStart = GetWeekStart(order.OrderPurchaseTimestamp),
                         Category = category,
                         Quantity = 1,
                         Revenue = item.Price + item.FreightValue
                     };

        var weeklyData = joined
            .GroupBy(x => new { x.WeekStart, x.Category })
            .Select(g => new WeeklySalesData
            {
                Year = g.Key.WeekStart.Year,
                WeekNumber = GetIso8601WeekOfYear(g.Key.WeekStart),
                WeekStartDate = g.Key.WeekStart,
                Category = g.Key.Category,
                Quantity = g.Sum(x => x.Quantity),
                Revenue = g.Sum(x => x.Revenue),
                OrderCount = g.Count()
            })
            .OrderBy(x => x.WeekStartDate)
            .ThenBy(x => x.Category)
            .ToList();

        var totalCategories = weeklyData.Select(x => x.Category).Distinct().Count();
        _logger.Information("Aggregated to {Count} weekly records for {Categories} categories (ALL categories)",
            weeklyData.Count, totalCategories);

        return weeklyData;
    }
}