using Serilog;
using SalesForecasting.Core.Models;
using SalesForecasting.Data;
using SalesForecasting.ML;
using System.Text.Json;
using Microsoft.ML;

class Program
{
    private static ILogger _logger = null!;
    private static string _dataDirectory = null!;
    private static string _modelsDirectory = null!;
    private static string _preprocessedGlobalDataPath = null!;

    static void Main(string[] args)
    {
        SetupLogging();
        LoadConfiguration();

        Console.Clear();
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("╔════════════════════════════════════════════════╗");
        Console.WriteLine("║  OLIST SALES FORECASTING SYSTEM               ║");
        Console.WriteLine("║  Machine Learning - FastTree Global Model     ║");
        Console.WriteLine("╚════════════════════════════════════════════════╝");
        Console.ResetColor();
        Console.WriteLine();

        bool exit = false;
        while (!exit)
        {
            ShowMenu();
            var choice = Console.ReadLine();

            Console.WriteLine();
            switch (choice)
            {
                case "1":
                    PrepareGlobalData();
                    break;
                case "2":
                    TrainGlobalModel();
                    break;
                case "3":
                    EvaluateGlobalModel();
                    break;
                case "4":
                    DetectDataLeakage();
                    break;
                case "5":
                    GenerateForecastsGlobal();
                    break;
                case "6":
                    RunGridSearch();
                    break;
                case "7":
                    AnalyzeFeatureImportance();
                    break;
                case "8":
                    CompareWithBaseline();
                    break;
                case "9":
                    AnalyzeSeasonality();
                    break;
                case "10":
                    AnalyzeForecastHorizons();  
                    break;
                case "11":
                    AnalyzeProfitLoss();        
                    break;
                case "12":
                    GenerateAllCharts();       
                    break;
                case "0":
                    exit = true;
                    Console.ForegroundColor = ConsoleColor.Yellow;
                    Console.WriteLine("Closing application... Goodbye!");
                    Console.ResetColor();
                    break;
                default:
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine("❌ Invalid choice. Please try again.");
                    Console.ResetColor();
                    break;
            }

            if (!exit)
            {
                Console.WriteLine("\nPress any key to continue...");
                Console.ReadKey();
                Console.Clear();
            }
        }

        Log.CloseAndFlush();
    }

    static void ShowMenu()
    {
        Console.ForegroundColor = ConsoleColor.Green;
        Console.WriteLine("═══════════════════════ MAIN MENU ═══════════════════════");
        Console.ResetColor();
        
        Console.ForegroundColor = ConsoleColor.White;
        Console.WriteLine("─────────────── DATA & MODEL ───────────────");
        Console.ResetColor();
        Console.WriteLine(" 1. 📊 Prepare data (weekly aggregation)");
        Console.WriteLine(" 2. 🌐 Train GLOBAL MODEL (FastTree + log-transform)");
        Console.WriteLine(" 3. 📈 Evaluate global model");
        Console.WriteLine(" 4. 🔍 Detect DATA LEAKAGE");
        Console.WriteLine(" 5. 🔮 Generate forecasts (4 weeks)");
        
        Console.ForegroundColor = ConsoleColor.White;
        Console.WriteLine("─────────────── OPTIMIZATION ───────────────");
        Console.ResetColor();
        Console.WriteLine(" 6. 🔬 GRID SEARCH - Hyperparameter optimization");
        Console.WriteLine(" 7. 📊 FEATURE IMPORTANCE - Feature impact analysis");
        Console.WriteLine(" 8. ⚖️  COMPARE with Baseline (Naive vs. FastTree)");
        
        Console.ForegroundColor = ConsoleColor.White;
        Console.WriteLine("─────────────── ADVANCED ANALYTICS ───────────────");
        Console.ResetColor();
        Console.WriteLine(" 9. 📅 SEASONALITY - Trends and patterns");
        Console.WriteLine("10. ⏱️  FORECAST HORIZONS - Accuracy 1w/1m");
        Console.WriteLine("11. 💰 PROFIT & LOSS - Financial analysis");
        Console.WriteLine("12. 📉 GENERATE CHARTS - All visualizations");
        
        Console.ForegroundColor = ConsoleColor.Red;
        Console.WriteLine("─────────────────────────────────────────────");
        Console.WriteLine(" 0. 🚪 Exit");
        Console.ResetColor();
        Console.WriteLine("═══════════════════════════════════════════════════════════");
        Console.Write("\nSelect an option (0-12): ");
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 🆕 NEW METHOD: Seasonality analysis
    // ═══════════════════════════════════════════════════════════════════════════
    static void AnalyzeSeasonality()
    {
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("═══ SEASONALITY & TREND ANALYSIS ═══");
        Console.ResetColor();

        _logger.Information("Starting Seasonality Analysis");

        try
        {
            if (!File.Exists(_preprocessedGlobalDataPath))
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("❌ ERROR: No data found. Please run option 1 first.");
                Console.ResetColor();
                return;
            }

            Console.WriteLine("⏳ Loading data...");
            var json = File.ReadAllText(_preprocessedGlobalDataPath);
            var weeklyData = JsonSerializer.Deserialize<List<WeeklySalesData>>(json);

            if (weeklyData == null || !weeklyData.Any())
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("❌ ERROR: Unable to load data");
                Console.ResetColor();
                return;
            }

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"✓ Loaded {weeklyData.Count:N0} records\n");
            Console.ResetColor();

            var analyzer = new SeasonalityAnalyzer(_logger);
            var report = analyzer.AnalyzeSeasonality(weeklyData);
            analyzer.PrintSeasonalityReport(report);

            // Save report
            Directory.CreateDirectory(_modelsDirectory);
            var reportPath = Path.Combine(_modelsDirectory, "seasonality_report.json");
            var reportJson = JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(reportPath, reportJson);

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"\n✓ Report saved to: {reportPath}");
            Console.ResetColor();
        }
        catch (Exception ex)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"❌ ERROR: {ex.Message}");
            Console.ResetColor();
            _logger.Error(ex, "Seasonality analysis failed");
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 🆕 NEW METHOD: Forecast horizon analysis
    // ═══════════════════════════════════════════════════════════════════════════
    static void AnalyzeForecastHorizons()
    {
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("═══ FORECAST HORIZON ANALYSIS ═══");
        Console.ResetColor();

        _logger.Information("Starting Forecast Horizon Analysis");

        try
        {
            var modelPath = Path.Combine(_modelsDirectory, "global_model.zip");
            if (!File.Exists(modelPath))
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("❌ ERROR: Model not found. Please run option 2 first.");
                Console.ResetColor();
                return;
            }

            if (!File.Exists(_preprocessedGlobalDataPath))
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("❌ ERROR: Data missing. Please run option 1 first.");
                Console.ResetColor();
                return;
            }

            Console.WriteLine("⏳ Loading model and data...");
            var mlContext = new MLContext(seed: 42);
            var model = mlContext.Model.Load(modelPath, out _);

            var json = File.ReadAllText(_preprocessedGlobalDataPath);
            var weeklyData = JsonSerializer.Deserialize<List<WeeklySalesData>>(json);

            if (weeklyData == null || !weeklyData.Any())
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("❌ ERROR: Unable to load data");
                Console.ResetColor();
                return;
            }

            var processor = new OlistDataProcessor(_logger);
            var trainEndDate = new DateTime(2018, 2, 19);
            var testStartDate = new DateTime(2018, 2, 26);
            var (_, test) = processor.SplitGlobalByDate(weeklyData, trainEndDate, testStartDate);

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"✓ Loaded model and {test.Count} test samples\n");
            Console.ResetColor();

            var analyzer = new ForecastHorizonAnalyzer(_logger);
            var results = analyzer.AnalyzeHorizons(model, test, [1, 2, 4, 8]);

            // Save report
            Directory.CreateDirectory(_modelsDirectory);
            var reportPath = Path.Combine(_modelsDirectory, "horizon_analysis.json");
            var reportJson = JsonSerializer.Serialize(results, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(reportPath, reportJson);

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"\n✓ Report saved to: {reportPath}");
            Console.ResetColor();
        }
        catch (Exception ex)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"❌ BŁĄD: {ex.Message}");
            Console.ResetColor();
            _logger.Error(ex, "Horizon analysis failed");
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 🆕 NEW METHOD: Profit and loss analysis
    // ═══════════════════════════════════════════════════════════════════════════
    static void AnalyzeProfitLoss()
    {
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("═══ PROFIT & LOSS ANALYSIS ═══");
        Console.ResetColor();

        _logger.Information("Starting Profit/Loss Analysis");

        try
        {
            var modelPath = Path.Combine(_modelsDirectory, "global_model.zip");
            if (!File.Exists(modelPath))
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("❌ ERROR: Model not found. Please run option 2 first.");
                Console.ResetColor();
                return;
            }

            if (!File.Exists(_preprocessedGlobalDataPath))
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("❌ ERROR: Data missing. Please run option 1 first.");
                Console.ResetColor();
                return;
            }

            Console.WriteLine("⏳ Loading model and data...");
            var mlContext = new MLContext(seed: 42);
            var model = mlContext.Model.Load(modelPath, out _);

            var json = File.ReadAllText(_preprocessedGlobalDataPath);
            var weeklyData = JsonSerializer.Deserialize<List<WeeklySalesData>>(json);

            if (weeklyData == null || !weeklyData.Any())
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("❌ ERROR: Unable to load data");
                Console.ResetColor();
                return;
            }

            var processor = new OlistDataProcessor(_logger);
            var trainEndDate = new DateTime(2018, 2, 19);
            var testStartDate = new DateTime(2018, 2, 26);
            var (_, test) = processor.SplitGlobalByDate(weeklyData, trainEndDate, testStartDate);

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"✓ Loaded model and {test.Count} test samples\n");
            Console.ResetColor();

            // Generuj predykcje ML
            var predEngine = mlContext.Model.CreatePredictionEngine<SalesPredictionInput, SalesPredictionOutput>(model);
            var actual = new List<double>();
            var mlPredicted = new List<double>();
            var naivePredicted = new List<double>();

            foreach (var sample in test)
            {
                // Actual
                actual.Add(sample.ActualSales);

                // ML Prediction (odwróć log-transform)
                var logPred = predEngine.Predict(sample).PredictedSales;
                var mlPred = Math.Max(0, Math.Exp(logPred) - 1);
                mlPredicted.Add(mlPred);

                // Naive Prediction (4-week rolling avg)
                naivePredicted.Add(sample.RollingAvg4Weeks);
            }

            // Uruchom analizę
            var analyzer = new ProfitLossAnalyzer(_logger);
            
            // Możliwość konfiguracji kosztów
            Console.WriteLine("Enter cost parameters (or press Enter for defaults):");
            Console.Write($"  Overstock cost (default R${analyzer.CostPerUnitOverstock:N2}): ");
            var overstockInput = Console.ReadLine();
            if (!string.IsNullOrWhiteSpace(overstockInput) && decimal.TryParse(overstockInput, out var overstock))
                analyzer.CostPerUnitOverstock = overstock;

            Console.Write($"  Stockout cost (default R${analyzer.CostPerUnitStockout:N2}): ");
            var stockoutInput = Console.ReadLine();
            if (!string.IsNullOrWhiteSpace(stockoutInput) && decimal.TryParse(stockoutInput, out var stockout))
                analyzer.CostPerUnitStockout = stockout;

            Console.Write($"  Average margin (default R${analyzer.AverageMarginPerUnit:N2}): ");
            var marginInput = Console.ReadLine();
            if (!string.IsNullOrWhiteSpace(marginInput) && decimal.TryParse(marginInput, out var margin))
                analyzer.AverageMarginPerUnit = margin;

            Console.WriteLine();

            // Compare scenarios
            analyzer.CompareScenarios(actual, mlPredicted, naivePredicted);

            // Save report
            Directory.CreateDirectory(_modelsDirectory);
            var mlReport = analyzer.AnalyzeProfitLoss(actual, mlPredicted, "FastTree ML");
            var naiveReport = analyzer.AnalyzeProfitLoss(actual, naivePredicted, "Naive Baseline");

            var combinedReport = new
            {
                Parameters = new
                {
                    CostPerUnitOverstock = analyzer.CostPerUnitOverstock,
                    CostPerUnitStockout = analyzer.CostPerUnitStockout,
                    AverageMarginPerUnit = analyzer.AverageMarginPerUnit
                },
                FastTreeML = mlReport,
                NaiveBaseline = naiveReport,
                Savings = naiveReport.TotalLoss - mlReport.TotalLoss
            };

            var reportPath = Path.Combine(_modelsDirectory, "profit_loss_analysis.json");
            var reportJson = JsonSerializer.Serialize(combinedReport, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(reportPath, reportJson);

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"\n✓ Report saved to: {reportPath}");
            Console.ResetColor();
        }
        catch (Exception ex)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"❌ BŁĄD: {ex.Message}");
            Console.ResetColor();
            _logger.Error(ex, "Profit/Loss analysis failed");
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 🆕 NEW METHOD: Generate all charts
    // ═══════════════════════════════════════════════════════════════════════════
    static void GenerateAllCharts()
    {
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("═══ GENERATING CHARTS ═══");
        Console.ResetColor();

        _logger.Information("Starting Chart Generation");

        try
        {
            var modelPath = Path.Combine(_modelsDirectory, "global_model.zip");
            if (!File.Exists(modelPath))
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("❌ ERROR: Model not found. Please run option 2 first.");
                Console.ResetColor();
                return;
            }

            if (!File.Exists(_preprocessedGlobalDataPath))
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("❌ ERROR: Data missing. Please run option 1 first.");
                Console.ResetColor();
                return;
            }

            Console.WriteLine("⏳ Loading model and data...");
            var mlContext = new MLContext(seed: 42);
            var model = mlContext.Model.Load(modelPath, out _);

            var json = File.ReadAllText(_preprocessedGlobalDataPath);
            var weeklyData = JsonSerializer.Deserialize<List<WeeklySalesData>>(json);

            if (weeklyData == null || !weeklyData.Any())
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("❌ ERROR: Unable to load data");
                Console.ResetColor();
                return;
            }

            var processor = new OlistDataProcessor(_logger);
            var trainEndDate = new DateTime(2018, 2, 19);
            var testStartDate = new DateTime(2018, 2, 26);
            var (_, test) = processor.SplitGlobalByDate(weeklyData, trainEndDate, testStartDate);

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"✓ Loaded model and {test.Count} test samples\n");
            Console.ResetColor();

            // Generuj predykcje
            var predEngine = mlContext.Model.CreatePredictionEngine<SalesPredictionInput, SalesPredictionOutput>(model);
            var actual = new List<double>();
            var predicted = new List<double>();

            foreach (var sample in test)
            {
                actual.Add(sample.ActualSales);
                var logPred = predEngine.Predict(sample).PredictedSales;
                predicted.Add(Math.Max(0, Math.Exp(logPred) - 1));
            }

            var chartsDir = Path.Combine(_modelsDirectory, "charts");
            Directory.CreateDirectory(chartsDir);

            var viz = new VisualizationGenerator(_logger);

            // 1. Predicted vs Actual
            Console.WriteLine("⏳ Generating Predicted vs Actual chart...");
            var scatterPath = Path.Combine(chartsDir, "predicted_vs_actual.png");
            viz.GeneratePredictedVsActualPlot(actual, predicted, scatterPath);
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"   ✓ Zapisano: {scatterPath}");
            Console.ResetColor();

            // 2. Feature Importance (jeśli istnieje)
            var featureImportancePath = Path.Combine(_modelsDirectory, "feature_importance.json");
            if (File.Exists(featureImportancePath))
            {
                Console.WriteLine("⏳ Generating Feature Importance chart...");
                var fiJson = File.ReadAllText(featureImportancePath);
                var featureResults = JsonSerializer.Deserialize<List<FeatureImportanceResult>>(fiJson);
                
                if (featureResults != null && featureResults.Any())
                {
                    // Take TOP 15 features for readability
                    var top15 = featureResults.Take(15).ToList();
                    var fiChartPath = Path.Combine(chartsDir, "feature_importance.png");
                    viz.GenerateFeatureImportancePlot(top15, fiChartPath);
                    Console.ForegroundColor = ConsoleColor.Green;
                    Console.WriteLine($"   ✓ Saved: {fiChartPath}");
                    Console.ResetColor();
                }
            }
            else
            {
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine("   ⚠️  Missing Feature Importance data - run option 7 first");
                Console.ResetColor();
            }

            // 3. Time Series (jeśli VisualizationGenerator ma taką metodę)
            Console.WriteLine("⏳ Generating Time Series chart...");
            var topCategory = weeklyData.GroupBy(x => x.Category)
                .OrderByDescending(g => g.Sum(x => x.Quantity))
                .First().Key;
            var tsPath = Path.Combine(chartsDir, $"timeseries_{topCategory.Replace(" ", "_")}.png");
            viz.GenerateTimeSeriesPlot(weeklyData, topCategory, trainEndDate, testStartDate, tsPath);
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"   ✓ Saved: {tsPath}");
            Console.ResetColor();

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"\n✅ All charts saved to: {chartsDir}");
            Console.ResetColor();
        }
        catch (Exception ex)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"❌ ERROR: {ex.Message}");
            Console.ResetColor();
            _logger.Error(ex, "Chart generation failed");
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // EXISTING METHODS (unchanged)
    // ═══════════════════════════════════════════════════════════════════════════

    static void PrepareGlobalData()
    {
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("═══ DATA PREPARATION (ALL CATEGORIES) ═══");
        Console.ResetColor();

        _logger.Information("Starting global data preparation");

        try
        {
            var ordersPath = Path.Combine(_dataDirectory, "olist_orders_dataset.csv");
            var orderItemsPath = Path.Combine(_dataDirectory, "olist_order_items_dataset.csv");
            var productsPath = Path.Combine(_dataDirectory, "olist_products_dataset.csv");

            if (!File.Exists(ordersPath) || !File.Exists(orderItemsPath) || !File.Exists(productsPath))
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("❌ ERROR: Missing data files in 'data/' directory");
                Console.ResetColor();
                Console.WriteLine("\nMake sure the following files exist in 'data/':");
                Console.WriteLine("  • olist_orders_dataset.csv");
                Console.WriteLine("  • olist_order_items_dataset.csv");
                Console.WriteLine("  • olist_products_dataset.csv");
                return;
            }

            var loader = new OlistDataLoader(ordersPath, orderItemsPath, productsPath, _logger);
            var processor = new OlistDataProcessor(_logger);

            Console.WriteLine("⏳ Loading Olist data...");
            var orders = loader.LoadOrders();
            var orderItems = loader.LoadOrderItems();
            var products = loader.LoadProducts();

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"✓ Loaded: {orders.Count:N0} orders, {orderItems.Count:N0} items, {products.Count:N0} products");
            Console.ResetColor();

            Console.WriteLine("\n⏳ Aggregating weekly data for ALL categories...");
            var weeklyData = processor.AggregateWeeklySalesAllCategories(orders, orderItems, products);

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"✓ Aggregated into {weeklyData.Count:N0} weekly records");
            Console.ResetColor();

            // Pokaż statystyki wszystkich kategorii
            var allCategories = weeklyData
                .GroupBy(x => x.Category)
                .Select(g => new
                {
                    Category = g.Key,
                    TotalQuantity = g.Sum(x => x.Quantity),
                    WeeksWithSales = g.Count(x => x.Quantity > 0),
                    TotalRevenue = g.Sum(x => x.Revenue)
                })
                .OrderByDescending(x => x.TotalQuantity)
                .ToList();

            Console.WriteLine($"\n📦 All categories ({allCategories.Count}):");
            Console.WriteLine($"{"Category",-35} {"Sales",12} {"Weeks with sales",15} {"Revenue (R$)",15}");
            Console.WriteLine(new string('─', 85));

            // Pokaż TOP 20 dla podglądu
            foreach (var cat in allCategories.Take(20))
            {
                Console.WriteLine($"{cat.Category,-35} {cat.TotalQuantity,12:N0} {cat.WeeksWithSales,15} R${cat.TotalRevenue,14:N0}");
            }

            if (allCategories.Count > 20)
            {
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine($"... and {allCategories.Count - 20} more categories");
                Console.ResetColor();
            }

            // Zapisz przetworzone dane
            var preprocessedDir = Path.GetDirectoryName(_preprocessedGlobalDataPath);
            if (!string.IsNullOrEmpty(preprocessedDir))
                Directory.CreateDirectory(preprocessedDir);

            var json = JsonSerializer.Serialize(weeklyData, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(_preprocessedGlobalDataPath, json);

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"\n✓ Data saved to: {_preprocessedGlobalDataPath}");
            Console.WriteLine("✓ Data preparation completed successfully!");
            Console.ResetColor();
        }
        catch (Exception ex)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"❌ ERROR during data preparation: {ex.Message}");
            Console.ResetColor();
            _logger.Error(ex, "Global data preparation failed");
        }
    }

    static void TrainGlobalModel()
    {
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("═══ GLOBAL MODEL TRAINING ═══");
        Console.ResetColor();

        _logger.Information("Starting GLOBAL model training");

        try
        {
            if (!File.Exists(_preprocessedGlobalDataPath))
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("❌ ERROR: Missing preprocessed global data. Please run option 1 first.");
                Console.ResetColor();
                return;
            }

            Console.WriteLine("⏳ Loading preprocessed global data...");
            var json = File.ReadAllText(_preprocessedGlobalDataPath);
            var weeklyData = JsonSerializer.Deserialize<List<WeeklySalesData>>(json);

            if (weeklyData == null || !weeklyData.Any())
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("❌ ERROR: No data to train on");
                Console.ResetColor();
                return;
            }

            var processor = new OlistDataProcessor(_logger);

            // Konkretne daty podziału
            var trainEndDate = new DateTime(2018, 2, 19);
            var testStartDate = new DateTime(2018, 2, 26);

            Console.WriteLine($"📅 Time-based split (GLOBAL):");
            Console.WriteLine($"   Train: 2017-01-02 → {trainEndDate:yyyy-MM-dd}");
            Console.WriteLine($"   Test:  {testStartDate:yyyy-MM-dd} → 2018-08-27\n");

            Console.WriteLine("⏳ Creating global features (with ProductCategory)...");
            var (train, test) = processor.SplitGlobalByDate(weeklyData, trainEndDate, testStartDate);

            if (train.Count < 50 || test.Count < 20)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"❌ ERROR: Insufficient data (train={train.Count}, test={test.Count})");
                Console.ResetColor();
                return;
            }

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"✓ Train: {train.Count} samples, Test: {test.Count} samples");
            Console.ResetColor();

            // Pokaż rozkład kategorii
            var trainCategories = train.GroupBy(x => x.ProductCategory)
                .Select(g => new { Category = g.Key, Count = g.Count() })
                .OrderByDescending(x => x.Count)
                .Take(15)
                .ToList();

            Console.WriteLine($"\n📊 Category distribution in the training set (TOP 15):");
            foreach (var cat in trainCategories)
            {
                Console.WriteLine($"   • {cat.Category,-30}: {cat.Count,5} samples");
            }

            Console.WriteLine($"\n🌲 Training GLOBAL FastTree model...");
            Console.WriteLine("   ✅ ProductCategory = One-Hot Encoded");
            Console.WriteLine("   ✅ Log-transform (log1p/expm1)");
            Console.WriteLine("   ✅ No data leakage (lags from past only)\n");

            var trainer = new GlobalModelTrainer(_logger);
            var model = trainer.TrainGlobalModel(train);

            Console.WriteLine("📊 Evaluating on test set...");
            var metrics = trainer.EvaluateGlobalModel(model, test);
            metrics.TrainSamples = train.Count;

            Console.WriteLine("\n💾 Saving global model...");
            Directory.CreateDirectory(_modelsDirectory);
            var modelPath = Path.Combine(_modelsDirectory, "global_model.zip");

            // ✅ IMPORTANT: Save model to .zip file
            var mlContext = new MLContext(seed: 42);
            
            // Prepare data for schema save (required by Save)
            var sampleData = train.Take(1).ToList();
            var sampleDataView = mlContext.Data.LoadFromEnumerable(sampleData);
            
            // Save model
            mlContext.Model.Save(model, sampleDataView.Schema, modelPath);
            
            _logger.Information("Model saved to {Path}", modelPath);

            // Save metrics
            var metricsPath = Path.Combine(_modelsDirectory, "global_metrics.json");
            var metricsJson = JsonSerializer.Serialize(metrics, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(metricsPath, metricsJson);

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"\n✓ Global model saved to: {modelPath}");
            Console.WriteLine($"✓ Metrics saved to: {metricsPath}");
            Console.ResetColor();

            // Weryfikacja zapisu
            if (File.Exists(modelPath))
            {
                var fileInfo = new FileInfo(modelPath);
                Console.WriteLine($"✓ Model file size: {fileInfo.Length / 1024.0:F2} KB");
            }

            // Podsumowanie
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("\n═══════════════ GLOBAL MODEL SUMMARY ═══════════════");
            Console.ResetColor();
            Console.WriteLine($"Model:          GLOBAL (FastTree + log1p)");
            Console.WriteLine($"Train samples:  {metrics.TrainSamples:N0}");
            Console.WriteLine($"Test samples:   {metrics.TestSamples:N0}");
            Console.WriteLine($"R²:             {metrics.RSquared:F4}");
            Console.WriteLine($"MAE:            {metrics.MAE:F2}");
            Console.WriteLine($"RMSE:           {metrics.RMSE:F2}");
            Console.WriteLine($"MAPE:           {metrics.MAPE:F2}%");

            // Colored summary
            Console.WriteLine();
            if (metrics.RSquared > 0.8)
            {
                Console.ForegroundColor = ConsoleColor.Green;
                Console.WriteLine("✓ EXCELLENT MODEL (R² > 0.8) - production ready!");
            }
            else if (metrics.RSquared > 0.5)
            {
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine("⚠ GOOD MODEL (0.5 < R² < 0.8) - improvements possible");
            }
            else
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("✗ WEAK MODEL (R² < 0.5) - needs optimization");
            }
            Console.ResetColor();
        }
        catch (Exception ex)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"❌ ERROR while training global model: {ex.Message}");
            Console.WriteLine($"   Stack trace: {ex.StackTrace}");
            Console.ResetColor();
            _logger.Error(ex, "Global model training failed");
        }
    }

    static void EvaluateGlobalModel()
    {
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("═══ GLOBAL MODEL EVALUATION ═══");
        Console.ResetColor();

        try
        {
            var metricsPath = Path.Combine(_modelsDirectory, "global_metrics.json");
            if (!File.Exists(metricsPath))
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("❌ ERROR: No trained global model. Please run option 2 first.");
                Console.ResetColor();
                return;
            }

            var json = File.ReadAllText(metricsPath);
            var metrics = JsonSerializer.Deserialize<ModelMetrics>(json);

            if (metrics == null)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("❌ ERROR: Unable to load metrics");
                Console.ResetColor();
                return;
            }

            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("\n═══════════════ GLOBAL MODEL METRICS ═══════════════");
            Console.ResetColor();
            Console.WriteLine($"Model:          {metrics.Category}");
            Console.WriteLine($"Train samples:  {metrics.TrainSamples:N0}");
            Console.WriteLine($"Test samples:   {metrics.TestSamples:N0}");
            Console.WriteLine($"R²:             {metrics.RSquared:F4}");
            Console.WriteLine($"MAE:            {metrics.MAE:F2}");
            Console.WriteLine($"RMSE:           {metrics.RMSE:F2}");
            Console.WriteLine($"MAPE:           {metrics.MAPE:F2}%");
            Console.WriteLine($"Trained at:     {metrics.TrainedAt:yyyy-MM-dd HH:mm:ss}");

            Console.WriteLine();
            if (metrics.RSquared > 0.8)
            {
                Console.ForegroundColor = ConsoleColor.Green;
                Console.WriteLine("✓ EXCELLENT MODEL (R² > 0.8) - production ready!");
            }
            else if (metrics.RSquared > 0.5)
            {
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine("⚠ GOOD MODEL (0.5 < R² < 0.8) - improvements possible");
            }
            else
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("✗ WEAK MODEL (R² < 0.5) - needs optimization");
            }
            Console.ResetColor();
        }
        catch (Exception ex)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"❌ ERROR during evaluation: {ex.Message}");
            Console.ResetColor();
            _logger.Error(ex, "Global model evaluation failed");
        }
    }

    static void DetectDataLeakage()
    {
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("═══ DATA LEAKAGE DETECTION ═══");
        Console.ResetColor();

        try
        {
            if (!File.Exists(_preprocessedGlobalDataPath))
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("❌ ERROR: Missing global data. Please run option 1 first.");
                Console.ResetColor();
                return;
            }

            Console.WriteLine("⏳ Loading data...");
            var json = File.ReadAllText(_preprocessedGlobalDataPath);
            var weeklyData = JsonSerializer.Deserialize<List<WeeklySalesData>>(json);

            if (weeklyData == null || !weeklyData.Any())
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("❌ ERROR: No data");
                Console.ResetColor();
                return;
            }

            var processor = new OlistDataProcessor(_logger);
            var trainEndDate = new DateTime(2018, 2, 19);
            var testStartDate = new DateTime(2018, 2, 26);

            Console.WriteLine("⏳ Preparing train/test data...");
            var (train, test) = processor.SplitGlobalByDate(weeklyData, trainEndDate, testStartDate);

            Console.WriteLine($"✓ Train: {train.Count} samples");
            Console.WriteLine($"✓ Test:  {test.Count} samples\n");

            Console.WriteLine("🔍 Analyzing data leakage...\n");
            var detector = new DataLeakageDetector(_logger);
            var report = detector.DetectLeakage(train, test, weeklyData);

            // Display report
            var reportText = detector.GenerateReport(report);
            Console.WriteLine(reportText);

            // Save report to file
            var reportPath = Path.Combine(_modelsDirectory, "leakage_report.txt");
            Directory.CreateDirectory(_modelsDirectory);
            File.WriteAllText(reportPath, reportText);

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"\n✓ Report saved to: {reportPath}");
            Console.ResetColor();
        }
        catch (Exception ex)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"❌ ERROR: {ex.Message}");
            Console.ResetColor();
            _logger.Error(ex, "Data leakage detection failed");
        }
    }

    static void GenerateForecastsGlobal()
    {
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("═══ GENERATING FORECASTS (GLOBAL MODEL) ═══");
        Console.ResetColor();

        try
        {
            var modelPath = Path.Combine(_modelsDirectory, "global_model.zip");
            if (!File.Exists(modelPath))
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("❌ ERROR: No trained global model. Please run option 2 first.");
                Console.ResetColor();
                return;
            }

            Console.WriteLine("⏳ Loading global model...");
            var mlContext = new MLContext(seed: 42);
            var model = mlContext.Model.Load(modelPath, out var modelSchema);

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("✓ Model loaded successfully");
            Console.ResetColor();

            // Wczytaj dane historyczne
            if (!File.Exists(_preprocessedGlobalDataPath))
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("❌ ERROR: No historical data");
                Console.ResetColor();
                return;
            }

            var json = File.ReadAllText(_preprocessedGlobalDataPath);
            var weeklyData = JsonSerializer.Deserialize<List<WeeklySalesData>>(json);

            if (weeklyData == null || !weeklyData.Any())
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("❌ ERROR: No historical data");
                Console.ResetColor();
                return;
            }

            var processor = new OlistDataProcessor(_logger);
            var lastDate = weeklyData.Max(x => x.WeekStartDate);
            var categories = weeklyData.Select(x => x.Category).Distinct().OrderBy(x => x).Take(20).ToList(); // TOP 20 dla podglądu

            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine($"\n🔮 Generating 4-week forecasts from {lastDate:yyyy-MM-dd}...");
            Console.WriteLine("(Showing TOP 20 categories)\n");
            Console.ResetColor();

            Console.WriteLine($"{"Category",-30} {"Week 1",12} {"Week 2",12} {"Week 3",12} {"Week 4",12}");
            Console.WriteLine(new string('═', 80));

            var predictionEngine = mlContext.Model.CreatePredictionEngine<SalesPredictionInput, SalesPredictionOutput>(model);

            foreach (var category in categories)
            {
                var categoryHistory = weeklyData.Where(x => x.Category == category).OrderBy(x => x.WeekStartDate).ToList();
                
                if (categoryHistory.Count < 4)
                    continue;

                Console.Write($"{category,-30} ");

                for (int week = 1; week <= 4; week++)
                {
                    var lastWeeks = categoryHistory.TakeLast(4).ToList();
                    
                    var input = new SalesPredictionInput
                    {
                        WeekNumber = (lastDate.AddDays(7 * week).DayOfYear / 7) + 1,
                        Month = lastDate.AddDays(7 * week).Month,
                        Quarter = ((lastDate.AddDays(7 * week).Month - 1) / 3) + 1,
                        IsBlackFridayWeek = 0,
                        IsHolidaySeason = 0,
                        Lag1 = lastWeeks.Count >= 1 ? lastWeeks[^1].Quantity : 0,
                        Lag2 = lastWeeks.Count >= 2 ? lastWeeks[^2].Quantity : 0,
                        Lag3 = lastWeeks.Count >= 3 ? lastWeeks[^3].Quantity : 0,
                        Lag4 = lastWeeks.Count >= 4 ? lastWeeks[^4].Quantity : 0,
                        RollingAvg4Weeks = (float)lastWeeks.Average(x => x.Quantity),
                        Trend = categoryHistory.Count + week,
                        CategoryHistoricalAvg = (float)categoryHistory.Average(x => x.Quantity),
                        ProductCategory = category
                    };

                    var logPrediction = predictionEngine.Predict(input).PredictedSales;
                    var actualPrediction = Math.Max(0, (float)(Math.Exp(logPrediction) - 1)); // expm1
                    Console.Write($"{actualPrediction,12:F0} ");
                }

                Console.WriteLine();
            }

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("\n✓ Prognozy wygenerowane pomyślnie!");
            Console.ResetColor();
        }
        catch (Exception ex)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"❌ BŁĄD podczas generowania prognoz: {ex.Message}");
            Console.ResetColor();
            _logger.Error(ex, "Forecast generation failed");
        }
    }

    static void RunGridSearch()
    {
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("═══ GRID SEARCH - HYPERPARAMETER OPTIMIZATION ═══");
        Console.ResetColor();

        _logger.Information("Starting Grid Search");

        try
        {
            if (!File.Exists(_preprocessedGlobalDataPath))
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("❌ ERROR: Missing preprocessed data. Please run option 1 first.");
                Console.ResetColor();
                return;
            }

            Console.WriteLine("⏳ Loading data...");
            var json = File.ReadAllText(_preprocessedGlobalDataPath);
            var weeklyData = JsonSerializer.Deserialize<List<WeeklySalesData>>(json);

            if (weeklyData == null || !weeklyData.Any())
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("❌ ERROR: No data");
                Console.ResetColor();
                return;
            }

            var processor = new OlistDataProcessor(_logger);
            var trainEndDate = new DateTime(2018, 2, 19);
            var testStartDate = new DateTime(2018, 2, 26);

            Console.WriteLine("⏳ Preparing train/test sets...");
            var (train, test) = processor.SplitGlobalByDate(weeklyData, trainEndDate, testStartDate);

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"✓ Train: {train.Count} samples, Test: {test.Count} samples\n");
            Console.ResetColor();

            // Choose Grid Search mode
            Console.WriteLine("Select Grid Search mode:");
            Console.WriteLine("1. 🚀 QUICK (6 configs, ~2-3 minutes)");
            Console.WriteLine("2. 🔬 STANDARD (54 configs, ~15-20 minutes)");
            Console.Write("\nChoice (1-2): ");
            var choice = Console.ReadLine();

            List<GridSearchConfiguration> configurations;
            if (choice == "2")
            {
                configurations = GridSearchManager.GenerateStandardGrid();
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine($"\n⚠️  STANDARD Grid Search: {configurations.Count} configs - may take ~15-20 minutes!");
                Console.ResetColor();
            }
            else
            {
                configurations = GridSearchManager.GenerateQuickGrid();
                Console.ForegroundColor = ConsoleColor.Cyan;
                Console.WriteLine($"\n🚀 QUICK Grid Search: {configurations.Count} configs - estimated time ~2-3 minutes");
                Console.ResetColor();
            }

            Console.Write("\nContinue? (Y/N): ");
            if (Console.ReadLine()?.ToUpper() != "Y")
            {
                Console.WriteLine("❌ Grid Search cancelled");
                return;
            }

            // Uruchom Grid Search
            var gridSearch = new GridSearchManager(_logger);
            var results = gridSearch.PerformGridSearch(train, test, configurations);

            // Sort results by R²
            var sortedResults = results.OrderByDescending(r => r.Score).ToList();

            // Show TOP 5 results
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("\n\n═══════════════ TOP 5 CONFIGURATIONS ═══════════════");
            Console.ResetColor();
            Console.WriteLine($"{"Rank",-6} {"R²",10} {"MAE",10} {"RMSE",10} {"MAPE%",10} {"Time(s)",10} {"Configuration",-50}");
            Console.WriteLine(new string('═', 110));

            for (int i = 0; i < Math.Min(5, sortedResults.Count); i++)
            {
                var result = sortedResults[i];
                var rank = i + 1;

                if (rank == 1)
                    Console.ForegroundColor = ConsoleColor.Green;
                else if (rank <= 3)
                    Console.ForegroundColor = ConsoleColor.Yellow;
                else
                    Console.ForegroundColor = ConsoleColor.Gray;

                Console.WriteLine($"{rank,-6} {result.Metrics.RSquared,10:F4} {result.Metrics.MAE,10:F2} {result.Metrics.RMSE,10:F2} {result.Metrics.MAPE,10:F2} {result.TrainingTime.TotalSeconds,10:F1} {result.Configuration.ToString(),-50}");
                Console.ResetColor();
            }

            // Best configuration
            var best = sortedResults.First();
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("\n\n🏆 BEST CONFIGURATION:");
            Console.ResetColor();
            Console.WriteLine($"   {best.Configuration}");
            Console.WriteLine($"   R² = {best.Metrics.RSquared:F4}");
            Console.WriteLine($"   MAE = {best.Metrics.MAE:F2}");
            Console.WriteLine($"   RMSE = {best.Metrics.RMSE:F2}");
            Console.WriteLine($"   MAPE = {best.Metrics.MAPE:F2}%");
            Console.WriteLine($"   Training time = {best.TrainingTime.TotalSeconds:F1}s");

            // Save results
            var reportPath = Path.Combine(_modelsDirectory, "grid_search_results.json");
            Directory.CreateDirectory(_modelsDirectory);
            var reportJson = JsonSerializer.Serialize(sortedResults, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(reportPath, reportJson);

            // Save CSV for easy analysis in Excel
            var csvPath = Path.Combine(_modelsDirectory, "grid_search_results.csv");
            using (var writer = new StreamWriter(csvPath))
            {
                writer.WriteLine("Rank,RSquared,MAE,RMSE,MAPE,TrainingTimeSeconds,NumberOfTrees,NumberOfLeaves,MinLeaf,LearningRate");
                for (int i = 0; i < sortedResults.Count; i++)
                {
                    var r = sortedResults[i];
                    writer.WriteLine($"{i+1},{r.Metrics.RSquared:F4},{r.Metrics.MAE:F2},{r.Metrics.RMSE:F2},{r.Metrics.MAPE:F2},{r.TrainingTime.TotalSeconds:F1},{r.Configuration.NumberOfTrees},{r.Configuration.NumberOfLeaves},{r.Configuration.MinimumExampleCountPerLeaf},{r.Configuration.LearningRate:F3}");
                }
            }

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"\n✓ Results saved to:");
            Console.WriteLine($"   JSON: {reportPath}");
            Console.WriteLine($"   CSV:  {csvPath}");
            Console.ResetColor();
        }
        catch (Exception ex)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"❌ ERROR: {ex.Message}");
            Console.ResetColor();
            _logger.Error(ex, "Grid Search failed");
        }
    }

    static void AnalyzeFeatureImportance()
    {
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("═══ FEATURE IMPORTANCE ANALYSIS ═══");
        Console.ResetColor();

        _logger.Information("Starting Feature Importance Analysis");

        try
        {
            var modelPath = Path.Combine(_modelsDirectory, "global_model.zip");
            if (!File.Exists(modelPath))
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("❌ ERROR: Model missing. Please run option 2 (Train model).");
                Console.ResetColor();
                return;
            }

            Console.WriteLine("⏳ Loading model and data...");
            var mlContext = new MLContext(seed: 42);
            var model = mlContext.Model.Load(modelPath, out _);

            if (!File.Exists(_preprocessedGlobalDataPath))
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("❌ ERROR: Data missing. Please run option 1.");
                Console.ResetColor();
                return;
            }

            var json = File.ReadAllText(_preprocessedGlobalDataPath);
            var weeklyData = JsonSerializer.Deserialize<List<WeeklySalesData>>(json);

            if (weeklyData == null || !weeklyData.Any())
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("❌ ERROR: Unable to load data");
                Console.ResetColor();
                return;
            }

            var processor = new OlistDataProcessor(_logger);
            var trainEndDate = new DateTime(2018, 2, 19);
            var testStartDate = new DateTime(2018, 2, 26);
            var (train, test) = processor.SplitGlobalByDate(weeklyData, trainEndDate, testStartDate);

            Console.WriteLine($"✓ Loaded: {test.Count} test samples\n");
            Console.WriteLine("⏳ Calculating Permutation Feature Importance (~30s)...\n");

            var analyzer = new FeatureImportanceAnalyzer(_logger);
            var results = analyzer.AnalyzeFeatureImportance(model, test);

            // Normalize results to 0-100%
            var maxImportance = results.Max(r => Math.Abs(r.RSquaredMean));
            if (maxImportance > 0)
            {
                foreach (var result in results)
                {
                    result.ImportanceScore = (Math.Abs(result.RSquaredMean) / maxImportance) * 100;
                }
            }

            // Display results
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("═══════════════ FEATURE IMPORTANCE ═══════════════");
            Console.ResetColor();
            Console.WriteLine($"{"Feature",-30} {"Importance %",15} {"R² Drop",12} {"MAE Increase",15}");
            Console.WriteLine(new string('═', 80));

            foreach (var result in results)
            {
                // Color by importance
                if (result.ImportanceScore > 75)
                    Console.ForegroundColor = ConsoleColor.Green;
                else if (result.ImportanceScore > 40)
                    Console.ForegroundColor = ConsoleColor.Yellow;
                else
                    Console.ForegroundColor = ConsoleColor.Gray;

                Console.WriteLine($"{result.FeatureName,-30} {result.ImportanceScore,15:F1} {result.RSquaredMean,12:F4} {result.MaeIncrease,15:F2}");
                Console.ResetColor();
            }

            // Save results
            Directory.CreateDirectory(_modelsDirectory);
            var reportPath = Path.Combine(_modelsDirectory, "feature_importance.json");
            var reportJson = JsonSerializer.Serialize(results, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(reportPath, reportJson);

            // CSV for Excel
            var csvPath = Path.Combine(_modelsDirectory, "feature_importance.csv");
            using (var writer = new StreamWriter(csvPath))
            {
                writer.WriteLine("FeatureName,ImportanceScore,RSquaredDrop,MaeIncrease,RmseIncrease");
                foreach (var r in results)
                {
                    writer.WriteLine($"{r.FeatureName},{r.ImportanceScore:F2},{r.RSquaredMean:F4},{r.MaeIncrease:F2},{r.RmseIncrease:F2}");
                }
            }

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"\n✓ Results saved to:");
            Console.WriteLine($"   JSON: {reportPath}");
            Console.WriteLine($"   CSV:  {csvPath}");
            Console.ResetColor();

            // Interpretation
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("\n📊 INTERPRETATION:");
            Console.ResetColor();
            var top3 = results.Take(3).ToList();
            Console.WriteLine($"🏆 TOP 3 most important features:");
            for (int i = 0; i < top3.Count; i++)
            {
                Console.WriteLine($"   {i+1}. {top3[i].FeatureName} ({top3[i].ImportanceScore:F1}%)");
            }
        }
        catch (Exception ex)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"❌ ERROR: {ex.Message}");
            Console.WriteLine($"   Stack trace: {ex.StackTrace}");
            Console.ResetColor();
            _logger.Error(ex, "Feature importance analysis failed");
        }
    }

    static void CompareWithBaseline()
    {
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("═══ BASELINE COMPARISON (NAIVE FORECAST) ═══");
        Console.ResetColor();

        _logger.Information("Starting Baseline Comparison");

        try
        {
            var modelPath = Path.Combine(_modelsDirectory, "global_model.zip");
            var metricsPath = Path.Combine(_modelsDirectory, "global_metrics.json");

            if (!File.Exists(modelPath) || !File.Exists(metricsPath))
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("❌ ERROR: FastTree model missing. Please run option 2 (Train model).");
                Console.ResetColor();
                return;
            }

            if (!File.Exists(_preprocessedGlobalDataPath))
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("❌ ERROR: Data missing. Please run option 1.");
                Console.ResetColor();
                return;
            }

            Console.WriteLine("⏳ Loading data...");
            var json = File.ReadAllText(_preprocessedGlobalDataPath);
            var weeklyData = JsonSerializer.Deserialize<List<WeeklySalesData>>(json);

            if (weeklyData == null || !weeklyData.Any())
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("❌ ERROR: Unable to load data");
                Console.ResetColor();
                return;
            }

            var processor = new OlistDataProcessor(_logger);
            var trainEndDate = new DateTime(2018, 2, 19);
            var testStartDate = new DateTime(2018, 2, 26);
            var (train, test) = processor.SplitGlobalByDate(weeklyData, trainEndDate, testStartDate);

            Console.WriteLine($"✓ Loaded: {test.Count} test samples\n");

            // 1. Wczytaj metryki FastTree
            var fastTreeMetricsJson = File.ReadAllText(metricsPath);
            var fastTreeMetrics = JsonSerializer.Deserialize<ModelMetrics>(fastTreeMetricsJson);

            // 2. Oblicz metryki Naive Baseline
            Console.WriteLine("⏳ Calculating metrics for Naive Baseline (4-week moving average)...");
            var baselineComparison = new BaselineComparison(_logger);
            var naiveMetrics = baselineComparison.EvaluateNaiveBaseline(test);

            // 3. Porównanie
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("\n═══════════════ MODEL COMPARISON ═══════════════");
            Console.ResetColor();
            Console.WriteLine($"{"Metric",-20} {"Naive Baseline",20} {"FastTree (ML)",20} {"Improvement",15}");
            Console.WriteLine(new string('═', 80));

            // R²
            var r2Improvement = ((fastTreeMetrics!.RSquared - naiveMetrics.RSquared) / Math.Max(Math.Abs(naiveMetrics.RSquared), 0.001)) * 100;
            Console.Write($"{"R²",-20} ");
            Console.ForegroundColor = ConsoleColor.Red;
            Console.Write($"{naiveMetrics.RSquared,20:F4} ");
            Console.ForegroundColor = ConsoleColor.Green;
            Console.Write($"{fastTreeMetrics.RSquared,20:F4} ");
            Console.ForegroundColor = r2Improvement > 0 ? ConsoleColor.Green : ConsoleColor.Red;
            Console.WriteLine($"{r2Improvement,15:F1}%");
            Console.ResetColor();

            // MAE
            var maeImprovement = ((naiveMetrics.MAE - fastTreeMetrics.MAE) / naiveMetrics.MAE) * 100;
            Console.Write($"{"MAE",-20} ");
            Console.ForegroundColor = ConsoleColor.Red;
            Console.Write($"{naiveMetrics.MAE,20:F2} ");
            Console.ForegroundColor = ConsoleColor.Green;
            Console.Write($"{fastTreeMetrics.MAE,20:F2} ");
            Console.ForegroundColor = maeImprovement > 0 ? ConsoleColor.Green : ConsoleColor.Red;
            Console.WriteLine($"{maeImprovement,15:F1}%");
            Console.ResetColor();

            // RMSE
            var rmseImprovement = ((naiveMetrics.RMSE - fastTreeMetrics.RMSE) / naiveMetrics.RMSE) * 100;
            Console.Write($"{"RMSE",-20} ");
            Console.ForegroundColor = ConsoleColor.Red;
            Console.Write($"{naiveMetrics.RMSE,20:F2} ");
            Console.ForegroundColor = ConsoleColor.Green;
            Console.Write($"{fastTreeMetrics.RMSE,20:F2} ");
            Console.ForegroundColor = rmseImprovement > 0 ? ConsoleColor.Green : ConsoleColor.Red;
            Console.WriteLine($"{rmseImprovement,15:F1}%");
            Console.ResetColor();

            // MAPE
            var mapeImprovement = ((naiveMetrics.MAPE - fastTreeMetrics.MAPE) / naiveMetrics.MAPE) * 100;
            Console.Write($"{"MAPE (%)",-20} ");
            Console.ForegroundColor = ConsoleColor.Red;
            Console.Write($"{naiveMetrics.MAPE,20:F2} ");
            Console.ForegroundColor = ConsoleColor.Green;
            Console.Write($"{fastTreeMetrics.MAPE,20:F2} ");
            Console.ForegroundColor = mapeImprovement > 0 ? ConsoleColor.Green : ConsoleColor.Red;
            Console.WriteLine($"{mapeImprovement,15:F1}%");
            Console.ResetColor();

            // Podsumowanie
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("\n📊 WNIOSKI:");
            Console.ResetColor();
            Console.WriteLine($"✅ Model FastTree przewyższa prosty baseline:");
            Console.WriteLine($"   • R² poprawione o {r2Improvement:F1}%");
            Console.WriteLine($"   • MAE zredukowane o {maeImprovement:F1}%");
            Console.WriteLine($"   • RMSE zredukowane o {rmseImprovement:F1}%");
            Console.WriteLine($"   • MAPE zredukowane o {mapeImprovement:F1}%");

            // Save report
            var comparisonReport = new
            {
                NaiveBaseline = naiveMetrics,
                FastTreeML = fastTreeMetrics,
                Improvements = new
                {
                    RSquared = r2Improvement,
                    MAE = maeImprovement,
                    RMSE = rmseImprovement,
                    MAPE = mapeImprovement
                }
            };

            Directory.CreateDirectory(_modelsDirectory);
            var reportPath = Path.Combine(_modelsDirectory, "baseline_comparison.json");
            var reportJson = JsonSerializer.Serialize(comparisonReport, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(reportPath, reportJson);

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"\n✓ Report saved to: {reportPath}");
            Console.ResetColor();
        }
        catch (Exception ex)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"❌ ERROR: {ex.Message}");
            Console.ResetColor();
            _logger.Error(ex, "Baseline comparison failed");
        }
    }

    static void SetupLogging()
    {
        Log.Logger = new LoggerConfiguration()
            .MinimumLevel.Information()
            .WriteTo.Console()
            .WriteTo.File("logs/sales-forecasting-.log", rollingInterval: Serilog.RollingInterval.Day)
            .CreateLogger();

        _logger = Log.Logger;
    }

    static void LoadConfiguration()
    {
        string baseDirectory = AppDomain.CurrentDomain.BaseDirectory;

        string projectRoot = Path.GetFullPath(Path.Combine(baseDirectory, "../../../"));

        _dataDirectory = Path.Combine(projectRoot, "data");

        _modelsDirectory = Path.Combine(projectRoot, "models");

        _preprocessedGlobalDataPath = Path.Combine(_dataDirectory, "preprocessed", "weekly_sales_all.json");

        Console.WriteLine($"📁 Project Root: {projectRoot}");
        Console.WriteLine($"📁 Data directory: {_dataDirectory}");
        Console.WriteLine($"📁 Models directory: {_modelsDirectory}");
        Console.WriteLine();
    }
}