using CsvHelper;
using CsvHelper.Configuration;
using System.Globalization;
using SalesForecasting.Core.Models;
using Serilog;

namespace SalesForecasting.Data;

public class OlistDataLoader
{
    private readonly string _ordersPath;
    private readonly string _orderItemsPath;
    private readonly string _productsPath;
    private readonly ILogger _logger;

    public OlistDataLoader(string ordersPath, string orderItemsPath, string productsPath, ILogger logger)
    {
        _ordersPath = ordersPath;
        _orderItemsPath = orderItemsPath;
        _productsPath = productsPath;
        _logger = logger;
    }

    public List<OlistOrder> LoadOrders()
    {
        _logger.Information("Loading orders from {Path}", _ordersPath);
        
        using var reader = new StreamReader(_ordersPath);
        using var csv = new CsvReader(reader, new CsvConfiguration(CultureInfo.InvariantCulture)
        {
            HeaderValidated = null,
            MissingFieldFound = null,
            PrepareHeaderForMatch = args => args.Header.ToLower().Replace("_", "")
        });
        
        var records = csv.GetRecords<OlistOrder>().ToList();
        _logger.Information("Loaded {Count} orders", records.Count);
        return records;
    }

    public List<OlistOrderItem> LoadOrderItems()
    {
        _logger.Information("Loading order items from {Path}", _orderItemsPath);
        
        using var reader = new StreamReader(_orderItemsPath);
        using var csv = new CsvReader(reader, new CsvConfiguration(CultureInfo.InvariantCulture)
        {
            HeaderValidated = null,
            MissingFieldFound = null,
            PrepareHeaderForMatch = args => args.Header.ToLower().Replace("_", "")
        });
        
        var records = csv.GetRecords<OlistOrderItem>().ToList();
        _logger.Information("Loaded {Count} order items", records.Count);
        return records;
    }

    public List<OlistProduct> LoadProducts()
    {
        _logger.Information("Loading products from {Path}", _productsPath);
        
        using var reader = new StreamReader(_productsPath);
        using var csv = new CsvReader(reader, new CsvConfiguration(CultureInfo.InvariantCulture)
        {
            HeaderValidated = null,
            MissingFieldFound = null,
            PrepareHeaderForMatch = args => args.Header.ToLower().Replace("_", "")
        });
        
        var records = csv.GetRecords<OlistProduct>().ToList();
        _logger.Information("Loaded {Count} products", records.Count);
        return records;
    }
}