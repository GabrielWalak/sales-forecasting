namespace SalesForecasting.Core.Models;

public class OlistProduct
{
    public string ProductId { get; set; } = string.Empty;
    public string ProductCategoryName { get; set; } = string.Empty;
    public decimal? ProductWeightG { get; set; }
    public decimal? ProductLengthCm { get; set; }
}