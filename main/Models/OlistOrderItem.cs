namespace SalesForecasting.Core.Models;

public class OlistOrderItem
{
    public string OrderId { get; set; } = string.Empty;
    public int OrderItemId { get; set; }
    public string ProductId { get; set; } = string.Empty;
    public string SellerId { get; set; } = string.Empty;
    public decimal Price { get; set; }
    public decimal FreightValue { get; set; }
}