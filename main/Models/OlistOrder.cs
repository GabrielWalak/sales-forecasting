namespace SalesForecasting.Core.Models;

public class OlistOrder
{
    public string OrderId { get; set; } = string.Empty;
    public string CustomerId { get; set; } = string.Empty;
    public DateTime OrderPurchaseTimestamp { get; set; }
    public DateTime? OrderApprovedAt { get; set; }
    public DateTime? OrderDeliveredCustomerDate { get; set; }
    public string OrderStatus { get; set; } = string.Empty;
}