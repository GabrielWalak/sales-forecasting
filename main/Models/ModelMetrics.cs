namespace SalesForecasting.Core.Models;

public class ModelMetrics
{
    public string Category { get; set; } = string.Empty;
    public double RSquared { get; set; }
    public double MAE { get; set; }
    public double RMSE { get; set; }
    public double MAPE { get; set; }
    public int TrainSamples { get; set; }
    public int TestSamples { get; set; }
    public DateTime TrainedAt { get; set; }
}