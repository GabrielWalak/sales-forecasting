namespace SalesForecasting.Core.Models;

public class CrossValidationMetrics
{
    public string Category { get; set; } = string.Empty;
    public int NFolds { get; set; }
    
    // Œrednie z wszystkich foldów
    public double MeanRSquared { get; set; }
    public double StdRSquared { get; set; }
    
    public double MeanMAE { get; set; }
    public double StdMAE { get; set; }
    
    public double MeanRMSE { get; set; }
    public double StdRMSE { get; set; }
    
    public double MeanMAPE { get; set; }
    public double StdMAPE { get; set; }
    
    // Metryki z ka¿dego folda
    public List<ModelMetrics> FoldMetrics { get; set; } = new();
}