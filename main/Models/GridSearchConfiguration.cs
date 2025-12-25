namespace SalesForecasting.Core.Models;

/// <summary>
/// Konfiguracja pojedynczego zestawu hiperparametrów FastTree
/// </summary>
public class GridSearchConfiguration
{
    public int NumberOfTrees { get; set; }
    public int NumberOfLeaves { get; set; }
    public int MinimumExampleCountPerLeaf { get; set; }
    public double LearningRate { get; set; }

    public override string ToString()
    {
        return $"Trees={NumberOfTrees}, Leaves={NumberOfLeaves}, MinLeaf={MinimumExampleCountPerLeaf}, LR={LearningRate:F3}";
    }
}

/// <summary>
/// Wynik Grid Search dla konkretnej konfiguracji
/// </summary>
public class GridSearchResult
{
    public GridSearchConfiguration Configuration { get; set; } = new();
    public ModelMetrics Metrics { get; set; } = new();
    public TimeSpan TrainingTime { get; set; }
    
    public double Score => Metrics.RSquared; // G³ówna metryka do optymalizacji
}