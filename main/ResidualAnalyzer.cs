/// <summary>
/// Analiza residuals (b³êdów predykcji)
/// </summary>
public class ResidualAnalyzer
{
    public ResidualAnalysisReport AnalyzeResiduals(
        List<double> actual,
        List<double> predicted)
    {
        var residuals = actual.Zip(predicted, (a, p) => a - p).ToList();

        return new ResidualAnalysisReport
        {
            MeanResidual = residuals.Average(),
            StdDevResidual = Math.Sqrt(residuals.Average(r => Math.Pow(r - residuals.Average(), 2))),
            MedianResidual = residuals.OrderBy(r => r).ElementAt(residuals.Count / 2),
            MaxOverprediction = residuals.Min(), // Najwiêksza nadpredykcja (negatywna)
            MaxUnderprediction = residuals.Max()  // Najwiêksza niedopredykcja (pozytywna)
        };
    }
}

public class ResidualAnalysisReport
{
    public double MeanResidual { get; set; }
    public double StdDevResidual { get; set; }
    public double MedianResidual { get; set; }
    public double MaxOverprediction { get; set; }
    public double MaxUnderprediction { get; set; }
}