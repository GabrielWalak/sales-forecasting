namespace SalesForecasting.Core.Models;

public class LeakageReport
{
    public int TrainSamples { get; set; }
    public int TestSamples { get; set; }
    public DateTime AnalysisDate { get; set; }
    
    public Dictionary<string, double> FeatureCorrelations { get; set; } = new();
    public List<string> SuspiciousFeatures { get; set; } = new();
    public LagValidationResult LagValidation { get; set; } = new();
    public TemporalValidationResult TemporalValidation { get; set; } = new();
    public NormalizationCheckResult NormalizationCheck { get; set; } = new();
    public DataOverlapResult DataOverlap { get; set; } = new();
}

public class LagValidationResult
{
    public double Lag1Correlation { get; set; }
    public double Lag2Correlation { get; set; }
    public double Lag3Correlation { get; set; }
    public double Lag4Correlation { get; set; }
    public bool IsValid { get; set; }
    public List<string> Issues { get; set; } = new();
}

public class TemporalValidationResult
{
    public int MaxTrainTrend { get; set; }
    public int MinTestTrend { get; set; }
    public bool IsChronological { get; set; }
    public List<string> Issues { get; set; } = new();
}

public class NormalizationCheckResult
{
    public List<FeatureRangeCheck> Checks { get; set; } = new();
    public bool IsSafe { get; set; }
}

public class FeatureRangeCheck
{
    public string FeatureName { get; set; } = string.Empty;
    public double TrainMin { get; set; }
    public double TrainMax { get; set; }
    public double TestMin { get; set; }
    public double TestMax { get; set; }
    public bool IsWithinRange { get; set; }
}

public class DataOverlapResult
{
    public int OverlappingRecords { get; set; }
    public bool HasOverlap { get; set; }
}