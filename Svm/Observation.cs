namespace Svm {
  /// <summary>
  /// A labeled observation for training using <see cref="MultiClassSvmTrainer"/>.
  /// </summary>
  public class Observation<TValue, TLabel> {
    
    public Observation( TValue value, TLabel label ) {
      Value = value;
      Label = label;
    }

    /// <summary>
    /// The features to be processed in the kernel.
    /// </summary>
    public TValue Value { get; }
    
    /// <summary>
    /// The label identifying the observation class.
    /// </summary>
    public TLabel Label { get; }
  }
}
