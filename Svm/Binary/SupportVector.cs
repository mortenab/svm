namespace Svm.Binary {
  /// <summary>
  /// A support vector that is used as part of classification.
  /// </summary>
  public class SupportVector<TValue> {
    public SupportVector( double alpha, BinaryObservation<TValue> observation ) {
      Alpha = alpha;
      Observation = observation;
    }

    public double Alpha { get; }
    public BinaryObservation<TValue> Observation { get; }
  }
}
