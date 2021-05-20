using System;

namespace Svm.Binary {
  /// <summary>
  /// A training observation used in <see cref="BinarySvmTrainer"/>
  /// </summary>
  public class BinaryObservation<TValue> : Observation<TValue, int> {
    public BinaryObservation( TValue value, int label ) : base( value, label ) {
      if ( label != 1 && label != -1 ) {
        throw new ArgumentOutOfRangeException( nameof(label), "Label must be 1 or -1" );
      }
    }
  }
}
