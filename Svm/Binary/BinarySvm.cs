using System.Collections.Generic;
using System.Linq;

namespace Svm.Binary {
  /// <summary>
  /// This classifier can classify a value in one of two classes; -1 and 1.
  /// </summary>
  public class BinarySvm<TValue> {
    private readonly double _b;
    private readonly IKernel<TValue> _kernel;

    internal BinarySvm(
      double b,
      IReadOnlyList<SupportVector<TValue>> supportVectors,
      IKernel<TValue> kernel ) {
      _b = b;
      SupportVectors = supportVectors;
      _kernel = kernel;
    }
    
    /// <summary>
    /// The support vectors that define the classifier's hyperplane.
    /// Mainly for debugging purposes.
    /// </summary>
    public IReadOnlyList<SupportVector<TValue>> SupportVectors { get; }

    /// <summary>
    /// Classifies the value as belonging to either class -1 or class 1.
    /// </summary>
    public int Classify( TValue value ) {
      var sum = SupportVectors.Sum(
        s => s.Alpha * s.Observation.Label * _kernel.Compute( s.Observation.Value, value ) );
      sum -= _b;
      return sum < 0 ? -1 : 1;
    }
    
    /// <summary>
    /// Trains a Binary SVM based on the specified observations and kernel.
    /// </summary>
    /// <param name="targets">The observations.</param>
    /// <param name="k">The kernel.</param>
    /// <typeparam name="TValue">The samples that the resulting classifier can handle.</typeparam>
    /// <returns>A binary classifier that can distinguish two classes.</returns>
    public static BinarySvm<TValue> Train( BinaryObservation<TValue>[] targets, IKernel<TValue> k ) {
      return new SequentialMinimalOptimization<TValue>( targets, k ).Optimize();
    }
  }
}
