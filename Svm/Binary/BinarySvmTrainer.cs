namespace Svm.Binary {
  /// <summary>
  /// Trains <see cref="BinarySvm{TValue}"/>s.
  /// </summary>
  public static class BinarySvmTrainer {
    /// <summary>
    /// Trains a Binary SVM based on the specified observations and kernel.
    /// </summary>
    /// <param name="targets">The observations.</param>
    /// <param name="k">The kernel.</param>
    /// <typeparam name="TValue">The samples that the resulting classifier can handle.</typeparam>
    /// <returns>A binary classifier that can distinguish two classes.</returns>
    public static BinarySvm<TValue> Train<TValue>( BinaryObservation<TValue>[] targets, IKernel<TValue> k ) {
      return new SequentialMinimalOptimization<TValue>( targets, k ).Optimize();
    }
  }
}
