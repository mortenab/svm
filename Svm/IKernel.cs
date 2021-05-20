namespace Svm {
  public interface IKernel<in TValue> {
    double Compute( TValue a, TValue b );
  }
}
