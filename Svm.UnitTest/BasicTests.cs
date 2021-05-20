using System.Linq;
using NUnit.Framework;
using Svm.Binary;

namespace Svm.UnitTest {
  public class BasicTests {
    
    [Test]
    public void MultiClassTest() {
      var observations = new[] {
        new Observation<int[], string>( new[] { 1, 2 }, "a" ),
        new Observation<int[], string>( new[] { 1, 3 }, "a" ),
        new Observation<int[], string>( new[] { 1, 4 }, "a" ),
        new Observation<int[], string>( new[] { 5, 2 }, "b" ),
        new Observation<int[], string>( new[] { 5, 3 }, "b" ),
        new Observation<int[], string>( new[] { 5, 4 }, "b" ),
        new Observation<int[], string>( new[] { 5, 9 }, "c" ),
        new Observation<int[], string>( new[] { 6, 9 }, "c" ),
        new Observation<int[], string>( new[] { 7, 9 }, "c" )
      };
      
      var svm = MultiClassSvmTrainer.Train( observations, new DotProduct() );
      Assert.That( svm.Classify( new[] { 1, 2 } ), Is.EqualTo( "a" ) );
      Assert.That( svm.Classify( new[] { 7, 2 } ), Is.EqualTo( "b" ) );
      Assert.That( svm.Classify( new[] { 1, 9 } ), Is.EqualTo( "c" ) );
    }
    
    [Test]
    public void BinarySvmTest() {
      var svm = BinarySvmTrainer.Train(
        new[] {
          new BinaryObservation<int[]>( new[] { 0, 1 }, -1 ), new BinaryObservation<int[]>( new[] { 1, 1 }, -1 ),
          new BinaryObservation<int[]>( new[] { 4, 3 }, -1 ), new BinaryObservation<int[]>( new[] { 3, 1 }, 1 ),
          new BinaryObservation<int[]>( new[] { 4, 2 }, 1 )
        },
        new DotProduct()
      );

      Assert.That( svm.Classify( new[] { 0, 0 } ), Is.EqualTo( -1 ) );
    }

    private class DotProduct : IKernel<int[]> {
      public double Compute( int[] a, int[] b ) {
        return a.Zip( b, ( i, j ) => i * j ).Sum();
      }
    }
  }
}
