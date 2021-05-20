using System.Collections.Generic;
using Svm.Binary;

namespace Svm {
  /// <summary>
  /// Classifies an instance of <typeparamref name="TValue"/> to a label provided in the training observations.
  /// </summary>
  /// <typeparam name="TValue">The type to classify.</typeparam>
  /// <typeparam name="TLabel">The type of the label.</typeparam>
  public class MultiClassSvm<TValue, TLabel> {
    private readonly BinarySvm<TValue>[,] _classifiers;
    private readonly Dictionary<int, TLabel> _idToLabelMap;

    internal MultiClassSvm( BinarySvm<TValue>[,] classifiers, Dictionary<int, TLabel> idToLabelMap ) {
      _classifiers = classifiers;
      _idToLabelMap = idToLabelMap;
    }

    /// <summary>
    /// Classifies an instance of <typeparamref name="TValue"/> to a label provided in the training observations.
    /// </summary>
    /// <param name="value">The value to classify.</param>
    /// <returns>The resulting label.</returns>
    public TLabel Classify( TValue value ) {
      int first = 0;
      int last = _classifiers.GetLength( 0 ) - 1;
      int classId = 0;

      while ( first != last ) {
        int result = _classifiers[first, last].Classify( value );
        if ( result == -1 ) {
          classId = first;
          last--;
        }
        else {
          classId = last;
          first++;
        }
      }

      return _idToLabelMap[classId];
    }
  }
}
