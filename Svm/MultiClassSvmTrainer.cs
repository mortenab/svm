using System.Collections.Generic;
using System.Linq;
using Svm.Binary;

namespace Svm {
  /// <summary>
  /// Trains a number of binary SVMs to build a multi class classifier.
  /// </summary>
  public static class MultiClassSvmTrainer {
    public static MultiClassSvm<TValue, TLabel> Train<TValue, TLabel>(
      IEnumerable<Observation<TValue, TLabel>> observations,
      IKernel<TValue> k,
      IEqualityComparer<TLabel> equalityComparer = null ) {
      var classToObservations =
        new Dictionary<TLabel, List<TValue>>( equalityComparer ?? EqualityComparer<TLabel>.Default );
      
      var idToLabelMap = new Dictionary<int, TLabel>();
      var labelId = 0;

      foreach ( var labelGroup in observations.GroupBy( t => t.Label ) ) {
        idToLabelMap.Add( labelId, labelGroup.Key );
        classToObservations.Add( labelGroup.Key, labelGroup.Select( o => o.Value ).ToList() );
        labelId++;
      }

      var classifiers = new BinarySvm<TValue>[classToObservations.Count, classToObservations.Count];

      for ( int i = 0; i < classToObservations.Count; i++ ) {
        for ( int j = i + 1; j < classToObservations.Count; j++ ) {
          var group1 = classToObservations[idToLabelMap[i]];
          var group2 = classToObservations[idToLabelMap[j]];
          var merged = group1.Select( entry => new BinaryObservation<TValue>( entry, -1 ) )
            .Concat( group2.Select( entry => new BinaryObservation<TValue>( entry, 1 ) ) );
          classifiers[i, j] = BinarySvmTrainer.Train( merged.ToArray(), k );
        }
      }

      return new MultiClassSvm<TValue, TLabel>( classifiers, idToLabelMap );
    }
  }
}
