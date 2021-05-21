using System;

namespace Svm.WordCount {
  /// <summary>
  /// A kernel for word counting scenarios.
  /// To use this kernel, you need to provide an implementation of <see cref="ICorpus"/>.
  /// </summary>
  public class WordCountKernel : IKernel<Document> {
    private readonly ICorpus _corpus;

    public WordCountKernel( ICorpus corpus ) {
      _corpus = corpus;
    }

    /// <inheritdoc />
    public double Compute( Document a, Document b ) {
      var i = 0;
      var j = 0;
      double result = 0;

      while ( i < a.Words.Count && j < b.Words.Count ) {
        var aWordOcc = a.Words[i];
        var bWordOcc = b.Words[j];

        if ( aWordOcc.WordId == bWordOcc.WordId ) {
          var documentFrequency = _corpus.GetDocumentFrequency( aWordOcc.WordId ) + 1;
          var inverseFrequency = Math.Log( (double) _corpus.CorpusSize / documentFrequency );
          var weight = inverseFrequency * inverseFrequency;
          result += aWordOcc.Frequency * bWordOcc.Frequency * weight;
          i++;
          j++;
        }
        else if ( aWordOcc.WordId < bWordOcc.WordId ) {
          i++;
        }
        else {
          j++;
        }
      }

      return result;
    }
  }
}
